#include "mcts_core.h"
#include <torch/cuda.h>
#include <ATen/Context.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <cstring> // For std::memcpy
#include <iomanip>

// Thread-local random number generator for mutex-free concurrent sampling
thread_local std::mt19937 rng(std::random_device{}());
std::uniform_real_distribution<float> dist(0.0f, 1.0f);

// ---------- BatchEvaluator ----------
BatchEvaluator::BatchEvaluator(const std::string &model_path, int batch_size, int obs_flat_size, std::shared_ptr<PerfMetrics> metrics, bool use_fp16)
    : batch_size(batch_size), obs_flat_size(obs_flat_size), metrics(metrics), use_fp16(use_fp16)
{
    // Enable CuDNN Benchmark to accelerate fixed-size CNN inference
    at::globalContext().setBenchmarkCuDNN(true);

    try
    {
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "[C++] CUDA detected. Loading model to GPU..." << std::endl;
        } else {
            std::cout << "[C++] CUDA not detected. Using CPU." << std::endl;
        }

        model = torch::jit::load(model_path, device);
        model.eval();
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading model from " << model_path << ": " << e.what() << "\n";
    }

    // Pre-allocate the CPU batch buffer (pinned memory for fast DMA on CUDA)
    auto dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
    if (device.is_cuda()) {
        batch_buffer = torch::zeros({batch_size, obs_flat_size}, torch::TensorOptions().dtype(dtype).pinned_memory(true));
    } else {
        batch_buffer = torch::zeros({batch_size, obs_flat_size}, torch::TensorOptions().dtype(dtype));
    }

    std::cout << "[C++] Pre-allocated batch buffer: [" << batch_size << ", " << obs_flat_size << "] "
              << (use_fp16 ? "FP16" : "FP32")
              << (device.is_cuda() ? " (pinned)" : "") << std::endl;

    worker_thread = std::thread(&BatchEvaluator::run_inference, this);
}

BatchEvaluator::~BatchEvaluator()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        stop_flag = true;
    }
    cv_batch.notify_all();
    if (worker_thread.joinable())
    {
        worker_thread.join();
    }
}

EvaluatorResult BatchEvaluator::evaluate(const float* obs_data)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::future<EvaluatorResult> future;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::promise<EvaluatorResult> promise;
        future = promise.get_future();
        // Push a cheap vector copy (~500 bytes) — no torch::Tensor allocation, no blocking
        queue.push({std::vector<float>(obs_data, obs_data + obs_flat_size), std::move(promise)});
        cv_batch.notify_one();
    }
    auto result = future.get();
    auto end = std::chrono::high_resolution_clock::now();
    metrics->mcts_eval_wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return result;
}

void BatchEvaluator::run_inference()
{
    // JIT & CuDNN Warmup
    try {
        torch::NoGradGuard no_grad;
        auto dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
        auto dummy_input = torch::zeros({batch_size, obs_flat_size}, torch::TensorOptions().dtype(dtype).device(device));
        model.forward({dummy_input});
        if (device.is_cuda()) {
            torch::cuda::synchronize();
        }
        std::cout << "[C++] Model Warmup Complete (Batch Size: " << batch_size
                  << ", " << (use_fp16 ? "FP16" : "FP32") << ")" << std::endl;
    } catch (...) {}

    while (true)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        int trigger_threshold = std::max(1, (int)(batch_size * 0.8));

        auto wait_start = std::chrono::high_resolution_clock::now();
        cv_batch.wait_for(lock, std::chrono::milliseconds(2), [this, trigger_threshold]
                          { return (int)queue.size() >= trigger_threshold || stop_flag; });
        auto wait_end = std::chrono::high_resolution_clock::now();
        metrics->wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();

        if (stop_flag && queue.empty())
            break;

        if (queue.empty())
            continue;

        int current_batch_size = std::min((int)queue.size(), batch_size);

        // Drain queue into pre-allocated buffer via memcpy (no torch::cat needed)
        auto cat_start = std::chrono::high_resolution_clock::now();
        std::vector<std::promise<EvaluatorResult>> batch_promises;
        batch_promises.reserve(current_batch_size);

        for (int i = 0; i < current_batch_size; ++i)
        {
            auto req = std::move(queue.front());
            queue.pop();

            if (use_fp16) {
                auto accessor = batch_buffer.accessor<at::Half, 2>();
                for (int j = 0; j < obs_flat_size; ++j) {
                    accessor[i][j] = static_cast<at::Half>(req.obs[j]);
                }
            } else {
                float* dest = batch_buffer.data_ptr<float>() + i * obs_flat_size;
                std::memcpy(dest, req.obs.data(), obs_flat_size * sizeof(float));
            }

            batch_promises.push_back(std::move(req.promise));
        }
        lock.unlock();

        // Transfer buffer to device (single contiguous copy, no torch::cat)
        torch::Tensor input = batch_buffer.to(device);
        auto cat_end = std::chrono::high_resolution_clock::now();
        metrics->cat_time_us += std::chrono::duration_cast<std::chrono::microseconds>(cat_end - cat_start).count();

        try {
            torch::NoGradGuard no_grad;

            auto f_start = std::chrono::high_resolution_clock::now();
            auto output = model.forward({input}).toTuple();
            auto f_end = std::chrono::high_resolution_clock::now();
            metrics->forward_time_us += std::chrono::duration_cast<std::chrono::microseconds>(f_end - f_start).count();

            auto p_start = std::chrono::high_resolution_clock::now();

            // 1. Get tensors (data still resides on GPU at this point)
            torch::Tensor policy_tensor = output->elements()[0].toTensor();
            torch::Tensor value_tensor = output->elements()[1].toTensor();

            // 2. Force FP16 to FP32 conversion on the GPU
            if (use_fp16 && policy_tensor.scalar_type() == torch::kHalf) {
                policy_tensor = policy_tensor.to(torch::kFloat32);
                value_tensor = value_tensor.to(torch::kFloat32);
            }

            // 3. Perform D2H memory transfer (transferring FP32 data)
            policy_tensor = policy_tensor.to(torch::kCPU).contiguous();
            value_tensor = value_tensor.to(torch::kCPU).contiguous();

            float* policy_ptr = policy_tensor.data_ptr<float>();
            float* value_ptr = value_tensor.data_ptr<float>();
            int num_actions = policy_tensor.size(1);

            for (int i = 0; i < current_batch_size; ++i)
            {
                EvaluatorResult res;
                res.value = value_ptr[i];
                
                // 4. Allocate exact memory for the vector and use low-level block copy 
                // instead of element-wise loop assignment
                res.policy.resize(num_actions);
                std::memcpy(res.policy.data(), policy_ptr + i * num_actions, num_actions * sizeof(float));
                
                batch_promises[i].set_value(std::move(res));
            }
            auto p_end = std::chrono::high_resolution_clock::now();
            metrics->parse_time_us += std::chrono::duration_cast<std::chrono::microseconds>(p_end - p_start).count();

            metrics->total_batches++;
            metrics->total_requests += current_batch_size;

        } catch (const c10::Error& e) {
            std::cerr << "LibTorch Error in run_inference: " << e.what() << std::endl;
            for (auto& p : batch_promises) p.set_value(EvaluatorResult());
            if (stop_flag) break;
        }
    }
}

// ---------- SelfPlayEngine ----------
SelfPlayEngine::SelfPlayEngine(const std::string& model_path, int batch_size, int obs_flat_size, int num_threads, int num_iters, float temperature, float c_puct, float dirichlet_alpha, float dirichlet_epsilon, bool use_fp16, bool use_undo)
    : obs_flat_size(obs_flat_size), num_threads(num_threads), num_iters(num_iters), temperature(temperature), c_puct(c_puct), dirichlet_alpha(dirichlet_alpha), dirichlet_epsilon(dirichlet_epsilon), use_undo(use_undo)
{
    metrics = std::make_shared<PerfMetrics>();
    evaluator = std::make_shared<BatchEvaluator>(model_path, batch_size, obs_flat_size, metrics, use_fp16);
}

SelfPlayEngine::~SelfPlayEngine() {}

void SelfPlayEngine::expand_node(Node *node, const open_spiel::State& state, const std::vector<float> &policy)
{
    if (node->is_expanded)
        return;

    std::vector<open_spiel::Action> legal_actions = state.LegalActions();

    for (open_spiel::Action action : legal_actions)
    {
        // Safety check to ensure action is within vector bounds
        float prob = (action >= 0 && action < static_cast<open_spiel::Action>(policy.size())) ? policy[action] : 0.0f;
        node->children[action] = std::make_unique<Node>(node, prob);
    }
    node->is_expanded = true;
}

std::pair<open_spiel::Action, Node *> SelfPlayEngine::select_best_child(Node *node, const std::vector<open_spiel::Action>& legal_actions)
{
    open_spiel::Action best_action = -1;
    Node *best_child = nullptr;
    float best_score = -1e9;

    float sqrt_parent_visits = std::sqrt(std::max(1.0f, (float)node->visit_count));

    for (open_spiel::Action action : legal_actions)
    {
        if (!node->children.count(action)) {
            node->children[action] = std::make_unique<Node>(node, 1.0f / std::max(1, (int)legal_actions.size()));
        }

        Node *child = node->children[action].get();

        float q_value = child->visit_count > 0 ? -child->mean_value : 0.0f;
        float u_value = c_puct * child->prior_prob * sqrt_parent_visits / (1.0f + child->visit_count);
        float score = q_value + u_value;

        if (score > best_score)
        {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }

    return {best_action, best_child};
}

void SelfPlayEngine::backpropagate(Node *node, float value)
{
    Node *cur = node;
    while (cur != nullptr)
    {
        cur->visit_count++;
        cur->total_value += value;
        cur->mean_value = cur->total_value / cur->visit_count;

        // [Fix Issue 3]: Conditional Negation
        if (cur->parent != nullptr && cur->parent->player_id != cur->player_id)
        {
            value = -value;
        }

        cur = cur->parent;
    }
}

void SelfPlayEngine::advance_chance_nodes(open_spiel::State* state, std::vector<std::pair<open_spiel::Player, open_spiel::Action>>* action_path)
{
    while (state->IsChanceNode() && !state->IsTerminal()) {
        auto outcomes = state->ChanceOutcomes();
        
        float r = dist(rng);
        float cumulative = 0.0f;
        open_spiel::Action sampled_action = outcomes[0].first;
        for (auto& outcome : outcomes) {
            cumulative += outcome.second;
            if (r <= cumulative) {
                sampled_action = outcome.first;
                break;
            }
        }
        
        if (action_path) {
            action_path->push_back({state->CurrentPlayer(), sampled_action});
        }
        state->ApplyAction(sampled_action);
    }
}

void SelfPlayEngine::run_mcts(Node *root, open_spiel::State& current_state)
{
    // Create a scratch state for simulation (clone once if using undo, clone per-iter otherwise)
    std::unique_ptr<open_spiel::State> sim_state;
    if (use_undo) {
        sim_state = current_state.Clone();
    }

    // [NEW]: Ensure root has the current player_id
    root->player_id = current_state.CurrentPlayer();

    for (int i = 0; i < num_iters; ++i)
    {
        // [FIX] Mathematical Early Stopping
        // It is mathematically impossible for the leader to be guaranteed before the halfway point.
        if (i > num_iters / 2 && i % 16 == 0) {
            int max_v = -1;
            int second_v = -1;
            for (auto const& [action, child] : root->children) {
                if (child->visit_count > max_v) {
                    second_v = max_v;
                    max_v = child->visit_count;
                } else if (child->visit_count > second_v) {
                    second_v = child->visit_count;
                }
            }
            // If condition met: even if all remaining simulations go to the runner-up,
            // the current winner will still have more visits.
            if (max_v > second_v + (num_iters - i)) {
                metrics->iters_saved += (num_iters - i);
                break;
            }
        }

        Node *cur_node = root;
        int current_depth = 0;

        if (use_undo) {
            // Undo-based path: apply actions in-place, track path for reversal
            std::vector<std::pair<open_spiel::Player, open_spiel::Action>> action_path;

            // Selection
            while (cur_node->is_expanded)
            {
                current_depth++;
                advance_chance_nodes(sim_state.get(), &action_path);
                if (sim_state->IsTerminal()) break;
                
                open_spiel::Player current_p = sim_state->CurrentPlayer();
                auto best = select_best_child(cur_node, sim_state->LegalActions());
                sim_state->ApplyAction(best.first);
                action_path.push_back({current_p, best.first});
                
                cur_node = best.second;
                // Synchronize player ID securely
                if (!sim_state->IsTerminal()) {
                    cur_node->player_id = sim_state->CurrentPlayer();
                } else {
                    cur_node->player_id = current_p; // Maintain alignment for terminal
                }
            }
            advance_chance_nodes(sim_state.get(), &action_path);

            // Evaluation & Expansion
            float value = 0.0f;
            if (sim_state->IsTerminal())
            {
                if (cur_node->player_id < 0) {
                    cur_node->player_id = cur_node->parent != nullptr ? cur_node->parent->player_id : 0;
                }
                value = sim_state->PlayerReturn(cur_node->player_id);
            }
            else
            {
                std::vector<float> obs_vec = sim_state->ObservationTensor();
                EvaluatorResult res = evaluator->evaluate(obs_vec.data());
                expand_node(cur_node, *sim_state, res.policy);
                value = res.value;
                cur_node->player_id = sim_state->CurrentPlayer();
            }

            backpropagate(cur_node, value);

            // Undo all actions to restore sim_state to root state
            for (auto it = action_path.rbegin(); it != action_path.rend(); ++it) {
                sim_state->UndoAction(it->first, it->second);
            }
        } else {
            // Clone-based path (original behavior)
            std::unique_ptr<open_spiel::State> cur_state = current_state.Clone();

            // Selection
            while (cur_node->is_expanded)
            {
                current_depth++;
                advance_chance_nodes(cur_state.get(), nullptr);
                if (cur_state->IsTerminal()) break;
                
                open_spiel::Player current_p = cur_state->CurrentPlayer();
                auto best = select_best_child(cur_node, cur_state->LegalActions());
                cur_state->ApplyAction(best.first);
                
                cur_node = best.second;
                if (!cur_state->IsTerminal()) {
                    cur_node->player_id = cur_state->CurrentPlayer();
                } else {
                    cur_node->player_id = current_p;
                }
            }
            advance_chance_nodes(cur_state.get(), nullptr);

            // Evaluation & Expansion
            float value = 0.0f;
            if (cur_state->IsTerminal())
            {
                if (cur_node->player_id < 0) {
                    cur_node->player_id = cur_node->parent != nullptr ? cur_node->parent->player_id : 0;
                }
                value = cur_state->PlayerReturn(cur_node->player_id);
            }
            else
            {
                std::vector<float> obs_vec = cur_state->ObservationTensor();
                EvaluatorResult res = evaluator->evaluate(obs_vec.data());
                expand_node(cur_node, *cur_state, res.policy);
                value = res.value;
                cur_node->player_id = cur_state->CurrentPlayer();
            }

            backpropagate(cur_node, value);
        }

        // Record statistics
        metrics->total_search_depth += current_depth;
        metrics->num_searches++;
        
        // Update maximum depth using atomic operations
        int current_max = metrics->max_search_depth.load();
        while (current_depth > current_max && 
               !metrics->max_search_depth.compare_exchange_weak(current_max, current_depth)) {}
    }
}

void SelfPlayEngine::play_game(const std::string& game_name, std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>>& all_trajectories, std::mutex& traj_mutex)
{
    auto game_start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_name);
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    
    advance_chance_nodes(state.get(), nullptr);
    
    std::unique_ptr<Node> root = std::make_unique<Node>(nullptr, 1.0f);
    std::vector<StepRecord> trajectory;

    while (!state->IsTerminal())
    {
        if (!root->is_expanded) {
            std::vector<float> obs_vec = state->ObservationTensor();
            EvaluatorResult res = evaluator->evaluate(obs_vec.data());

            // Standard expansion
            expand_node(root.get(), *state, res.policy);

            // [FIX] Apply Dirichlet Noise to the root's prior probabilities
            if (dirichlet_epsilon > 0.0f) {
                std::vector<open_spiel::Action> legal_actions = state->LegalActions();
                int n = legal_actions.size();
                if (n > 0) {
                    std::gamma_distribution<float> gamma_dist(dirichlet_alpha, 1.0f);
                    std::vector<float> noise(n);
                    float noise_sum = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        noise[j] = gamma_dist(rng);
                        noise_sum += noise[j];
                    }

                    for (int j = 0; j < n; ++j) {
                        open_spiel::Action a = legal_actions[j];
                        float n_val = noise[j] / (noise_sum + 1e-8f);
                        root->children[a]->prior_prob = (1.0f - dirichlet_epsilon) * root->children[a]->prior_prob + dirichlet_epsilon * n_val;
                    }
                }
            }
        }

        run_mcts(root.get(), *state);

        int num_actions = game->NumDistinctActions();
        std::vector<float> pi(num_actions, 0.0f);
        open_spiel::Action best_action = -1;
        
        std::vector<open_spiel::Action> current_legal_actions = state->LegalActions();
        absl::flat_hash_set<open_spiel::Action> legal_set(current_legal_actions.begin(), current_legal_actions.end());

        if (temperature <= 1e-3f)
        {
            // [Fix Issue 4]: Collect all maximally-visited actions and sample
            // uniformly among ties, avoiding hash-map-order bias.
            int max_visits = -1;
            std::vector<open_spiel::Action> best_actions;
            for (auto &pair : root->children) {
                if (!legal_set.count(pair.first)) continue;
                if (pair.second->visit_count > max_visits) {
                    max_visits = pair.second->visit_count;
                    best_actions.clear();
                    best_actions.push_back(pair.first);
                } else if (pair.second->visit_count == max_visits) {
                    best_actions.push_back(pair.first);
                }
            }
            if (!best_actions.empty()) {
                std::uniform_int_distribution<int> tie_dist(0, (int)best_actions.size() - 1);
                best_action = best_actions[tie_dist(rng)];
            }
            if (best_action != -1) pi[best_action] = 1.0f;
        }
        else
        {
            float sum = 0.0f;
            absl::flat_hash_map<open_spiel::Action, float> weights;
            for (auto &pair : root->children) {
                if (!legal_set.count(pair.first)) continue;
                float w = std::pow(pair.second->visit_count, 1.0f / temperature);
                weights[pair.first] = w;
                sum += w;
            }

            float r = dist(rng);
            float cumulative = 0.0f;
            for (auto &pair : weights) {
                float prob = sum > 0 ? (pair.second / sum) : (1.0f / weights.size());
                pi[pair.first] = prob;
                cumulative += prob;
                if (r <= cumulative && best_action == -1) best_action = pair.first;
            }
            if (best_action == -1 && !weights.empty()) best_action = weights.begin()->first;
        }
        
        if (best_action == -1 && !current_legal_actions.empty()) {
            best_action = current_legal_actions[0]; // fallback safely
            pi[best_action] = 1.0f;
        }

        StepRecord step;
        step.obs = state->ObservationTensor();
        step.pi = pi;
        step.player = state->CurrentPlayer();
        trajectory.push_back(step);

        state->ApplyAction(best_action);
        advance_chance_nodes(state.get(), nullptr);

        if (root->children.count(best_action)) {
            std::unique_ptr<Node> next_root = std::move(root->children[best_action]);
            next_root->parent = nullptr;
            root = std::move(next_root);
        } else {
            root = std::make_unique<Node>(nullptr, 1.0f);
        }
    }

    std::vector<double> returns = state->Returns();
    std::vector<std::tuple<std::vector<float>, std::vector<float>, float>> final_traj;
    for (const auto& step : trajectory) {
        final_traj.push_back({step.obs, step.pi, static_cast<float>(returns[step.player])});
    }

    std::lock_guard<std::mutex> lock(traj_mutex);
    all_trajectories.push_back(final_traj);

    auto game_end = std::chrono::high_resolution_clock::now();
    metrics->mcts_search_time_us += std::chrono::duration_cast<std::chrono::microseconds>(game_end - game_start).count();
}

py::list SelfPlayEngine::generate_games(int num_games, const std::string& game_name)
{
    metrics->reset();
    std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>> all_trajectories;
    std::mutex traj_mutex;

    auto start_total = std::chrono::high_resolution_clock::now();

    {
        py::gil_scoped_release release;
        std::vector<std::thread> threads;
        std::atomic<int> games_played{0};

        for (int i = 0; i < num_threads; ++i)
        {
            threads.emplace_back([&]() {
                while (true) {
                    int game_idx = games_played.fetch_add(1);
                    if (game_idx >= num_games) break;
                    play_game(game_name, all_trajectories, traj_mutex);
                }
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    long long total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();

    // Construction of results with GIL
    py::gil_scoped_acquire acquire;

    // Performance Summary Print
    double total_sec = total_us / 1000000.0;
    double search_sec = metrics->mcts_search_time_us / 1000000.0;
    double eval_wait_sec = metrics->mcts_eval_wait_time_us / 1000000.0;

    std::cout << "\n---------------- Performance Monitor ----------------" << std::endl;
    std::cout << "Total Games: " << num_games << " | Total Time: " << std::fixed << std::setprecision(2) << total_sec << "s" << std::endl;
    std::cout << "Games / Sec: " << num_games / total_sec << std::endl;
    std::cout << std::endl;
    std::cout << "[Inference Pipeline]" << std::endl;
    std::cout << "  Avg Batch Size: " << (metrics->total_batches > 0 ? (double)metrics->total_requests / metrics->total_batches : 0) << std::endl;
    std::cout << "  Queue Wait Time: " << (double)metrics->wait_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "  Tensor Cat Time: " << (double)metrics->cat_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "  Model Forward:   " << (double)metrics->forward_time_us / 1000.0 << " ms" << std::endl;
    std::cout << "  Data Parsing:    " << (double)metrics->parse_time_us / 1000.0 << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "[MCTS Worker Stats (Aggregated across " << num_threads << " threads)]" << std::endl;
    std::cout << "  MCTS Logic Time: " << (search_sec - eval_wait_sec) << " s" << std::endl;
    std::cout << "  Evaluator Wait:  " << eval_wait_sec << " s" << std::endl;
    std::cout << "  Iterations Saved:" << metrics->iters_saved.load() << std::endl;
    std::cout << "-----------------------------------------------------\n" << std::endl;

    py::list py_all_trajectories;
    for (const auto& traj : all_trajectories) {
        py::list py_traj;
        for (const auto& step : traj) {
            py::array_t<float> py_obs(obs_flat_size);
            py::array_t<float> py_pi(static_cast<int>(std::get<1>(step).size()));
            std::memcpy(py_obs.mutable_data(), std::get<0>(step).data(), std::get<0>(step).size() * sizeof(float));
            std::memcpy(py_pi.mutable_data(), std::get<1>(step).data(), std::get<1>(step).size() * sizeof(float));
            py_traj.append(py::make_tuple(py_obs, py_pi, std::get<2>(step)));
        }
        py_all_trajectories.append(py_traj);
    }
    return py_all_trajectories;
}

py::dict SelfPlayEngine::get_metrics() {
    py::dict d;
    double avg_depth = metrics->num_searches > 0 ? 
                       (double)metrics->total_search_depth / metrics->num_searches : 0.0;
    d["avg_search_depth"] = avg_depth;
    d["max_search_depth"] = (double)metrics->max_search_depth.load();
    return d;
}

// Connect Four board constants (open_spiel ObservationTensor layout: 3 planes x 6 rows x 7 cols)
// Plane 0: current player pieces, Plane 1: other player pieces, Plane 2: empty
static constexpr int CF_ROWS = 6;
static constexpr int CF_COLS = 7;
static constexpr int CF_PLANES = 3;

// Extract a cell from the flat observation tensor.
// Returns +1 if 'player' occupies [r][c], -1 if opponent, 0 if empty.
static inline int CF_Cell(const std::vector<float>& obs, int plane, int row, int col) {
    int idx = plane * CF_ROWS * CF_COLS + row * CF_COLS + col;
    return (obs[idx] > 0.5f) ? 1 : 0;
}

// Score a window of 4 cells for a given player (returns heuristic contribution).
static int CF_ScoreWindow(int p_count, int opp_count, int empty_count) {
    if (p_count == 4)                          return 1000;
    if (p_count == 3 && empty_count == 1)      return 10;
    if (p_count == 2 && empty_count == 2)      return 2;
    if (opp_count == 3 && empty_count == 1)    return -80;
    return 0;
}// Full Connect Four board heuristic from an observation tensor.
// player=0 means the perspective of the current player (plane 0).
static float EvaluateConnectFour(const std::vector<float>& obs) {
    // Build a 2D board function: +1 = current player, -1 = opponent, 0 = empty
    auto cell = [&](int r, int c) -> int {
        if (CF_Cell(obs, 0, r, c)) return +1;
        if (CF_Cell(obs, 1, r, c)) return -1;
        return 0;
    };

    float score = 0.0f;

    // Centre column preference
    int centre_col = CF_COLS / 2;
    for (int r = 0; r < CF_ROWS; ++r)
        if (cell(r, centre_col) == +1) score += 3.0f;

    // score_window: returns the window's heuristic value (no mutation of outer scope)
    auto score_window = [&](int r0, int c0, int dr, int dc) -> int {
        int p_cnt = 0, opp_cnt = 0, emp_cnt = 0;
        for (int k = 0; k < 4; ++k) {
            int v = cell(r0 + k*dr, c0 + k*dc);
            if (v == +1) p_cnt++;
            else if (v == -1) opp_cnt++;
            else emp_cnt++;
        }
        return CF_ScoreWindow(p_cnt, opp_cnt, emp_cnt);
    };

    // Horizontal
    for (int r = 0; r < CF_ROWS; ++r)
        for (int c = 0; c <= CF_COLS - 4; ++c)
            score += score_window(r, c, 0, 1);
    // Vertical
    for (int r = 0; r <= CF_ROWS - 4; ++r)
        for (int c = 0; c < CF_COLS; ++c)
            score += score_window(r, c, 1, 0);
    // Diagonal /
    for (int r = 3; r < CF_ROWS; ++r)
        for (int c = 0; c <= CF_COLS - 4; ++c)
            score += score_window(r, c, -1, 1);
    // Diagonal backslash
    for (int r = 0; r <= CF_ROWS - 4; ++r)
        for (int c = 0; c <= CF_COLS - 4; ++c)
            score += score_window(r, c, 1, 1);

    return score;
}

// Minimax with alpha-beta pruning. Only implemented for connect_four.
// val is from the perspective of the *root* player (maximising player).
static float AlphaBeta(
    const open_spiel::State& state,
    int depth, float alpha, float beta,
    bool maximising,
    open_spiel::Player root_player)
{
    if (state.IsTerminal()) {
        float ret = state.PlayerReturn(root_player);
        // Scale terminal returns to large values so they dominate heuristic scores
        return ret * 1e6f;
    }
    if (depth == 0) {
        // Leaf evaluation: get the observation tensor for the current player
        std::vector<float> obs = state.ObservationTensor(root_player);
        // If it's not the root player's turn, flip perspective
        float h = EvaluateConnectFour(obs);
        return maximising ? h : -h;
    }

    std::vector<open_spiel::Action> actions = state.LegalActions();
    if (actions.empty()) return 0.0f;

    if (maximising) {
        float val = -1e9f;
        for (auto a : actions) {
            auto child = state.Clone();
            child->ApplyAction(a);
            val = std::max(val, AlphaBeta(*child, depth - 1, alpha, beta, false, root_player));
            alpha = std::max(alpha, val);
            if (alpha >= beta) break; // beta cut-off
        }
        return val;
    } else {
        float val = 1e9f;
        for (auto a : actions) {
            auto child = state.Clone();
            child->ApplyAction(a);
            val = std::min(val, AlphaBeta(*child, depth - 1, alpha, beta, true, root_player));
            beta = std::min(beta, val);
            if (beta <= alpha) break; // alpha cut-off
        }
        return val;
    }
}

open_spiel::Action GetMinimaxAction(open_spiel::State& state, const std::string& game_name, int depth) {
    // Minimax is only implemented for Connect Four
    if (game_name != "connect_four") {
        // Fallback to greedy for other games
        return GetGreedyAction(state, game_name);
    }

    std::vector<open_spiel::Action> actions = state.LegalActions();
    if (actions.empty()) return open_spiel::kInvalidAction;

    open_spiel::Player root_player = state.CurrentPlayer();
    open_spiel::Action best_action = actions[0];
    float best_val = -1e9f;

    for (auto a : actions) {
        auto child = state.Clone();
        child->ApplyAction(a);
        // After applying our move we are in the minimising position for the opponent
        float val = AlphaBeta(*child, depth - 1, -1e9f, 1e9f, false, root_player);
        if (val > best_val) {
            best_val = val;
            best_action = a;
        }
    }
    return best_action;
}

float EvaluateStateGreedy(const open_spiel::State& state, const std::string& game_name, open_spiel::Player player) {
    if (state.IsTerminal()) {
        return state.PlayerReturn(player);
    }
    
    if (game_name == "connect_four") {
        // Use the real Connect Four heuristic from the perspective of `player`
        std::vector<float> obs = state.ObservationTensor(player);
        return EvaluateConnectFour(obs);
    } else if (game_name == "backgammon") {
        std::vector<float> obs = state.ObservationTensor(player);
        
        // OpenSpiel Backgammon observation tensor feature indices:
        // 192: Number of checkers for Player 0 on the Bar
        // 193: Number of checkers for Player 0 borne off (normalised: count / 15)
        // 195: Number of checkers for Player 1 on the Bar
        // 196: Number of checkers for Player 1 borne off (normalised: count / 15)
        
        float p0_score = obs[193] * 10.0f - obs[192] * 2.0f;
        float p1_score = obs[196] * 10.0f - obs[195] * 2.0f;
        
        if (player == 0) {
            return p0_score - p1_score;
        } else {
            return p1_score - p0_score;
        }
    }
    return 0.0f;
}

open_spiel::Action GetGreedyAction(open_spiel::State& state, const std::string& game_name) {
    std::vector<open_spiel::Action> legal_actions = state.LegalActions();
    if (legal_actions.empty()) return open_spiel::kInvalidAction;

    // Backgammon can generate up to ~1352 legal action combinations per turn (each
    // "action" encodes a full multi-pip move sequence for both dice). Cloning state
    // for every candidate is the dominant tournament cost. We cap evaluation to a
    // random sample so per-move work is O(kMaxGreedyEvals) regardless of branching.
    constexpr int kMaxGreedyEvals = 30;
    if ((int)legal_actions.size() > kMaxGreedyEvals) {
        // Shuffle in-place using the thread-local RNG, then truncate.
        std::shuffle(legal_actions.begin(), legal_actions.end(), rng);
        legal_actions.resize(kMaxGreedyEvals);
    }

    open_spiel::Player player = state.CurrentPlayer();
    open_spiel::Action best_action = legal_actions[0];
    float best_val = -1e9f;
    
    for (open_spiel::Action a : legal_actions) {
        std::unique_ptr<open_spiel::State> next_state = state.Clone();
        next_state->ApplyAction(a);
        
        if (next_state->IsTerminal()) {
            float ret = next_state->PlayerReturn(player);
            if (ret > 0) return a; 
            if (ret > best_val) {
                best_val = ret;
                best_action = a;
            }
            continue;
        }
        
        float val = EvaluateStateGreedy(*next_state, game_name, player);
        if (val > best_val) {
            best_val = val;
            best_action = a;
        }
    }
    
    return best_action;
}

TournamentEngine::TournamentEngine(const std::string& model_path, int batch_size, int obs_flat_size, int num_threads, int num_iters, float temperature, float c_puct, bool use_fp16, bool use_undo, int opening_temp_moves)
    : obs_flat_size(obs_flat_size), num_threads(num_threads), num_iters(num_iters), temperature(temperature), c_puct(c_puct), use_undo(use_undo), opening_temp_moves(opening_temp_moves)
{
    metrics = std::make_shared<PerfMetrics>();
    evaluator = std::make_shared<BatchEvaluator>(model_path, batch_size, obs_flat_size, metrics, use_fp16);
}

TournamentEngine::~TournamentEngine() {}

void TournamentEngine::expand_node(Node *node, const open_spiel::State& state, const std::vector<float> &policy)
{
    if (node->is_expanded) return;

    std::vector<open_spiel::Action> legal_actions = state.LegalActions();
    for (open_spiel::Action action : legal_actions) {
        float prob = (action >= 0 && action < static_cast<open_spiel::Action>(policy.size())) ? policy[action] : 0.0f;
        node->children[action] = std::make_unique<Node>(node, prob);
    }
    node->is_expanded = true;
}

std::pair<open_spiel::Action, Node *> TournamentEngine::select_best_child(Node *node, const std::vector<open_spiel::Action>& legal_actions)
{
    open_spiel::Action best_action = -1;
    Node *best_child = nullptr;
    float best_score = -1e9;
    float sqrt_parent_visits = std::sqrt(std::max(1.0f, (float)node->visit_count));

    for (open_spiel::Action action : legal_actions) {
        if (!node->children.count(action)) {
            node->children[action] = std::make_unique<Node>(node, 1.0f / std::max(1, (int)legal_actions.size()));
        }

        Node *child = node->children[action].get();
        float q_value = child->visit_count > 0 ? -child->mean_value : 0.0f;
        float u_value = c_puct * child->prior_prob * sqrt_parent_visits / (1.0f + child->visit_count);
        float score = q_value + u_value;

        if (score > best_score) {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }
    return {best_action, best_child};
}

void TournamentEngine::backpropagate(Node *node, float value)
{
    Node *cur = node;
    while (cur != nullptr) {
        cur->visit_count++;
        cur->total_value += value;
        cur->mean_value = cur->total_value / cur->visit_count;
        if (cur->parent != nullptr && cur->parent->player_id != cur->player_id) {
            value = -value;
        }
        cur = cur->parent;
    }
}

void TournamentEngine::advance_chance_nodes(open_spiel::State* state, std::vector<std::pair<open_spiel::Player, open_spiel::Action>>* action_path)
{
    while (state->IsChanceNode() && !state->IsTerminal()) {
        auto outcomes = state->ChanceOutcomes();
        float r = dist(rng);
        float cumulative = 0.0f;
        open_spiel::Action sampled_action = outcomes[0].first;
        for (auto& outcome : outcomes) {
            cumulative += outcome.second;
            if (r <= cumulative) {
                sampled_action = outcome.first;
                break;
            }
        }
        if (action_path) action_path->push_back({state->CurrentPlayer(), sampled_action});
        state->ApplyAction(sampled_action);
    }
}

void TournamentEngine::run_mcts(Node *root, open_spiel::State& current_state)
{
    std::unique_ptr<open_spiel::State> sim_state;
    if (use_undo) {
        sim_state = current_state.Clone();
    }

    root->player_id = current_state.CurrentPlayer();

    for (int i = 0; i < num_iters; ++i) {
        if (i > num_iters / 2 && i % 16 == 0) {
            int max_v = -1, second_v = -1;
            for (auto const& [action, child] : root->children) {
                if (child->visit_count > max_v) {
                    second_v = max_v;
                    max_v = child->visit_count;
                } else if (child->visit_count > second_v) {
                    second_v = child->visit_count;
                }
            }
            if (max_v > second_v + (num_iters - i)) {
                metrics->iters_saved += (num_iters - i);
                break;
            }
        }

        Node *cur_node = root;
        int current_depth = 0;

        if (use_undo) {
            std::vector<std::pair<open_spiel::Player, open_spiel::Action>> action_path;
            
            while (cur_node->is_expanded) {
                current_depth++;
                advance_chance_nodes(sim_state.get(), &action_path);
                if (sim_state->IsTerminal()) break;
                
                open_spiel::Player current_p = sim_state->CurrentPlayer();
                auto best = select_best_child(cur_node, sim_state->LegalActions());
                sim_state->ApplyAction(best.first);
                action_path.push_back({current_p, best.first});
                
                cur_node = best.second;
                if (!sim_state->IsTerminal()) {
                    cur_node->player_id = sim_state->CurrentPlayer();
                } else {
                    cur_node->player_id = current_p;
                }
            }
            advance_chance_nodes(sim_state.get(), &action_path);

            float value = 0.0f;
            if (sim_state->IsTerminal()) {
                if (cur_node->player_id < 0) {
                    cur_node->player_id = cur_node->parent != nullptr ? cur_node->parent->player_id : 0;
                }
                value = sim_state->PlayerReturn(cur_node->player_id);
            } else {
                std::vector<float> obs_vec = sim_state->ObservationTensor();
                EvaluatorResult res = evaluator->evaluate(obs_vec.data());
                expand_node(cur_node, *sim_state, res.policy);
                value = res.value;
                cur_node->player_id = sim_state->CurrentPlayer();
            }

            backpropagate(cur_node, value);

            for (auto it = action_path.rbegin(); it != action_path.rend(); ++it) {
                sim_state->UndoAction(it->first, it->second);
            }
        } else {
            std::unique_ptr<open_spiel::State> cur_state = current_state.Clone();
            while (cur_node->is_expanded) {
                current_depth++;
                advance_chance_nodes(cur_state.get(), nullptr);
                if (cur_state->IsTerminal()) break;
                
                open_spiel::Player current_p = cur_state->CurrentPlayer();
                auto best = select_best_child(cur_node, cur_state->LegalActions());
                cur_state->ApplyAction(best.first);
                
                cur_node = best.second;
                if (!cur_state->IsTerminal()) {
                    cur_node->player_id = cur_state->CurrentPlayer();
                } else {
                    cur_node->player_id = current_p;
                }
            }
            advance_chance_nodes(cur_state.get(), nullptr);

            float value = 0.0f;
            if (cur_state->IsTerminal()) {
                if (cur_node->player_id < 0) {
                    cur_node->player_id = cur_node->parent != nullptr ? cur_node->parent->player_id : 0;
                }
                value = cur_state->PlayerReturn(cur_node->player_id);
            } else {
                std::vector<float> obs_vec = cur_state->ObservationTensor();
                EvaluatorResult res = evaluator->evaluate(obs_vec.data());
                expand_node(cur_node, *cur_state, res.policy);
                value = res.value;
                cur_node->player_id = cur_state->CurrentPlayer();
            }

            backpropagate(cur_node, value);
        }
        
        metrics->total_search_depth += current_depth;
        metrics->num_searches++;
        
        int current_max = metrics->max_search_depth.load();
        while (current_depth > current_max && 
               !metrics->max_search_depth.compare_exchange_weak(current_max, current_depth)) {}
    }
}

void TournamentEngine::play_match(const std::string& game_name, const std::string& opponent, int model_player, std::atomic<int>& wins, std::atomic<int>& losses, std::atomic<int>& draws, std::atomic<long long>& total_moves)
{
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_name);
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    
    advance_chance_nodes(state.get(), nullptr);

    int move_count = 0;
    while (!state->IsTerminal()) {
        open_spiel::Player current_p = state->CurrentPlayer();
        open_spiel::Action best_action = -1;
        
        if (current_p == model_player) {
            std::unique_ptr<Node> root = std::make_unique<Node>(nullptr, 1.0f);
            std::vector<float> obs_vec = state->ObservationTensor();
            EvaluatorResult res = evaluator->evaluate(obs_vec.data());
            expand_node(root.get(), *state, res.policy);
            
            run_mcts(root.get(), *state);
            
            std::vector<open_spiel::Action> legal_actions = state->LegalActions();
            absl::flat_hash_set<open_spiel::Action> legal_set(legal_actions.begin(), legal_actions.end());

            // [Fix Issue 3]: Use temperature=1 (visit-count proportional sampling) for the
            // first opening_temp_moves game plies to diversify openings under deterministic eval.
            float effective_temp = (opening_temp_moves > 0 && move_count < opening_temp_moves)
                                   ? 1.0f : temperature;

            if (effective_temp <= 1e-3f) {
                // [Fix Issue 4]: Collect all maximally-visited actions and sample
                // uniformly among ties, avoiding hash-map-order bias.
                int max_visits = -1;
                std::vector<open_spiel::Action> best_actions;
                for (auto &pair : root->children) {
                    if (!legal_set.count(pair.first)) continue;
                    if (pair.second->visit_count > max_visits) {
                        max_visits = pair.second->visit_count;
                        best_actions.clear();
                        best_actions.push_back(pair.first);
                    } else if (pair.second->visit_count == max_visits) {
                        best_actions.push_back(pair.first);
                    }
                }
                if (!best_actions.empty()) {
                    std::uniform_int_distribution<int> tie_dist(0, (int)best_actions.size() - 1);
                    best_action = best_actions[tie_dist(rng)];
                }
            } else {
                float sum = 0.0f;
                absl::flat_hash_map<open_spiel::Action, float> weights;
                for (auto &pair : root->children) {
                    if (!legal_set.count(pair.first)) continue;
                    float w = std::pow(pair.second->visit_count, 1.0f / effective_temp);
                    weights[pair.first] = w;
                    sum += w;
                }

                float r = dist(rng);
                float cumulative = 0.0f;
                for (auto &pair : weights) {
                    float prob = sum > 0 ? (pair.second / sum) : (1.0f / weights.size());
                    cumulative += prob;
                    if (r <= cumulative && best_action == -1) best_action = pair.first;
                }
                if (best_action == -1 && !weights.empty()) best_action = weights.begin()->first;
            }
            if (best_action == -1 && !legal_actions.empty()) best_action = legal_actions[0];
        } else {
            // Opponent Turn
            if (opponent == "greedy") {
                best_action = GetGreedyAction(*state, game_name);
            } else if (opponent == "minimax") {
                // Default depth=4 for minimax; fast enough with alpha-beta on Connect Four
                best_action = GetMinimaxAction(*state, game_name, /*depth=*/4);
            } else if (opponent == "random") {
                std::vector<open_spiel::Action> legal_actions = state->LegalActions();
                if (!legal_actions.empty()) {
                    std::uniform_int_distribution<int> distrib(0, legal_actions.size() - 1);
                    best_action = legal_actions[distrib(rng)];
                }
            } else {
                // Unknown opponent: random fallback
                std::vector<open_spiel::Action> legal_actions = state->LegalActions();
                if (!legal_actions.empty()) {
                    std::uniform_int_distribution<int> distrib(0, legal_actions.size() - 1);
                    best_action = legal_actions[distrib(rng)];
                }
            }
        }
        
        if (best_action != -1) {
            state->ApplyAction(best_action);
            move_count++;
        }
        advance_chance_nodes(state.get(), nullptr);
    }

    std::vector<double> returns = state->Returns();
    if (returns[model_player] > 0) wins++;
    else if (returns[model_player] < 0) losses++;
    else draws++;
    total_moves += move_count;
}

py::dict TournamentEngine::play_tournament(int num_games, const std::string& game_name, const std::string& opponent)
{
    metrics->reset();
    std::atomic<int> wins{0}, losses{0}, draws{0};
    std::atomic<long long> total_moves{0};

    auto start_total = std::chrono::high_resolution_clock::now();

    {
        py::gil_scoped_release release;
        std::vector<std::thread> threads;
        std::atomic<int> games_played{0};

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                while (true) {
                    int game_idx = games_played.fetch_add(1);
                    if (game_idx >= num_games) break;
                    int model_player = game_idx % 2; // Alternate who goes first
                    play_match(game_name, opponent, model_player, wins, losses, draws, total_moves);
                }
            });
        }
        for (auto& t : threads) if (t.joinable()) t.join();
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    long long total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_total - start_total).count();
    double total_sec = total_us / 1e6;

    int w = wins.load(), l = losses.load(), d = draws.load();
    double avg_game_len = num_games > 0 ? (double)total_moves.load() / num_games : 0.0;
    double avg_batch = metrics->total_batches > 0 ? (double)metrics->total_requests / metrics->total_batches : 0.0;

    std::cout << "\n------------ Tournament Performance Monitor ------------" << std::endl;
    std::cout << "Opponent: " << opponent << " | Games: " << num_games
              << " | Threads: " << num_threads << " | Iters: " << num_iters << std::endl;
    std::cout << "Total Time: " << std::fixed << std::setprecision(2) << total_sec
              << "s  |  Games/sec: " << num_games / total_sec << std::endl;
    std::cout << "Results: W=" << w << " L=" << l << " D=" << d
              << "  |  Win Rate: " << std::setprecision(1) << 100.0 * w / std::max(1, num_games) << "%" << std::endl;
    std::cout << "Avg Game Length: " << std::setprecision(1) << avg_game_len << " moves" << std::endl;
    std::cout << "[Inference] Avg Batch: " << std::setprecision(2) << avg_batch
              << "  |  Model Forward: " << metrics->forward_time_us / 1000.0 << " ms"
              << "  |  Iters Saved: " << metrics->iters_saved.load() << std::endl;
    std::cout << "-------------------------------------------------------\n" << std::endl;

    py::gil_scoped_acquire acquire;
    py::dict results;
    results["wins"]           = w;
    results["losses"]         = l;
    results["draws"]          = d;
    results["total_time_s"]   = total_sec;
    results["games_per_sec"]  = num_games / total_sec;
    results["avg_game_length"]= avg_game_len;
    results["avg_batch_size"] = avg_batch;
    results["iters_saved"]    = (long long)metrics->iters_saved.load();
    results["avg_mcts_depth"] = metrics->num_searches > 0
                                  ? (double)metrics->total_search_depth / metrics->num_searches
                                  : 0.0;
    return results;
}
