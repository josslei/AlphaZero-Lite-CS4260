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

    // Pre-allocate the CPU pinned-memory batch buffer
    auto dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
    if (device.is_cuda()) {
        batch_buffer = torch::zeros({batch_size, obs_flat_size}, torch::TensorOptions().dtype(dtype).pinned_memory(true));
    } else {
        batch_buffer = torch::zeros({batch_size, obs_flat_size}, torch::TensorOptions().dtype(dtype));
    }
    slot_promises.resize(batch_size);

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
        std::unique_lock<std::mutex> lock(m_mutex);

        // Wait if all slots are full (inference thread hasn't consumed them yet)
        cv_slot.wait(lock, [this] { return next_slot < batch_size || stop_flag; });
        if (stop_flag) {
            return EvaluatorResult();
        }

        int slot = next_slot++;

        // Copy observation data directly into the pre-allocated buffer row
        if (use_fp16) {
            // Convert float32 input to float16 and write into the buffer
            auto accessor = batch_buffer.accessor<at::Half, 2>();
            for (int j = 0; j < obs_flat_size; ++j) {
                accessor[slot][j] = static_cast<at::Half>(obs_data[j]);
            }
        } else {
            float* dest = batch_buffer.data_ptr<float>() + slot * obs_flat_size;
            std::memcpy(dest, obs_data, obs_flat_size * sizeof(float));
        }

        std::promise<EvaluatorResult> promise;
        future = promise.get_future();
        slot_promises[slot] = std::move(promise);

        int trigger_threshold = std::max(1, (int)(batch_size * 0.8));
        if (next_slot >= trigger_threshold) {
            cv_batch.notify_one();
        }
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
                          { return next_slot >= trigger_threshold || stop_flag; });
        auto wait_end = std::chrono::high_resolution_clock::now();
        metrics->wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();

        if (stop_flag && next_slot == 0)
            break;

        if (next_slot == 0)
            continue;

        int current_batch_size = next_slot;

        // Collect promises for this batch
        std::vector<std::promise<EvaluatorResult>> batch_promises(current_batch_size);
        for (int i = 0; i < current_batch_size; ++i) {
            batch_promises[i] = std::move(slot_promises[i]);
        }

        // Reset slot counter so new requests can start filling
        next_slot = 0;
        lock.unlock();
        cv_slot.notify_all();  // Wake up any threads waiting for slots

        // Prepare input: slice the pre-allocated buffer (zero-copy view) and transfer to device
        auto cat_start = std::chrono::high_resolution_clock::now();
        torch::Tensor input;
        if (current_batch_size < batch_size) {
            // Pad to fixed batch size for consistent CuDNN performance
            input = batch_buffer.to(device, /*non_blocking=*/true);
        } else {
            input = batch_buffer.to(device, /*non_blocking=*/true);
        }
        auto cat_end = std::chrono::high_resolution_clock::now();
        metrics->cat_time_us += std::chrono::duration_cast<std::chrono::microseconds>(cat_end - cat_start).count();

        try {
            torch::NoGradGuard no_grad;

            auto f_start = std::chrono::high_resolution_clock::now();
            auto output = model.forward({input}).toTuple();
            auto f_end = std::chrono::high_resolution_clock::now();
            metrics->forward_time_us += std::chrono::duration_cast<std::chrono::microseconds>(f_end - f_start).count();

            auto p_start = std::chrono::high_resolution_clock::now();
            // Always cast output to FP32 for CPU-side parsing
            torch::Tensor policy_tensor = output->elements()[0].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
            torch::Tensor value_tensor = output->elements()[1].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();

            float* policy_ptr = policy_tensor.data_ptr<float>();
            float* value_ptr = value_tensor.data_ptr<float>();
            int num_actions = policy_tensor.size(1);

            for (int i = 0; i < current_batch_size; ++i)
            {
                EvaluatorResult res;
                res.value = value_ptr[i];
                for (int a = 0; a < num_actions; ++a)
                {
                    res.policy[a] = policy_ptr[i * num_actions + a];
                }
                batch_promises[i].set_value(res);
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

void SelfPlayEngine::expand_node(Node *node, const open_spiel::State& state, const absl::flat_hash_map<open_spiel::Action, float> &policy)
{
    if (node->is_expanded)
        return;

    std::vector<open_spiel::Action> legal_actions = state.LegalActions();

    for (open_spiel::Action action : legal_actions)
    {
        auto it = policy.find(action);
        float prob = (it != policy.end()) ? it->second : 0.0f;
        node->children[action] = std::make_unique<Node>(node, prob);
    }
    node->is_expanded = true;
}

std::pair<open_spiel::Action, Node *> SelfPlayEngine::select_best_child(Node *node)
{
    open_spiel::Action best_action = -1;
    Node *best_child = nullptr;
    float best_score = -1e9;

    float sqrt_parent_visits = std::sqrt(std::max(1.0f, (float)node->visit_count));

    for (auto &pair : node->children)
    {
        open_spiel::Action action = pair.first;
        Node *child = pair.second.get();

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

        cur = cur->parent;
        value = -value;
    }
}

void SelfPlayEngine::run_mcts(Node *root, open_spiel::State& current_state)
{
    // Create a scratch state for simulation (clone once if using undo, clone per-iter otherwise)
    std::unique_ptr<open_spiel::State> sim_state;
    if (use_undo) {
        sim_state = current_state.Clone();
    }

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
        open_spiel::Player last_player = open_spiel::kInvalidPlayer;

        if (use_undo) {
            // Undo-based path: apply actions in-place, track path for reversal
            std::vector<std::pair<open_spiel::Player, open_spiel::Action>> action_path;

            // Selection
            while (cur_node->is_expanded)
            {
                if (sim_state->IsTerminal()) break;
                last_player = sim_state->CurrentPlayer();
                auto best = select_best_child(cur_node);
                sim_state->ApplyAction(best.first);
                action_path.push_back({last_player, best.first});
                cur_node = best.second;
            }

            // Evaluation & Expansion
            float value = 0.0f;
            if (sim_state->IsTerminal())
            {
                if (last_player != open_spiel::kInvalidPlayer) {
                    int next_player = 1 - last_player;
                    value = sim_state->PlayerReturn(next_player);
                }
            }
            else
            {
                std::vector<float> obs_vec = sim_state->ObservationTensor();
                EvaluatorResult res = evaluator->evaluate(obs_vec.data());
                expand_node(cur_node, *sim_state, res.policy);
                value = res.value;
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
                if (cur_state->IsTerminal()) break;
                last_player = cur_state->CurrentPlayer();
                auto best = select_best_child(cur_node);
                cur_state->ApplyAction(best.first);
                cur_node = best.second;
            }

            // Evaluation & Expansion
            float value = 0.0f;
            if (cur_state->IsTerminal())
            {
                if (last_player != open_spiel::kInvalidPlayer) {
                    int next_player = 1 - last_player;
                    value = cur_state->PlayerReturn(next_player);
                }
            }
            else
            {
                std::vector<float> obs_vec = cur_state->ObservationTensor();
                EvaluatorResult res = evaluator->evaluate(obs_vec.data());
                expand_node(cur_node, *cur_state, res.policy);
                value = res.value;
            }

            backpropagate(cur_node, value);
        }
    }
}

void SelfPlayEngine::play_game(const std::string& game_name, std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>>& all_trajectories, std::mutex& traj_mutex)
{
    auto game_start = std::chrono::high_resolution_clock::now();

    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_name);
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
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

        if (temperature <= 1e-3f)
        {
            int max_visits = -1;
            for (auto &pair : root->children) {
                if (pair.second->visit_count > max_visits) {
                    max_visits = pair.second->visit_count;
                    best_action = pair.first;
                }
            }
            if (best_action != -1) pi[best_action] = 1.0f;
        }
        else
        {
            float sum = 0.0f;
            absl::flat_hash_map<open_spiel::Action, float> weights;
            for (auto &pair : root->children) {
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

        StepRecord step;
        step.obs = state->ObservationTensor();
        step.pi = pi;
        step.player = state->CurrentPlayer();
        trajectory.push_back(step);

        state->ApplyAction(best_action);

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
