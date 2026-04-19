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
BatchEvaluator::BatchEvaluator(const std::string &model_path, int batch_size, std::shared_ptr<PerfMetrics> metrics)
    : batch_size(batch_size), metrics(metrics)
{
    // Enable CuDNN Benchmark to accelerate fixed-size CNN inference
    at::globalContext().setBenchmarkCuDNN(true);

    try
    {
        torch::Device device(torch::kCPU);
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

EvaluatorResult BatchEvaluator::evaluate(const torch::Tensor &state)
{
    auto start = std::chrono::high_resolution_clock::now();
    std::future<EvaluatorResult> future;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::promise<EvaluatorResult> promise;
        future = promise.get_future();
        queue.push({state, std::move(promise)});
        cv_batch.notify_one(); 
    }
    auto result = future.get();
    auto end = std::chrono::high_resolution_clock::now();
    metrics->mcts_eval_wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return result;
}

void BatchEvaluator::run_inference()
{
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    // JIT & CuDNN Warmup
    try {
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({batch_size, 3, 6, 7}).to(device);
        model.forward({dummy_input});
        if (device.is_cuda()) {
            torch::cuda::synchronize();
        }
        std::cout << "[C++] Model Warmup Complete (Batch Size: " << batch_size << ")" << std::endl;
    } catch (...) {}

    while (true)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        int trigger_threshold = std::max(1, (int)(batch_size * 0.8));
        
        auto wait_start = std::chrono::high_resolution_clock::now();
        cv_batch.wait_for(lock, std::chrono::milliseconds(2), [this, trigger_threshold]
                          { return queue.size() >= trigger_threshold || stop_flag; });
        auto wait_end = std::chrono::high_resolution_clock::now();
        metrics->wait_time_us += std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();

        if (stop_flag && queue.empty())
            break;

        if (queue.empty())
            continue;

        int current_batch_size = std::min((int)queue.size(), batch_size);
        std::vector<torch::Tensor> batch_tensors;
        std::vector<std::promise<EvaluatorResult>> batch_promises;

        for (int i = 0; i < current_batch_size; ++i)
        {
            auto req = std::move(queue.front());
            queue.pop();
            batch_tensors.push_back(req.state);
            batch_promises.push_back(std::move(req.promise));
        }
        lock.unlock();

        if (batch_tensors.empty())
            continue;

        auto cat_start = std::chrono::high_resolution_clock::now();
        torch::Tensor input;
        if (current_batch_size < batch_size) {
            auto padding = torch::zeros({batch_size - current_batch_size, 3, 6, 7});
            batch_tensors.push_back(padding);
            input = torch::cat(batch_tensors, 0).to(device);
        } else {
            input = torch::cat(batch_tensors, 0).to(device);
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
            torch::Tensor policy_tensor = output->elements()[0].toTensor().to(torch::kCPU).contiguous();
            torch::Tensor value_tensor = output->elements()[1].toTensor().to(torch::kCPU).contiguous();

            float* policy_ptr = policy_tensor.data_ptr<float>();
            float* value_ptr = value_tensor.data_ptr<float>();
            int num_actions = policy_tensor.size(1); 

            for (int i = 0; i < current_batch_size; ++i)
            {
                EvaluatorResult res;
                res.value = value_ptr[i];
                // [FIX] Efficient data parsing: reserve space in map to reduce reallocations
                // For Connect Four, it's always 7.
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
SelfPlayEngine::SelfPlayEngine(const std::string& model_path, int batch_size, int num_threads, int num_iters, float temperature, float c_puct)
    : num_threads(num_threads), num_iters(num_iters), temperature(temperature), c_puct(c_puct)
{
    metrics = std::make_shared<PerfMetrics>();
    evaluator = std::make_shared<BatchEvaluator>(model_path, batch_size, metrics);
}

SelfPlayEngine::~SelfPlayEngine() {}

void SelfPlayEngine::expand_node(Node *node, const open_spiel::State& state, const std::map<open_spiel::Action, float> &policy)
{
    if (node->is_expanded)
        return;

    std::vector<open_spiel::Action> legal_actions = state.LegalActions();

    for (open_spiel::Action action : legal_actions)
    {
        float prob = policy.count(action) ? policy.at(action) : 0.0f;
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
        
        float q_value = child->visit_count > 0 ? child->mean_value : 0.0f;
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

void SelfPlayEngine::run_mcts(Node *root, const open_spiel::State& current_state)
{
    for (int i = 0; i < num_iters; ++i)
    {
        Node *cur_node = root;
        std::unique_ptr<open_spiel::State> cur_state = current_state.Clone();
        open_spiel::Player last_player = open_spiel::kInvalidPlayer;

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
            torch::Tensor obs = torch::from_blob(obs_vec.data(), {1, 3, 6, 7}, torch::kFloat).clone();
            EvaluatorResult res = evaluator->evaluate(obs);
            expand_node(cur_node, *cur_state, res.policy);
            value = res.value;
        }

        backpropagate(cur_node, value);
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
            torch::Tensor obs = torch::from_blob(obs_vec.data(), {1, 3, 6, 7}, torch::kFloat).clone();
            EvaluatorResult res = evaluator->evaluate(obs);
            expand_node(root.get(), *state, res.policy);
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
            std::map<open_spiel::Action, float> weights;
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
    std::cout << "-----------------------------------------------------\n" << std::endl;

    py::list py_all_trajectories;
    for (const auto& traj : all_trajectories) {
        py::list py_traj;
        for (const auto& step : traj) {
            py::array_t<float> py_obs({3, 6, 7});
            py::array_t<float> py_pi({static_cast<int>(std::get<1>(step).size())});
            std::memcpy(py_obs.mutable_data(), std::get<0>(step).data(), std::get<0>(step).size() * sizeof(float));
            std::memcpy(py_pi.mutable_data(), std::get<1>(step).data(), std::get<1>(step).size() * sizeof(float));
            py_traj.append(py::make_tuple(py_obs, py_pi, std::get<2>(step)));
        }
        py_all_trajectories.append(py_traj);
    }
    return py_all_trajectories;
}
