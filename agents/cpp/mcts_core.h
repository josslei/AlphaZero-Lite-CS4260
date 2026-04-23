#pragma once
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <string>
#include <atomic>
#include <future>
#include <chrono>

#include "open_spiel/spiel.h"

namespace py = pybind11;

// Performance Monitoring Structure
struct PerfMetrics {
    std::atomic<long long> wait_time_us{0};      // Time spent waiting to fill a batch
    std::atomic<long long> cat_time_us{0};       // Time spent in torch::cat and device move
    std::atomic<long long> forward_time_us{0};   // Time spent in model.forward()
    std::atomic<long long> parse_time_us{0};     // Time spent extracting data to results
    std::atomic<long long> total_batches{0};     // Number of batches processed
    std::atomic<long long> total_requests{0};    // Total inference requests handled
    
    std::atomic<long long> mcts_search_time_us{0};    // Time spent in MCTS logic (excluding eval wait)
    std::atomic<long long> mcts_eval_wait_time_us{0}; // Time spent waiting for NN results
    std::atomic<long long> iters_saved{0};            // Number of simulations skipped by early stopping
    
    std::atomic<long long> total_search_depth{0};
    std::atomic<long long> num_searches{0};
    std::atomic<int> max_search_depth{0};
    
    void reset() {
        wait_time_us = 0; cat_time_us = 0; forward_time_us = 0; parse_time_us = 0;
        total_batches = 0; total_requests = 0;
        mcts_search_time_us = 0; mcts_eval_wait_time_us = 0;
        iters_saved = 0;
        total_search_depth = 0; num_searches = 0; max_search_depth = 0;
    }
};

struct EvaluatorResult
{
    std::vector<float> policy;
    float value;
};

struct EvalRequest {
    std::vector<float> obs;           // Owned copy of observation data (cheap: ~500 bytes)
    std::promise<EvaluatorResult> promise;
};

class BatchEvaluator
{
public:
    BatchEvaluator(const std::string &model_path, int batch_size, int obs_flat_size, std::shared_ptr<PerfMetrics> metrics, bool use_fp16 = false);
    ~BatchEvaluator();

    EvaluatorResult evaluate(const float* obs_data);
    void run_inference();

private:
    torch::jit::script::Module model;
    int batch_size;
    int obs_flat_size;
    bool use_fp16;
    bool stop_flag = false;
    torch::Device device{torch::kCPU};

    // Pre-allocated batch buffer filled by inference thread from queue
    torch::Tensor batch_buffer;       // [batch_size, obs_flat_size] pinned CPU

    std::mutex m_mutex;
    std::condition_variable cv_batch;

    std::queue<EvalRequest> queue;    // Unbounded queue — workers never block on submission
    std::thread worker_thread;
    std::shared_ptr<PerfMetrics> metrics;
};

struct Node
{
    Node *parent;
    absl::flat_hash_map<open_spiel::Action, std::unique_ptr<Node>> children;
    bool is_expanded = false;

    int visit_count = 0;
    float total_value = 0.0f;
    float mean_value = 0.0f;
    float prior_prob = 1.0f;

    Node(Node *p = nullptr, float prior = 1.0f) : parent(p), prior_prob(prior) {}
};

struct StepRecord {
    std::vector<float> obs;
    std::vector<float> pi;
    open_spiel::Player player;
};

class SelfPlayEngine
{
public:
    SelfPlayEngine(const std::string& model_path, int batch_size, int obs_flat_size, int num_threads, int num_iters, float temperature, float c_puct, float dirichlet_alpha, float dirichlet_epsilon, bool use_fp16 = false, bool use_undo = false);
    ~SelfPlayEngine();

    py::list generate_games(int num_games, const std::string& game_name);
    py::dict get_metrics(); 

private:
    void play_game(const std::string& game_name, std::vector<std::vector<std::tuple<std::vector<float>, std::vector<float>, float>>>& all_trajectories, std::mutex& traj_mutex);
    void run_mcts(Node *root, open_spiel::State& current_state);
    std::pair<open_spiel::Action, Node *> select_best_child(Node *node, const std::vector<open_spiel::Action>& legal_actions);
    void expand_node(Node *node, const open_spiel::State& state, const std::vector<float> &policy);
    void backpropagate(Node *node, float value);
    void advance_chance_nodes(open_spiel::State* state, std::vector<std::pair<open_spiel::Player, open_spiel::Action>>* action_path = nullptr);

    std::shared_ptr<BatchEvaluator> evaluator;
    std::shared_ptr<PerfMetrics> metrics;
    int obs_flat_size;
    int num_threads;
    int num_iters;
    float temperature;
    float c_puct;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    bool use_undo;
};
