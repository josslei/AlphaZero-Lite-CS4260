#pragma once
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <memory>
#include <string>

namespace py = pybind11;

struct EvaluatorResult
{
    std::map<int, float> policy;
    float value;
};

class BatchEvaluator
{
public:
    BatchEvaluator(const std::string &model_path, int batch_size);
    ~BatchEvaluator();

    EvaluatorResult evaluate(const torch::Tensor &state);
    void run_inference();

private:
    torch::jit::script::Module model;
    int batch_size;
    int next_id = 0;
    bool stop_flag = false;

    std::mutex m_mutex;
    std::condition_variable cv;
    std::condition_variable cv_batch;

    std::queue<std::pair<int, torch::Tensor>> queue;
    std::map<int, EvaluatorResult> results;
    std::thread worker_thread;
};

struct Node
{
    Node *parent;
    std::map<int, std::unique_ptr<Node>> children;
    bool is_expanded = false;

    int visit_count = 0;
    float total_value = 0.0f;
    float mean_value = 0.0f;
    float prior_prob = 1.0f;
    std::mutex node_mutex; // Protect node properties from concurrent modification

    Node(Node *p = nullptr, float prior = 1.0f) : parent(p), prior_prob(prior) {}
};

class CppMCTS
{
public:
    CppMCTS(const std::string &model_path, int num_iters, float temperature, int num_threads, int batch_size, float c_puct);
    ~CppMCTS();

    py::dict search(py::object s_init);

private:
    void mcts_worker(Node *root, py::object s_init, int iters);
    std::pair<int, Node *> select_best_child(Node *node);
    void expand_node(Node *node, py::object state, const std::map<int, float> &policy);
    void backpropagate(Node *node, float value);

    std::shared_ptr<BatchEvaluator> evaluator;
    int num_iters;
    float temperature;
    int num_threads;
    float c_puct;
};
