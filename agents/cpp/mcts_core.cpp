#include "mcts_core.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// ---------- BatchEvaluator ----------
BatchEvaluator::BatchEvaluator(const std::string &model_path, int batch_size)
    : batch_size(batch_size)
{
    try
    {
        model = torch::jit::load(model_path);
        model.to(torch::kCPU); // If using GPU, change to torch::kCUDA
        model.eval();
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading model from " << model_path << "\n";
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
    std::unique_lock<std::mutex> lock(m_mutex);
    int request_id = next_id++;
    queue.push({request_id, state});

    cv_batch.notify_one();

    cv.wait(lock, [this, request_id]
            { return results.count(request_id) || stop_flag; });

    if (stop_flag && !results.count(request_id))
    {
        return EvaluatorResult(); // Protect against exit
    }

    auto res = results[request_id];
    results.erase(request_id);
    return res;
}

void BatchEvaluator::run_inference()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        // Wait until a batch is gathered, or an exit signal is received
        cv_batch.wait(lock, [this]
                      { return queue.size() >= batch_size || stop_flag; });

        if (stop_flag && queue.empty())
            break;

        int current_batch_size = std::min((int)queue.size(), batch_size);
        std::vector<torch::Tensor> batch_tensors;
        std::vector<int> batch_ids;

        for (int i = 0; i < current_batch_size; ++i)
        {
            auto req = queue.front();
            queue.pop();
            batch_ids.push_back(req.first);
            batch_tensors.push_back(req.second);
        }
        lock.unlock();

        if (batch_tensors.empty())
            continue;

        auto input = torch::stack(batch_tensors);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        // Assuming your model returns (policy_logits, value)
        auto output = model.forward(inputs).toTuple();
        torch::Tensor policy_tensor = torch::softmax(output->elements()[0].toTensor(), -1);
        torch::Tensor value_tensor = output->elements()[1].toTensor();

        lock.lock();
        for (int i = 0; i < current_batch_size; ++i)
        {
            EvaluatorResult res;
            res.value = value_tensor[i].item<float>();

            auto p = policy_tensor[i];
            for (int a = 0; a < p.size(0); ++a)
            {
                res.policy[a] = p[a].item<float>();
            }
            results[batch_ids[i]] = res;
        }
        cv.notify_all(); // Wake up all waiting worker threads
    }
}

// ---------- CppMCTS ----------
CppMCTS::CppMCTS(const std::string &model_path, int num_iters, float temperature, int num_threads, int batch_size, float c_puct)
    : num_iters(num_iters), temperature(temperature), num_threads(num_threads), c_puct(c_puct)
{
    evaluator = std::make_shared<BatchEvaluator>(model_path, batch_size);
}

CppMCTS::~CppMCTS() {}

void CppMCTS::expand_node(Node *node, py::object state, const std::map<int, float> &policy)
{
    std::lock_guard<std::mutex> lock(node->node_mutex);
    if (node->is_expanded)
        return;

    // Get legal actions from Python. Acquire GIL!
    py::gil_scoped_acquire acquire;
    py::list legal_actions = state.attr("legal_actions")();
    for (auto item : legal_actions)
    {
        int action = item.cast<int>();
        float prob = policy.count(action) ? policy.at(action) : 0.0f;
        node->children[action] = std::make_unique<Node>(node, prob);
    }
    node->is_expanded = true;
}

std::pair<int, Node *> CppMCTS::select_best_child(Node *node)
{
    std::lock_guard<std::mutex> lock(node->node_mutex);
    int best_action = -1;
    Node *best_child = nullptr;
    float best_score = -1e9;

    float sqrt_parent_visits = std::sqrt((float)node->visit_count);

    for (auto &pair : node->children)
    {
        int action = pair.first;
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

void CppMCTS::backpropagate(Node *node, float value)
{
    Node *cur = node;
    while (cur != nullptr)
    {
        std::lock_guard<std::mutex> lock(cur->node_mutex);
        cur->visit_count++;
        cur->total_value += value;
        cur->mean_value = cur->total_value / cur->visit_count;
        cur = cur->parent;
        value = -value; // Switch perspective
    }
}

void CppMCTS::mcts_worker(Node *root, py::object s_init, int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        Node *cur_node = root;

        // C++ calling Python's clone requires GIL.
        // (Warning: If the Python side is slow, multiple threads will queue up here competing for the GIL)
        py::gil_scoped_acquire acquire;
        py::object cur_state;
        if (py::hasattr(s_init, "clone"))
        {
            cur_state = s_init.attr("clone")();
        }
        else
        {
            // Fallback to deep copy
            py::object copy_module = py::module::import("copy");
            cur_state = copy_module.attr("deepcopy")(s_init);
        }
        py::gil_scoped_release release;

        // Selection
        while (cur_node->is_expanded)
        {
            bool is_terminal;
            {
                py::gil_scoped_acquire acquire_term;
                is_terminal = cur_state.attr("is_terminal")().cast<bool>();
            }
            if (is_terminal)
                break;

            auto best = select_best_child(cur_node);
            int action = best.first;
            Node *next_node = best.second;

            {
                py::gil_scoped_acquire acquire_apply;
                cur_state.attr("apply_action")(action);
            }
            cur_node = next_node;
        }

        // Evaluation & Expansion
        float value = 0.0f;
        bool is_terminal;
        {
            py::gil_scoped_acquire acquire_eval;
            is_terminal = cur_state.attr("is_terminal")().cast<bool>();
        }

        if (is_terminal)
        {
            py::gil_scoped_acquire acquire_eval;
            value = cur_state.attr("rewards")().cast<float>();
        }
        else
        {
            // Important: You need to provide a method (e.g., observation_tensor()) in the Python state object
            // to convert the game grid into a flattened 1D std::vector recognizable by the C++ side, or directly return a Tensor-compatible format.
            // Here we create a dummy all-zero Tensor as an example input
            torch::Tensor obs = torch::zeros({1, 3, 6, 7}); // Dummy Shape

            EvaluatorResult res = evaluator->evaluate(obs);
            expand_node(cur_node, cur_state, res.policy);
            value = res.value;
        }

        // Backpropagation
        backpropagate(cur_node, value);
    }
}

py::dict CppMCTS::search(py::object s_init)
{
    Node root(nullptr, 1.0f);

    // Initial evaluation for root
    py::gil_scoped_acquire acquire;
    torch::Tensor obs = torch::zeros({1, 3, 6, 7}); // Dummy
    py::gil_scoped_release release;

    EvaluatorResult res = evaluator->evaluate(obs);
    expand_node(&root, s_init, res.policy);

    // Start multiple worker threads for parallel tree search
    int iters_per_thread = num_iters / num_threads;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&CppMCTS::mcts_worker, this, &root, s_init, iters_per_thread);
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    // Policy generation
    py::dict probs;
    if (temperature <= 1e-3f)
    {
        int max_visits = -1;
        int best_action = -1;
        for (auto &pair : root.children)
        {
            if (pair.second->visit_count > max_visits)
            {
                max_visits = pair.second->visit_count;
                best_action = pair.first;
            }
        }
        for (auto &pair : root.children)
        {
            probs[py::cast(pair.first)] = (pair.first == best_action) ? 1.0f : 0.0f;
        }
    }
    else
    {
        float sum = 0.0f;
        std::map<int, float> weights;
        for (auto &pair : root.children)
        {
            float w = std::pow(pair.second->visit_count, 1.0f / temperature);
            weights[pair.first] = w;
            sum += w;
        }
        for (auto &pair : weights)
        {
            probs[py::cast(pair.first)] = sum > 0 ? (pair.second / sum) : (1.0f / weights.size());
        }
    }
    return probs;
}
