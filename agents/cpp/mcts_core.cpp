#include "mcts_core.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

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
        cv_batch.wait_for(lock, std::chrono::milliseconds(2), [this]
                          { return queue.size() >= batch_size || stop_flag; });

        if (stop_flag && queue.empty())
            break;

        if (queue.empty())
            continue;

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

        auto input = torch::cat(batch_tensors, 0); // Use cat instead of stack because each tensor is already [1, C, H, W]
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        try {
            auto output = model.forward(inputs).toTuple();
            torch::Tensor policy_tensor = output->elements()[0].toTensor();
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
        } catch (const c10::Error& e) {
            std::cerr << "LibTorch Error in run_inference: " << e.what() << std::endl;
            // Free waiting threads
            lock.lock();
            stop_flag = true;
            cv.notify_all();
            break;
        }
    }
}

// ---------- CppMCTS ----------
CppMCTS::CppMCTS(const std::string &model_path, int num_iters, float temperature, int num_threads, int batch_size, float c_puct)
    : num_iters(num_iters), temperature(temperature), num_threads(num_threads), c_puct(c_puct)
{
    evaluator = std::make_shared<BatchEvaluator>(model_path, batch_size);
}

CppMCTS::~CppMCTS() {}

void CppMCTS::expand_node(Node *node, const open_spiel::State& state, const std::map<open_spiel::Action, float> &policy)
{
    if (node->is_expanded)
        return;

    std::vector<open_spiel::Action> legal_actions = state.LegalActions();

    std::lock_guard<std::mutex> lock(node->node_mutex);
    
    // Double-check
    if (node->is_expanded)
        return; 

    for (open_spiel::Action action : legal_actions)
    {
        float prob = policy.count(action) ? policy.at(action) : 0.0f;
        node->children[action] = std::make_unique<Node>(node, prob);
    }
    node->is_expanded = true;
}

std::pair<open_spiel::Action, Node *> CppMCTS::select_best_child(Node *node)
{
    std::lock_guard<std::mutex> lock(node->node_mutex);
    open_spiel::Action best_action = -1;
    Node *best_child = nullptr;
    float best_score = -1e9;

    float parent_visits = node->visit_count + node->virtual_loss.load();
    float sqrt_parent_visits = std::sqrt(std::max(1.0f, parent_visits));

    for (auto &pair : node->children)
    {
        open_spiel::Action action = pair.first;
        Node *child = pair.second.get();
        
        float child_visits = child->visit_count + child->virtual_loss.load();
        
        float q_value = child_visits > 0 ? (child->total_value - child->virtual_loss.load()) / child_visits : 0.0f;
        float u_value = c_puct * child->prior_prob * sqrt_parent_visits / (1.0f + child_visits);
        float score = q_value + u_value;

        if (score > best_score)
        {
            best_score = score;
            best_action = action;
            best_child = child;
        }
    }
    
    if (best_child != nullptr) {
        best_child->virtual_loss++;
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
        
        if (cur->virtual_loss.load() > 0) {
            cur->virtual_loss--;
        }
        
        cur = cur->parent;
        value = -value; // Switch perspective
    }
}

void CppMCTS::mcts_worker(Node *root, std::shared_ptr<const open_spiel::Game> game, const std::vector<open_spiel::Action>& history, int iters)
{
    for (int i = 0; i < iters; ++i)
    {
        Node *cur_node = root;

        std::unique_ptr<open_spiel::State> cur_state = game->NewInitialState();
        for (open_spiel::Action action : history) {
            cur_state->ApplyAction(action);
        }

        open_spiel::Player last_player = open_spiel::kInvalidPlayer;

        // Selection
        while (cur_node->is_expanded)
        {
            if (cur_state->IsTerminal())
                break;

            last_player = cur_state->CurrentPlayer();

            auto best = select_best_child(cur_node);
            open_spiel::Action action = best.first;
            Node *next_node = best.second;

            cur_state->ApplyAction(action);
            cur_node = next_node;
        }

        // Evaluation & Expansion
        float value = 0.0f;
        if (cur_state->IsTerminal())
        {
            if (last_player != open_spiel::kInvalidPlayer) {
                // In 2-player zero-sum games, the perspective of the next player is the negative of the last player
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

py::dict CppMCTS::search(const std::string& game_string, const std::vector<open_spiel::Action>& history)
{
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame(game_string);
    std::unique_ptr<open_spiel::State> root_state = game->NewInitialState();
    for (open_spiel::Action action : history) {
        root_state->ApplyAction(action);
    }

    Node root(nullptr, 1.0f);

    std::vector<float> obs_vec = root_state->ObservationTensor();
    torch::Tensor obs = torch::from_blob(obs_vec.data(), {1, 3, 6, 7}, torch::kFloat).clone();

    EvaluatorResult res = evaluator->evaluate(obs);
    expand_node(&root, *root_state, res.policy);

    int iters_per_thread = num_iters / num_threads;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(&CppMCTS::mcts_worker, this, &root, game, history, iters_per_thread);
    }

    for (auto &t : threads)
    {
        if (t.joinable())
            t.join();
    }

    py::dict probs;
    if (temperature <= 1e-3f)
    {
        int max_visits = -1;
        open_spiel::Action best_action = -1;
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
        std::map<open_spiel::Action, float> weights;
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