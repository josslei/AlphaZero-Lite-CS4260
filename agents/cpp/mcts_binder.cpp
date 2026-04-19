#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mcts_core.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_backend, m) {
    py::class_<SelfPlayEngine>(m, "SelfPlayEngine")
        .def(py::init<const std::string&, int, int, int, float, float>(),
             py::arg("model_path"),
             py::arg("batch_size"),
             py::arg("num_threads"),
             py::arg("num_iters"),
             py::arg("temperature"),
             py::arg("c_puct") = 1.0f)
        .def("generate_games", &SelfPlayEngine::generate_games, 
             py::arg("num_games"), py::arg("game_name"));
}
