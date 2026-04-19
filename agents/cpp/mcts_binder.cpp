#include <pybind11/pybind11.h>
#include "mcts_core.h"

namespace py = pybind11;

PYBIND11_MODULE(mcts_backend, m) {
    py::class_<CppMCTS>(m, "CppMCTS")
        .def(py::init<const std::string&, int, float, int, int, float>(),
             py::arg("model_path"), 
             py::arg("num_iters"), 
             py::arg("temperature"), 
             py::arg("num_threads") = 4, 
             py::arg("batch_size") = 8, 
             py::arg("c_puct") = 1.0f)
        .def("search", &CppMCTS::search, "Run multi-threaded MCTS search on the given state");
}
