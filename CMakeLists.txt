cmake_minimum_required(VERSION 3.14)
project(mcts_engine)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
# Note: You MUST pass -DCMAKE_PREFIX_PATH=/path/to/libtorch when running cmake
find_package(Torch REQUIRED)

pybind11_add_module(mcts_backend agents/cpp/mcts_binder.cpp agents/cpp/mcts_core.cpp)

target_link_libraries(mcts_backend PRIVATE 
    ${TORCH_LIBRARIES} 
    pybind11::module
    pthread
)

target_compile_features(mcts_backend PRIVATE cxx_std_17)

if(NOT MSVC)
  target_compile_definitions(mcts_backend PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()
