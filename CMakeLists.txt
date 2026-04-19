cmake_minimum_required(VERSION 3.17)
project(mcts_engine)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
# Note: You MUST pass -DCMAKE_PREFIX_PATH=/path/to/libtorch when running cmake
find_package(Torch REQUIRED)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# OpenSpiel configuration
set(BUILD_SHARED_LIB OFF CACHE BOOL "Build shared lib for open_spiel" FORCE)
set(OPEN_SPIEL_BUILD_WITH_PYTHON OFF CACHE BOOL "Disable open_spiel python" FORCE)
set(OPEN_SPIEL_BUILD_WITH_TESTS OFF CACHE BOOL "Disable open_spiel tests" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable testing" FORCE)

add_subdirectory(third_party/open_spiel/open_spiel)

# Build our module
pybind11_add_module(mcts_backend agents/cpp/mcts_binder.cpp agents/cpp/mcts_core.cpp)

# Link against OpenSpiel object libraries directly to include them in our module
# Note: On some systems, linking to OBJECT libraries works directly. 
# We include them as source to be safe and ensure all symbols are in our SO.
target_sources(mcts_backend PRIVATE
    $<TARGET_OBJECTS:open_spiel_core>
    $<TARGET_OBJECTS:algorithms>
    $<TARGET_OBJECTS:games>
    $<TARGET_OBJECTS:game_transforms>
    $<TARGET_OBJECTS:evaluation>
    $<TARGET_OBJECTS:utils>
    $<TARGET_OBJECTS:bots>
    $<TARGET_OBJECTS:bridge_double_dummy_solver>
    $<TARGET_OBJECTS:tests>
)

# Link against dependencies
target_link_libraries(mcts_backend PRIVATE 
    ${TORCH_LIBRARIES} 
    pybind11::module
    pthread
    absl::algorithm
    absl::flags
    absl::flags_parse
    absl::flat_hash_map
    absl::flat_hash_set
    absl::optional
    absl::random_distributions
    absl::random_random
    absl::span
    absl::str_format
    absl::strings
    absl::synchronization
    absl::time
)

# Set visibility to hidden to avoid conflict with other modules (like pyspiel)
# This is crucial to avoid "Duplicate observer" errors when importing both.
set_target_properties(mcts_backend PROPERTIES 
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(APPLE)
    # On macOS, we can use -Wl,-hidden-l to hide linked library symbols, 
    # but since we are using OBJECT libraries, they are already part of our translation unit.
    # We just need to make sure they are not exported.
    target_compile_options(mcts_backend PRIVATE -fvisibility=hidden)
else()
    target_compile_options(mcts_backend PRIVATE -fvisibility=hidden)
endif()

target_include_directories(mcts_backend PRIVATE 
    third_party/open_spiel 
    third_party/open_spiel/open_spiel 
    third_party/open_spiel/open_spiel/abseil-cpp 
    third_party/open_spiel/open_spiel/json/include
)

target_compile_features(mcts_backend PRIVATE cxx_std_17)

if(NOT MSVC)
  target_compile_definitions(mcts_backend PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
endif()
