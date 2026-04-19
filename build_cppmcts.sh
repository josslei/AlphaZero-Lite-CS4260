#!/bin/bash
set -e

# Navigate to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building CppMCTS backend..."

# Create build directory
mkdir -p build
cd build

# Get the path to pybind11's CMake configuration dynamically from the current Python environment
PYBIND11_CMAKE_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')

# Configure with CMake
# Assumes libtorch is located at agents/cpp/libtorch relative to the project root
cmake -DCMAKE_PREFIX_PATH="$SCRIPT_DIR/agents/cpp/libtorch" \
      -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" \
      -DPython_EXECUTABLE="$(which python3)" \
      -DBUILD_SHARED_LIB=OFF \
      -DOPEN_SPIEL_BUILD_WITH_PYTHON=OFF \
      -DOPEN_SPIEL_BUILD_WITH_TESTS=OFF \
      -DBUILD_TESTING=OFF \
      ..

# Compile the C++ extension
make -j4

# Move the compiled shared object (.so) file to the agents/ directory so Python can import it
echo "Moving compiled extension to agents/..."
cp mcts_backend*.so ../agents/

echo "Build complete! mcts_backend is ready to import."
