# CudaKernels
Small educational project to learn how to write cuda kernels.

### Prerequisites
To compile and run the kernels, you need an Nvidia GPU, the [cuda toolkit](https://developer.nvidia.com/cuda/toolkit), and [CMake](https://cmake.org/) (3.24+) installed.

### Building and running

The project moved from a hand-written Makefile to CMake, which handles CUDA
compilation and GoogleTest natively instead of needing manual build rules per
kernel and a hand-rolled GoogleTest clone/compile step.

```bash
cmake -S . -B build   # configure (fetches GoogleTest into build/ on first run)
cmake --build build -j
```

Run a demo:
```bash
./build/add_vector_demo
./build/dot_product_demo
./build/matrix_product_demo
```

Run the tests:
```bash
ctest --test-dir build
```

Run benchmark (only for dot_product):
```bash
./build/dot_product_bench
```

Run profiler (Nsight Compute is not compatible with GTX 1080):
```bash
nvprof --print-gpu-trace ./build/dot_product_demo
```

