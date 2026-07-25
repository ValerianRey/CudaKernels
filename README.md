# CudaKernels
Small educational project to learn how to write cuda kernels.

### Prerequisites
To compile and run the kernels, you need an Nvidia GPU and the [cuda toolkit](https://developer.nvidia.com/cuda/toolkit) installed.

### Building and running

```bash
make demo   # build demo kernels
make test   # fetch GoogleTest into third_party/ (first run only), build, and run tests
```

Run benchmark (only for dot_product):
```bash
make bench  # build ./build/dot_product_bench
./build/dot_product_bench
```

Run profiler (Nsight Compute is not compatible with GTX 1080):
```bash
nvprof --print-gpu-trace ./build/dot_product_demo
```

