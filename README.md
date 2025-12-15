# CudaKernels
Small educational project to learn how to write cuda kernels.

### Prerequisites
To compile and run the kernels, you need an Nvidia GPU and the [cuda toolkit](https://developer.nvidia.com/cuda/toolkit) installed.

### Compiling
```bash
nvcc -o <kernel-name> <kernel-name>.cu
```

### Running
```bash
./<kernel-name>
```

### Profiling
```bash
nsys profile -t cuda --stats=true -o /dev/null ./<kernel-name>
```

