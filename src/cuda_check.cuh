#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Aborts with a clear message if a CUDA call fails. CUDA kernels/calls fail
// silently by default, so wrap calls that can error with this.
#define CUDA_CHECK(call)                                                     \
	do {                                                                      \
		cudaError_t err = (call);                                            \
		if (err != cudaSuccess) {                                            \
			std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
				__LINE__, cudaGetErrorString(err));                          \
			std::exit(EXIT_FAILURE);                                         \
		}                                                                     \
	} while (0)
