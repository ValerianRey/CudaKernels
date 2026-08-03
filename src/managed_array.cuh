#pragma once

#include "cuda_check.cuh"
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

// RAII wrapper around cudaMallocManaged / cudaFree.
// Behaves like a lightweight unique_ptr for a managed array of T.
template <typename T>
class CudaManagedArray final
{
public:
	explicit CudaManagedArray(std::size_t count) : count_(count)
	{
		CUDA_CHECK(cudaMallocManaged(&ptr_, count_ * sizeof(T)));
	}

	~CudaManagedArray()
	{
		if (ptr_)
			cudaFree(ptr_); // no throw in destructor; ignore error here
	}

	// Non-copyable
	CudaManagedArray(const CudaManagedArray &) = delete;
	CudaManagedArray &operator=(const CudaManagedArray &) = delete;

	// Movable
	CudaManagedArray(CudaManagedArray &&other) noexcept
		: ptr_(std::exchange(other.ptr_, nullptr)), count_(other.count_) {}

	CudaManagedArray &operator=(CudaManagedArray &&other) noexcept
	{
		if (this != &other) {
			if (ptr_)
				cudaFree(ptr_);
			ptr_ = std::exchange(other.ptr_, nullptr);
			count_ = other.count_;
		}
		return *this;
	}

	T *get() const { return ptr_; }
	std::size_t size() const { return count_; }

	T &operator[](std::size_t i) { return ptr_[i]; }
	const T &operator[](std::size_t i) const { return ptr_[i]; }

	void prefetchToDevice(int device = 0, cudaStream_t stream = 0) const
	{
		CUDA_CHECK(cudaMemPrefetchAsync(ptr_, count_ * sizeof(T), device, stream));
	}

private:
	T *ptr_ = nullptr;
	std::size_t count_ = 0;
};