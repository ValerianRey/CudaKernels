#include "matrix_product.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

void matrixProductCPU(int m, int n, int p, const float *A, const float *B, float *out)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			float sum = 0.f;
			for (int k = 0; k < n; k++)
				sum += A[i * n + k] * B[k * p + j];
			out[i * p + j] = sum;
		}
	}
}

void fillRandom(int size, float *x, unsigned seed)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-1.f, 1.f);
	for (int i = 0; i < size; i++)
		x[i] = dist(rng);
}

// A different summation order (and, for larger matrices, more terms) than the
// sequential CPU reference, so results only match up to floating-point rounding.
void expectClose(float expected, float actual)
{
	float tolerance = std::max(1e-3f, std::fabs(expected) * 1e-3f);
	EXPECT_NEAR(expected, actual, tolerance);
}

void expectAllClose(int size, const float *expected, const float *actual)
{
	for (int i = 0; i < size; i++)
		expectClose(expected[i], actual[i]);
}

} // namespace

TEST(MatrixProduct, ZeroRows)
{
	CudaManagedArray<float> A(1);
	CudaManagedArray<float> B(1);
	CudaManagedArray<float> out(1);

	matrix_product_GPU(0, 1, 1, A.get(), B.get(), out.get());
}

TEST(MatrixProduct, SingleElement)
{
	CudaManagedArray<float> A(1);
	CudaManagedArray<float> B(1);
	CudaManagedArray<float> out(1);

	A[0] = 3.f;
	B[0] = 5.f;

	matrix_product_GPU(1, 1, 1, A.get(), B.get(), out.get());

	EXPECT_FLOAT_EQ(out[0], 15.f);
}

TEST(MatrixProduct, IdentityMatrix)
{
	int n = 4;
	CudaManagedArray<float> A(n * n);
	CudaManagedArray<float> B(n * n);
	CudaManagedArray<float> out(n * n);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			A[i * n + j] = (i == j) ? 1.f : 0.f;
	}
	fillRandom(n * n, B.get(), 1);

	matrix_product_GPU(n, n, n, A.get(), B.get(), out.get());

	for (int i = 0; i < n * n; i++)
		EXPECT_FLOAT_EQ(out[i], B[i]);
}

TEST(MatrixProduct, NonSquareDimensions)
{
	int m = 3, n = 2, p = 4;
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> out(m * p);

	for (int i = 0; i < m * n; i++)
		A[i] = 2.f;
	for (int i = 0; i < n * p; i++)
		B[i] = 3.f;

	matrix_product_GPU(m, n, p, A.get(), B.get(), out.get());

	// Every output entry sums n terms of 2 * 3.
	for (int i = 0; i < m * p; i++)
		EXPECT_FLOAT_EQ(out[i], 6.f * n);
}

TEST(MatrixProduct, DimensionsNotMultipleOfBlockSize)
{
	int m = 37, n = 21, p = 29; // deliberately not multiples of BLOCK_DIM (16)
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> out(m * p);

	fillRandom(m * n, A.get(), 42);
	fillRandom(n * p, B.get(), 43);

	std::vector<float> expected(m * p);
	matrixProductCPU(m, n, p, A.get(), B.get(), expected.data());

	matrix_product_GPU(m, n, p, A.get(), B.get(), out.get());

	expectAllClose(m * p, expected.data(), out.get());
}

TEST(MatrixProduct, LargeRandomInput)
{
	int m = 200, n = 150, p = 180;
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> out(m * p);

	fillRandom(m * n, A.get(), 123);
	fillRandom(n * p, B.get(), 456);

	std::vector<float> expected(m * p);
	matrixProductCPU(m, n, p, A.get(), B.get(), expected.data());

	matrix_product_GPU(m, n, p, A.get(), B.get(), out.get());

	expectAllClose(m * p, expected.data(), out.get());
}
