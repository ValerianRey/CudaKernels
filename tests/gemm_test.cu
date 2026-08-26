#include "gemm.cuh"
#include "matrix_product.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {

void gemmCPU(int m, int n, int p, float alpha, const float *A, const float *B, float beta,
	const float *Cin, float *Cout)
{
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < p; j++) {
			float sum = 0.f;
			for (int k = 0; k < n; k++)
				sum += A[i * n + k] * B[k * p + j];
			Cout[i * p + j] = alpha * sum + beta * Cin[i * p + j];
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

TEST(Gemm, SingleElement)
{
	CudaManagedArray<float> A(1);
	CudaManagedArray<float> B(1);
	CudaManagedArray<float> C(1);

	A[0] = 3.f;
	B[0] = 5.f;
	C[0] = 100.f;

	gemmGPU(1, 1, 1, 2.f, A.get(), B.get(), 0.5f, C.get());

	// alpha * A@B + beta * C = 2 * 15 + 0.5 * 100 = 80
	EXPECT_FLOAT_EQ(C[0], 80.f);
}

TEST(Gemm, BetaZeroIgnoresNanInC)
{
	int m = 5, n = 4, p = 3;
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> C(m * p);

	fillRandom(m * n, A.get(), 1);
	fillRandom(n * p, B.get(), 2);
	for (int i = 0; i < m * p; i++)
		C[i] = std::numeric_limits<float>::quiet_NaN();

	gemmGPU(m, n, p, 1.f, A.get(), B.get(), 0.f, C.get());

	std::vector<float> expected(m * p);
	gemmCPU(m, n, p, 1.f, A.get(), B.get(), 0.f, C.get() /* unused when beta==0 */, expected.data());

	for (int i = 0; i < m * p; i++)
		EXPECT_FALSE(std::isnan(C[i])) << "at index " << i;
	expectAllClose(m * p, expected.data(), C.get());
}

TEST(Gemm, ZeroReductionDimOnlyScalesC)
{
	int m = 4, p = 3;
	CudaManagedArray<float> A(1); // unused: n == 0
	CudaManagedArray<float> B(1); // unused: n == 0
	CudaManagedArray<float> C(m * p);

	for (int i = 0; i < m * p; i++)
		C[i] = static_cast<float>(i + 1);

	gemmGPU(m, 0, p, 5.f, A.get(), B.get(), 2.f, C.get());

	for (int i = 0; i < m * p; i++)
		EXPECT_FLOAT_EQ(C[i], 2.f * static_cast<float>(i + 1));
}

TEST(Gemm, DimensionsNotMultipleOfBlockSize)
{
	// Deliberately not multiples of BLOCK_M/BLOCK_P (64) or BLOCK_K (16),
	// to exercise the kernel's boundary padding on every tile edge.
	int m = 70, n = 50, p = 90;
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> C(m * p);

	fillRandom(m * n, A.get(), 42);
	fillRandom(n * p, B.get(), 43);
	fillRandom(m * p, C.get(), 44);

	float alpha = 1.5f, beta = -0.5f;

	std::vector<float> expected(m * p);
	gemmCPU(m, n, p, alpha, A.get(), B.get(), beta, C.get(), expected.data());

	gemmGPU(m, n, p, alpha, A.get(), B.get(), beta, C.get());

	expectAllClose(m * p, expected.data(), C.get());
}

TEST(Gemm, MatchesNaiveMatrixProduct)
{
	int m = 137, n = 91, p = 113;
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> gemmOut(m * p);
	CudaManagedArray<float> naiveOut(m * p);

	fillRandom(m * n, A.get(), 7);
	fillRandom(n * p, B.get(), 8);

	gemmGPU(m, n, p, 1.f, A.get(), B.get(), 0.f, gemmOut.get());
	matrix_product_GPU(m, n, p, A.get(), B.get(), naiveOut.get());

	expectAllClose(m * p, naiveOut.get(), gemmOut.get());
}

TEST(Gemm, LargeRandomInput)
{
	int m = 300, n = 250, p = 275;
	CudaManagedArray<float> A(m * n);
	CudaManagedArray<float> B(n * p);
	CudaManagedArray<float> C(m * p);

	fillRandom(m * n, A.get(), 123);
	fillRandom(n * p, B.get(), 456);
	fillRandom(m * p, C.get(), 789);

	float alpha = 0.75f, beta = 1.25f;

	std::vector<float> expected(m * p);
	gemmCPU(m, n, p, alpha, A.get(), B.get(), beta, C.get(), expected.data());

	gemmGPU(m, n, p, alpha, A.get(), B.get(), beta, C.get());

	expectAllClose(m * p, expected.data(), C.get());
}
