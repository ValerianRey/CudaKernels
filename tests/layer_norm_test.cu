#include "layer_norm.cuh"
#include "managed_array.cuh"

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <vector>

namespace {

const float EPS = 1e-5F;

void layerNormCPU(int batch, int rows, int cols, float* x, const float* weight, const float* bias, float eps)
{
	for (int b = 0; b < batch; b++) {
		for (int r = 0; r < rows; r++) {
			float* row = x + (b * rows + r) * cols;

			float mean = 0.f;
			for (int c = 0; c < cols; c++)
				mean += row[c];
			mean /= cols;

			float var = 0.f;
			for (int c = 0; c < cols; c++) {
				float diff = row[c] - mean;
				var += diff * diff;
			}
			var /= cols;

			float invStd = 1.f / std::sqrt(var + eps);
			for (int c = 0; c < cols; c++)
				row[c] = (row[c] - mean) * invStd * weight[c] + bias[c];
		}
	}
}

void fillRandom(int size, float *x, unsigned seed)
{
	std::mt19937 rng(seed);
	std::uniform_real_distribution<float> dist(-10.f, 10.f);
	for (int i = 0; i < size; i++)
		x[i] = dist(rng);
}

// A different summation order than the sequential CPU reference, so results
// only match up to floating-point rounding.
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

TEST(LayerNorm, ZeroBatch)
{
	CudaManagedArray<float> x(1);
	CudaManagedArray<float> weight(1);
	CudaManagedArray<float> bias(1);
	weight[0] = 1.f;
	bias[0] = 0.f;

	layer_norm(x.get(), 0, 1, 1, weight.get(), bias.get(), EPS);
}

TEST(LayerNorm, SingleElementPerRow)
{
	int batch = 2, rows = 3, cols = 1;
	CudaManagedArray<float> x(batch * rows * cols);
	CudaManagedArray<float> weight(cols);
	CudaManagedArray<float> bias(cols);

	fillRandom(batch * rows * cols, x.get(), 1);
	weight[0] = 2.5f;
	bias[0] = -1.25f;

	layer_norm(x.get(), batch, rows, cols, weight.get(), bias.get(), EPS);

	// A single-element row has zero variance, so it always normalizes to 0,
	// leaving only the bias after the affine transform.
	for (int i = 0; i < batch * rows * cols; i++)
		EXPECT_FLOAT_EQ(x[i], bias[0]);
}

TEST(LayerNorm, ConstantRow)
{
	int batch = 1, rows = 4, cols = 16;
	CudaManagedArray<float> x(batch * rows * cols);
	CudaManagedArray<float> weight(cols);
	CudaManagedArray<float> bias(cols);

	// Every row is constant (a different value per row), so variance is 0.
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++)
			x[r * cols + c] = static_cast<float>(r) * 2.f - 3.f;
	}
	for (int c = 0; c < cols; c++) {
		weight[c] = static_cast<float>(c + 1);
		bias[c] = -static_cast<float>(c);
	}

	layer_norm(x.get(), batch, rows, cols, weight.get(), bias.get(), EPS);

	// Zero variance means every row normalizes to 0, leaving only the bias.
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++)
			EXPECT_FLOAT_EQ(x[r * cols + c], bias[c]);
	}
}

TEST(LayerNorm, MatchesCpuReferenceIrregularShape)
{
	int batch = 2, rows = 37, cols = 53; // deliberately irregular dimensions

	CudaManagedArray<float> x(batch * rows * cols);
	CudaManagedArray<float> weight(cols);
	CudaManagedArray<float> bias(cols);
	fillRandom(batch * rows * cols, x.get(), 42);
	fillRandom(cols, weight.get(), 7);
	fillRandom(cols, bias.get(), 8);

	std::vector<float> expected(x.get(), x.get() + batch * rows * cols);
	layerNormCPU(batch, rows, cols, expected.data(), weight.get(), bias.get(), EPS);

	layer_norm(x.get(), batch, rows, cols, weight.get(), bias.get(), EPS);

	expectAllClose(batch * rows * cols, expected.data(), x.get());
}

TEST(LayerNorm, MatchesCpuReferenceLargeInput)
{
	int batch = 4, rows = 64, cols = 512;

	CudaManagedArray<float> x(batch * rows * cols);
	CudaManagedArray<float> weight(cols);
	CudaManagedArray<float> bias(cols);
	fillRandom(batch * rows * cols, x.get(), 123);
	fillRandom(cols, weight.get(), 77);
	fillRandom(cols, bias.get(), 88);

	std::vector<float> expected(x.get(), x.get() + batch * rows * cols);
	layerNormCPU(batch, rows, cols, expected.data(), weight.get(), bias.get(), EPS);

	layer_norm(x.get(), batch, rows, cols, weight.get(), bias.get(), EPS);

	expectAllClose(batch * rows * cols, expected.data(), x.get());
}
