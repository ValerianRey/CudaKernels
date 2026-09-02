#pragma once

// In-place, affine layer normalization of a [batch, rows, cols] row-major
// tensor, matching PyTorch's nn.LayerNorm with normalized_shape = (cols,).
// For each (b, r) pair, normalizes across the last dimension (cols) and then
// applies a per-column affine transform:
// out[b, r, c] = (input[b, r, c] - mean) / sqrt(var + eps) * weight[c] + bias[c],
// for c in [0, cols), where mean and var (biased, i.e. divided by cols) are
// computed over that same range.
// input must be a device-accessible pointer (e.g. from cudaMalloc or
// cudaMallocManaged) holding at least batch*rows*cols floats.
// weight and bias must be device-accessible pointers holding at least cols
// floats each.
void layer_norm(float* input, int batch, int rows, int cols, float* weight, float* bias, float eps = 1e-5F);
