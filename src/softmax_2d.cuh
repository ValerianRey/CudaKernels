#pragma once

// In-place per-row softmax of a rows x cols row-major matrix.
void softmax_2d(float* input, int rows, int cols);
