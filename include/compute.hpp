#pragma once

#include <cstdint>
#include <vector>

namespace layer_forge {

// Quantization block sizes
constexpr int QK4_0 = 32;
constexpr int QK8_0 = 32;

// Q4_0 block: 16-bit float scale + 32 4-bit nibbles
struct block_q4_0 {
    uint16_t d;       // delta (scale)
    uint8_t qs[16];   // nibbles
};

// Q8_0 block: 16-bit float scale + 32 8-bit ints
struct block_q8_0 {
    uint16_t d;       // delta (scale)
    int8_t  qs[32];   // quants
};

// Function to dequantize Q4_0 block to float
void dequantize_q4_0(const block_q4_0* blocks, float* y, int n);

// Basic GEMM operation: y = Ax
// A: matrix (quantized), x: vector (float), y: result (float)
void gemv_q4_0(int m, int n, const block_q4_0* A, const float* x, float* y);

// Rotary Positional Embedding (RoPE)
void apply_rope(float* data, int n_dims, int n_heads, int head_dim, int pos, float theta = 10000.0f);

} // namespace layer_forge
