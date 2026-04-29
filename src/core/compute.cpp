#include "compute.hpp"
#include <cmath>
#include <cstring>

namespace layer_forge {

// Helper to convert FP16 to FP32 (naive implementation)
static float fp16_to_fp32(uint16_t h) {
    uint32_t e = (h & 0x7C00) >> 10;
    uint32_t m = (h & 0x03FF) << 13;
    uint32_t v = (h & 0x8000) << 16;
    if (e == 0x1F) {
        v |= 0x7F800000 | m;
    } else if (e == 0) {
        if (m != 0) {
            e = 1;
            while (!(m & 0x00800000)) {
                m <<= 1;
                e--;
            }
            m &= 0x007FFFFF;
            v |= ((e + 112) << 23) | m;
        }
    } else {
        v |= ((e + 112) << 23) | m;
    }
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

void dequantize_q4_0(const block_q4_0* blocks, float* y, int n) {
    for (int i = 0; i < n / QK4_0; ++i) {
        const float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uint8_t x = blocks[i].qs[j];
            y[i * QK4_0 + j] = d * ((x & 0x0F) - 8);
            y[i * QK4_0 + j + QK4_0 / 2] = d * ((x >> 4) - 8);
        }
    }
}

void gemv_q4_0(int m, int n, const block_q4_0* A, const float* x, float* y) {
    // A: [m, n], x: [n], y: [m]
    // A is stored row-major (each row is n/QK4_0 blocks)
    for (int i = 0; i < m; ++i) {
        float sum = 0.0f;
        const block_q4_0* row = A + i * (n / QK4_0);
        for (int j = 0; j < n / QK4_0; ++j) {
            const float d = fp16_to_fp32(row[j].d);
            for (int k = 0; k < QK4_0 / 2; ++k) {
                const uint8_t x_val = row[j].qs[k];
                sum += d * ((x_val & 0x0F) - 8) * x[j * QK4_0 + k];
                sum += d * ((x_val >> 4) - 8) * x[j * QK4_0 + k + QK4_0 / 2];
            }
        }
        y[i] = sum;
    }
}

void apply_rope(float* data, int n_dims, int n_heads, int head_dim, int pos, float theta) {
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n_dims / 2; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
            float val = pos * freq;
            float cos_v = std::cos(val);
            float sin_v = std::sin(val);
            
            float* x = data + h * head_dim + 2 * i;
            float x0 = x[0];
            float x1 = x[1];
            
            x[0] = x0 * cos_v - x1 * sin_v;
            x[1] = x0 * sin_v + x1 * cos_v;
        }
    }
}

} // namespace layer_forge
