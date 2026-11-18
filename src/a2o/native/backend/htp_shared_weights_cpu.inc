// CPU matmul over the repository's existing Hexagon row format.
// Each row stores quants first, then one FP16 scale per 32 elements.
#include "htp_shared_weights.h"

#include "quants.h"
#include <algorithm>
#include <arm_neon.h>

namespace a2o::shared_weights {

static int dot32(int8x16_t lo, int8x16_t hi, const int8_t *x) {
    auto sum = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x));
    return vaddvq_s32(vdotq_s32(sum, hi, vld1q_s8(x + 16)));
}

static int8x16_t centered(uint8x16_t nibbles) {
    return vreinterpretq_s8_u8(vsubq_u8(nibbles, vdupq_n_u8(8)));
}

static float scale_to_float(const void *bits) {
    __fp16 value;
    std::memcpy(&value, bits, sizeof(value));
    return value;
}

template<bool q4>
static float dot_row(const uint8_t *row, const block_q8_0 *x, int64_t width) {
    const uint8_t *scales = row + (q4 ? width / 2 : width);
    float result = 0;
    for (int64_t column = 0; column < width; column += 32, ++x, scales += 2) {
        int sum;
        if constexpr (q4) {
            const int64_t tile = column / 256 * 256;
            const bool tail = tile + 256 > width;
            const auto bytes = vld1q_u8(row + (tail ? column / 2 : tile / 2 + column % 128));
            if (tail) {
                // Short final tile stores two consecutive elements per byte.
                const auto even = centered(vandq_u8(bytes, vdupq_n_u8(15)));
                const auto odd = centered(vshrq_n_u8(bytes, 4));
                sum = dot32(vzip1q_s8(even, odd), vzip2q_s8(even, odd), x->qs);
            } else {
                // A full tile pairs corresponding values from its two 128-element halves.
                const auto nibbles = column % 256 < 128 ? vandq_u8(bytes, vdupq_n_u8(15)) : vshrq_n_u8(bytes, 4);
                const auto low = centered(nibbles);
                const auto other = vld1q_u8(row + tile / 2 + column % 128 + 16);
                const auto high = centered(column % 256 < 128 ? vandq_u8(other, vdupq_n_u8(15)) : vshrq_n_u8(other, 4));
                sum = dot32(low, high, x->qs);
            }
        } else {
            const auto *bytes = reinterpret_cast<const int8_t *>(row + column);
            sum = dot32(vld1q_s8(bytes), vld1q_s8(bytes + 16), x->qs);
        }
        result += sum * scale_to_float(scales) * scale_to_float(&x->d);
    }
    return result;
}

static size_t work_size(const ggml_tensor *op) {
    return ggml_row_size(GGML_TYPE_Q8_0, op->src[1]->ne[0]) * op->src[1]->ne[1];
}

template<bool q4>
static void matmul(const ggml_compute_params *params, ggml_tensor *op) {
    const auto *weights = op->src[0];
    const auto *input = op->src[1];
    const int64_t width = weights->ne[0];
    const int64_t rows = weights->ne[1];
    const auto stride = ggml_row_size(GGML_TYPE_Q8_0, width);
    auto *scratch = static_cast<uint8_t *>(params->wdata);
    GGML_ASSERT(params->wsize >= work_size(op));
    for (int64_t col = params->ith; col < input->ne[1]; col += params->nth) {
        const auto *values = reinterpret_cast<const float *>(static_cast<const uint8_t *>(input->data) + col * input->nb[1]);
        quantize_row_q8_0(values, scratch + col * stride, width);
    }
    ggml_barrier(params->threadpool);

    const int64_t lo = rows * params->ith / params->nth;
    const int64_t hi = rows * (params->ith + 1) / params->nth;
    // Keep each packed weight row hot across a small activation tile.
    constexpr int64_t tile_cols = 4;
    for (int64_t tile = 0; tile < input->ne[1]; tile += tile_cols) {
        const int64_t end = std::min(tile + tile_cols, input->ne[1]);
        for (int64_t row = lo; row < hi; ++row) {
            const auto *packed = static_cast<const uint8_t *>(weights->data) + row * weights->nb[1];
            for (int64_t col = tile; col < end; ++col) {
                const auto *x = reinterpret_cast<const block_q8_0 *>(scratch + col * stride);
                auto *dst = reinterpret_cast<float *>(static_cast<uint8_t *>(op->data) + col * op->nb[1]);
                dst[row] = dot_row<q4>(packed, x, width);
            }
        }
    }
}

} // namespace a2o::shared_weights
