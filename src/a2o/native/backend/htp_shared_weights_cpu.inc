// CPU matmul over the repository's existing Hexagon row format.
// Each row stores quants first, then one FP16 scale per 32 elements.
#include "htp_shared_weights.h"

#include "quants.h"
#include <algorithm>
#include <arm_neon.h>

namespace a2o::shared_weights {

// Match a full Hexagon tile. The activation scale is converted once per input
// column, rather than again for every output row.
struct alignas(16) ActivationTile {
    int8_t quants[256];
    float scales[8];
};

static int32x4_t dot32(int8x16_t lo, int8x16_t hi, const int8_t *x) {
    auto sum = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(x));
    return vdotq_s32(sum, hi, vld1q_s8(x + 16));
}

static int8x16_t centered(uint8x16_t nibbles) {
    return vreinterpretq_s8_u8(vsubq_u8(nibbles, vdupq_n_u8(8)));
}

static float scale_to_float(const void *bits) {
    __fp16 value;
    std::memcpy(&value, bits, sizeof(value));
    return value;
}

static float32x4_t scale_four(const uint8_t *bits, const float *activation_scales) {
    float16x4_t weight_scales;
    std::memcpy(&weight_scales, bits, sizeof(weight_scales));
    return vmulq_f32(vcvt_f32_f16(weight_scales), vld1q_f32(activation_scales));
}

static int32x4_t sum_blocks(const int32x4_t (&dots)[4]) {
    return vpaddq_s32(vpaddq_s32(dots[0], dots[1]), vpaddq_s32(dots[2], dots[3]));
}

template<bool q4>
static float dot_row(const uint8_t *row, const ActivationTile *x, int64_t width) {
    const uint8_t *scales = row + (q4 ? width / 2 : width);
    const auto mask = vdupq_n_u8(15);
    auto sums = vdupq_n_f32(0);
    for (int64_t column = 0; column + 256 <= width; column += 256, ++x, scales += 16) {
        int32x4_t lo[4], hi[4];
        for (int block = 0; block < 4; ++block) {
            const auto *w = row + (q4 ? column / 2 : column) + block * 32;
            const auto *a = x->quants + block * 32;
            if constexpr (q4) {
                // The two 128-value halves share the same bytes.
                const auto left = vld1q_u8(w);
                const auto right = vld1q_u8(w + 16);
                lo[block] = dot32(centered(vandq_u8(left, mask)), centered(vandq_u8(right, mask)), a);
                hi[block] = dot32(centered(vshrq_n_u8(left, 4)), centered(vshrq_n_u8(right, 4)), a + 128);
            } else {
                const auto *q = reinterpret_cast<const int8_t *>(w);
                lo[block] = dot32(vld1q_s8(q), vld1q_s8(q + 16), a);
                hi[block] = dot32(vld1q_s8(q + 128), vld1q_s8(q + 144), a + 128);
            }
        }
        sums = vfmaq_f32(sums, vcvtq_f32_s32(sum_blocks(lo)), scale_four(scales, x->scales));
        sums = vfmaq_f32(sums, vcvtq_f32_s32(sum_blocks(hi)), scale_four(scales + 8, x->scales + 4));
    }

    float result = vaddvq_f32(sums);
    const int tail = width % 256;
    const auto *last = row + (q4 ? (width - tail) / 2 : width - tail);
    for (int i = 0; i < tail; i += 32, scales += 2) {
        const auto *a = x->quants + i;
        int32x4_t dot;
        if constexpr (q4) {
            // A partial tile packs adjacent values into each byte.
            const auto bytes = vld1q_u8(last + i / 2);
            const auto even = centered(vandq_u8(bytes, mask));
            const auto odd = centered(vshrq_n_u8(bytes, 4));
            dot = dot32(vzip1q_s8(even, odd), vzip2q_s8(even, odd), a);
        } else {
            const auto *q = reinterpret_cast<const int8_t *>(last + i);
            dot = dot32(vld1q_s8(q), vld1q_s8(q + 16), a);
        }
        result += vaddvq_s32(dot) * scale_to_float(scales) * x->scales[i / 32];
    }
    return result;
}

static size_t activation_tiles(int64_t width) {
    return (width + 255) / 256;
}

static size_t work_size(const ggml_tensor *op) {
    const auto *input = op->src[1];
    const size_t tile_bytes = activation_tiles(input->ne[0]) * sizeof(ActivationTile);
    const size_t quant_bytes = ggml_row_size(GGML_TYPE_Q8_0, input->ne[0]);
    return (tile_bytes + quant_bytes) * input->ne[1];
}

template<bool q4>
static void matmul(const ggml_compute_params *params, ggml_tensor *op) {
    const auto *weights = op->src[0];
    const auto *input = op->src[1];
    const int64_t width = weights->ne[0];
    const int64_t rows = weights->ne[1];
    const size_t tiles = activation_tiles(width);
    auto *packed_input = static_cast<ActivationTile *>(params->wdata);
    auto *quantized = reinterpret_cast<block_q8_0 *>(packed_input + input->ne[1] * tiles);
    GGML_ASSERT(params->wsize >= work_size(op));
    for (int64_t col = params->ith; col < input->ne[1]; col += params->nth) {
        const auto *values = reinterpret_cast<const float *>(static_cast<const uint8_t *>(input->data) + col * input->nb[1]);
        auto *q8 = quantized + col * (width / 32);
        quantize_row_q8_0(values, q8, width);
        auto *packed = packed_input + col * tiles;
        for (int64_t block = 0; block < width / 32; ++block) {
            auto &tile = packed[block / 8];
            std::memcpy(tile.quants + block % 8 * 32, q8[block].qs, 32);
            tile.scales[block % 8] = scale_to_float(&q8[block].d);
        }
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
                const auto *x = packed_input + col * tiles;
                auto *dst = reinterpret_cast<float *>(static_cast<uint8_t *>(op->data) + col * op->nb[1]);
                dst[row] = dot_row<q4>(packed, x, width);
            }
        }
    }
}

} // namespace a2o::shared_weights
