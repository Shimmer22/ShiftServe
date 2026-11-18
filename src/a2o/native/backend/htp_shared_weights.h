#pragma once

#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>

namespace a2o::shared_weights {

inline bool enabled() {
    static const bool value = [] {
        const char *flag = std::getenv("A2O_SHARED_WEIGHTS");
        return flag && std::strcmp(flag, "1") == 0;
    }();
    return value;
}

inline bool hexagon_buffer(ggml_backend_buffer_type_t type, const char *name) {
    if (!type) return false;
    auto dev = ggml_backend_buft_get_device(type);
    return dev && std::strcmp(ggml_backend_dev_name(dev), "HTP0") == 0 &&
           std::strcmp(ggml_backend_buft_name(type), name) == 0;
}

inline bool packed_type(ggml_backend_buffer_type_t type) {
    return enabled() && hexagon_buffer(type, "HTP0-REPACK");
}

inline bool packed_tensor(const ggml_tensor *tensor) {
    if (!tensor) return false;
    auto buffer = tensor->buffer ? tensor->buffer : tensor->view_src ? tensor->view_src->buffer : nullptr;
    return buffer && packed_type(ggml_backend_buffer_get_type(buffer));
}

inline bool supports_matmul(const ggml_tensor *op) {
    const auto *weights = op->src[0];
    const auto *input = op->src[1];
    return op->op == GGML_OP_MUL_MAT && packed_tensor(weights) &&
           (weights->type == GGML_TYPE_Q4_0 || weights->type == GGML_TYPE_Q8_0) &&
           weights->ne[0] % 32 == 0 && !weights->view_src && ggml_is_contiguous(weights) &&
           input && input->type == GGML_TYPE_F32 && ggml_is_contiguous(input) &&
           op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
           ggml_is_matrix(weights) && ggml_is_matrix(input) && ggml_is_matrix(op);
}

} // namespace a2o::shared_weights
