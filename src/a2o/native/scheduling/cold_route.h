#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// 同一模型和执行布局的 H=0 实测。HC 计入 HTP 上下文准备和 KV 回迁。
struct ColdRoutePoint {
    int prompt_tokens;
    double cc_ms;
    double hc_ms;
};

class ColdRouteCosts {
public:
    std::string model_id;
    bool shared_weights = false;

    bool configured() const { return !points_.empty(); }

    std::optional<ColdRoutePoint> estimate(int tokens) const {
        if (!configured() || tokens < points_.front().prompt_tokens || tokens > points_.back().prompt_tokens) {
            return std::nullopt;
        }
        const auto right = std::lower_bound(points_.begin(), points_.end(), tokens,
                [](const ColdRoutePoint & point, int p) { return point.prompt_tokens < p; });
        if (right->prompt_tokens == tokens) return *right;
        const auto & left = *std::prev(right);
        const double weight = double(tokens - left.prompt_tokens) / (right->prompt_tokens - left.prompt_tokens);
        return ColdRoutePoint{tokens, left.cc_ms + weight * (right->cc_ms - left.cc_ms),
                                    left.hc_ms + weight * (right->hc_ms - left.hc_ms)};
    }

    template <typename Json>
    static ColdRouteCosts parse(const Json & data) {
        if (data.at("kind") != "cold_route_v1") throw std::runtime_error("unsupported cold route model");
        ColdRouteCosts table;
        table.model_id = data.at("model_id").template get<std::string>();
        table.shared_weights = data.at("shared_weights").template get<bool>();
        const auto & points = data.at("points");
        if (table.model_id.empty() || !points.is_array() || points.empty()) {
            throw std::runtime_error("cold route requires model ID and measured points");
        }
        for (const auto & row : points) {
            const auto & length = row.at("P");
            if (!length.is_number_integer()) throw std::runtime_error("cold route P must be an integer");
            const auto tokens = length.template get<int64_t>();
            ColdRoutePoint point{0, row.at("cc_ms").template get<double>(), row.at("hc_ms").template get<double>()};
            if (tokens <= 0 || tokens > std::numeric_limits<int>::max() ||
                    (!table.points_.empty() && tokens <= table.points_.back().prompt_tokens) ||
                    !std::isfinite(point.cc_ms) || !std::isfinite(point.hc_ms) || point.cc_ms <= 0 || point.hc_ms <= 0) {
                throw std::runtime_error("cold route points require increasing P and positive finite costs");
            }
            point.prompt_tokens = static_cast<int>(tokens);
            table.points_.push_back(point);
        }
        return table;
    }

private:
    std::vector<ColdRoutePoint> points_;
};
