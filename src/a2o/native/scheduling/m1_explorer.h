#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "scheduling/kv_reduce_cost.h"

namespace a2o_m1 {

inline std::vector<double> solve_linear(std::vector<std::vector<double>> a, const std::vector<double> &b) {
    const size_t n = b.size();
    for (size_t i = 0; i < n; ++i) a[i].push_back(b[i]);
    for (size_t col = 0; col < n; ++col) {
        size_t piv = col;
        for (size_t r = col + 1; r < n; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
        }
        if (std::abs(a[piv][col]) < 1e-12) {
            throw std::runtime_error("Singular matrix in BayesianLinearComponent");
        }
        std::swap(a[col], a[piv]);
        const double diag = a[col][col];
        for (size_t j = col; j <= n; ++j) a[col][j] /= diag;
        for (size_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = a[r][col];
            if (factor == 0.0) continue;
            for (size_t j = col; j <= n; ++j) a[r][j] -= factor * a[col][j];
        }
    }
    std::vector<double> x(n, 0.0);
    for (size_t i = 0; i < n; ++i) x[i] = a[i][n];
    return x;
}

inline std::vector<std::vector<double>> invert_matrix(const std::vector<std::vector<double>> &a) {
    const size_t n = a.size();
    std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> e(n, 0.0);
        e[i] = 1.0;
        auto col = solve_linear(a, e);
        for (size_t r = 0; r < n; ++r) inv[r][i] = col[r];
    }
    return inv;
}

class BayesianLinearComponent {
public:
    size_t dim = 0;
    double noise_var = 1.0;
    double forgetting = 0.98;
    std::vector<std::vector<double>> precision;
    std::vector<double> info_vec;
    std::vector<std::vector<double>> prior_precision;
    std::vector<double> prior_info;
    std::vector<std::vector<double>> cov;
    std::vector<double> mean;
    int n_obs = 0;

    BayesianLinearComponent() = default;
    BayesianLinearComponent(
            size_t d,
            const std::vector<double> &prior_mean,
            const std::vector<double> &prior_var,
            double obs_noise_std,
            double forget = 0.98)
        : dim(d),
          noise_var(std::max(1e-6, obs_noise_std * obs_noise_std)),
          forgetting(forget),
          precision(d, std::vector<double>(d, 0.0)),
          info_vec(d, 0.0) {
        if (prior_mean.size() != d || prior_var.size() != d) {
            throw std::runtime_error("prior dimension mismatch");
        }
        for (size_t i = 0; i < d; ++i) {
            const double pv = std::max(1e-9, prior_var[i]);
            precision[i][i] = 1.0 / pv;
            info_vec[i] = prior_mean[i] / pv;
        }
        prior_precision = precision;
        prior_info = info_vec;
        refresh();
    }

    void refresh() {
        cov = invert_matrix(precision);
        mean = solve_linear(precision, info_vec);
    }

    std::pair<double, double> predict(const std::vector<double> &x) const {
        double mu = 0.0;
        for (size_t i = 0; i < dim; ++i) mu += mean[i] * x[i];
        double var = noise_var;
        for (size_t i = 0; i < dim; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < dim; ++j) row_sum += cov[i][j] * x[j];
            var += x[i] * row_sum;
        }
        return {std::max(0.0, mu), std::sqrt(std::max(0.0, var))};
    }

    bool update(const std::vector<double> &x, double y, double weight = 1.0, bool robust = true) {
        if (x.size() != dim || !std::isfinite(y) || y < 0.0) return false;
        double eff_weight = weight;
        if (robust && n_obs >= 3) {
            auto [mu, sd] = predict(x);
            const double err = y - mu;
            const double abs_err = std::abs(err);
            const double z = abs_err / std::max(sd, 150.0);
            const double rel_err = abs_err / std::max(mu, 300.0);
            const int sign = (err >= 0.0) ? 1 : -1;
            if (z > 3.0 && rel_err > 0.8) {
                if (sign == last_outlier_sign_) {
                    ++consecutive_outliers_;
                } else {
                    consecutive_outliers_ = 1;
                    last_outlier_sign_ = sign;
                }
                if (consecutive_outliers_ < 2) {
                    return false;
                }
                eff_weight *= 0.5;
            } else {
                consecutive_outliers_ = 0;
                last_outlier_sign_ = 0;
                if (z > 1.5) {
                    eff_weight *= 1.5 / z;
                }
            }
        }
        const double lam = forgetting;
        const double inv_s2 = eff_weight / noise_var;
        for (size_t i = 0; i < dim; ++i) {
            info_vec[i] = lam * info_vec[i] + (1.0 - lam) * prior_info[i] + inv_s2 * x[i] * y;
            for (size_t j = 0; j < dim; ++j) {
                precision[i][j] =
                    lam * precision[i][j] + (1.0 - lam) * prior_precision[i][j] + inv_s2 * x[i] * x[j];
            }
        }
        ++n_obs;
        refresh();
        return true;
    }

private:
    int consecutive_outliers_ = 0;
    int last_outlier_sign_ = 0;
};

struct M1JointDecision {
    std::string path = "cc";  // "cc" | "hc_full" | "hc_compact"
    std::string route = "cpu";
    bool kv_admit = false;
    int keep_tokens = 0;
    int h_active = 0;
    int k_floor = 0;
    std::string reason = "m1_lcb_agree";
    bool explored = false;
    double cc_mean_ms = 0.0;
    double cc_std_ms = 0.0;
    double cc_lcb_ms = 0.0;
    double hc_full_mean_ms = 0.0;
    double hc_full_std_ms = 0.0;
    double hc_full_lcb_ms = 0.0;
    bool hc_compact_valid = false;
    double hc_compact_mean_ms = 0.0;
    double hc_compact_std_ms = 0.0;
    double hc_compact_lcb_ms = 0.0;
    double handoff_saving_ms = 0.0;
    double decode_saving_ms = 0.0;
    double prior_tax_ms = 0.0;
    double evict_ms = 0.0;
};

class M1JointExplorer {
public:
    int routing_p_min = 32;
    double beta = 1.25;
    double max_explore_penalty_ms = 4000.0;
    int min_obs_before_guard = 6;
    bool exploration_enabled = true;
    bool online_update_enabled = true;

    BayesianLinearComponent cpu_restore;
    BayesianLinearComponent cpu_to_htp;
    BayesianLinearComponent cpu_prefill;
    BayesianLinearComponent htp_prefill;
    BayesianLinearComponent htp_to_cpu;
    BayesianLinearComponent decode_per_tok;

    M1JointExplorer() { reset_defaults(); }

    void reset_defaults(double forgetting = 0.98) {
        cpu_restore = BayesianLinearComponent(2, {40.0, 0.09}, {2500.0, 0.02}, 80.0, forgetting);
        cpu_to_htp = BayesianLinearComponent(2, {40.0, 0.12}, {2500.0, 0.04}, 150.0, forgetting);
        cpu_prefill = BayesianLinearComponent(3, {100.0, 60.0, 8.0}, {250000.0, 900.0, 100.0}, 600.0, forgetting);
        htp_prefill = BayesianLinearComponent(3, {250.0, 30.0, 7.0}, {250000.0, 900.0, 100.0}, 500.0, forgetting);
        htp_to_cpu = BayesianLinearComponent(2, {50.0, 0.25}, {40000.0, 0.16}, 300.0, forgetting);
        decode_per_tok = BayesianLinearComponent(2, {40.0, 0.008}, {2500.0, 0.0004}, 10.0, forgetting);
    }

    void seed_from_linear_table(const RouteLinearCostTable &rt, const KvReduceCostTable &kt) {
        if (rt.valid) {
            routing_p_min = rt.routing_p_min;
            cpu_restore = BayesianLinearComponent(
                2, {rt.cpu_restore.intercept_ms, rt.cpu_restore.slope_ms_per_token}, {2500.0, 0.02}, 80.0);
            cpu_to_htp = BayesianLinearComponent(
                2, {rt.cpu_to_htp.intercept_ms, rt.cpu_to_htp.slope_ms_per_token}, {2500.0, 0.04}, 150.0);
            cpu_prefill = BayesianLinearComponent(
                3, {rt.cpu_prefill.intercept_ms, rt.cpu_prefill.slope_ms_per_token, 4.0}, {250000.0, 900.0, 100.0}, 600.0);
            htp_prefill = BayesianLinearComponent(
                3, {rt.htp_prefill.intercept_ms, rt.htp_prefill.slope_ms_per_token, 4.0}, {250000.0, 900.0, 100.0}, 500.0);
            htp_to_cpu = BayesianLinearComponent(
                2, {rt.htp_to_cpu.intercept_ms, rt.htp_to_cpu.slope_ms_per_token}, {40000.0, 0.16}, 300.0);
        }
        if (kt.valid) {
            decode_per_tok = BayesianLinearComponent(
                2, {kt.decode_per_tok.intercept_ms, kt.decode_per_tok.slope_ms_per_token}, {2500.0, 0.0004}, 10.0);
        }
    }

    static std::vector<double> prefill_feat(int h, int p) {
        return {1.0, double(p), (double(h) / 1000.0) * double(p)};
    }

    std::pair<double, double> predict_decode(int h_active, int expected_d, const KvReduceCostTable &kt) const {
        if (expected_d <= 0) return {0.0, 0.0};
        if (decode_per_tok.n_obs >= 2 || !kt.valid) {
            auto [mu_tok, sd_tok] = decode_per_tok.predict({1.0, double(h_active)});
            return {double(expected_d) * mu_tok, double(expected_d) * sd_tok};
        }
        return {double(expected_d) * kt.decode_per_tok.eval(double(h_active)), 0.0};
    }

    M1JointDecision select_joint(
            int h,
            int p,
            int expected_d,
            const KvReduceCostTable &kv_model,
            bool htp_allowed,
            bool kv_enabled,
            bool htp_ready,
            int h_logical = -1,
            int pinned_tokens = 0,
            int sink_tokens = 4,
            int recent_tokens = 32) const {
        const int total = h + p;
        auto [r_mu, r_sd] = cpu_restore.predict({1.0, double(h)});
        auto [cp_mu, cp_sd] = cpu_prefill.predict(prefill_feat(h, p));
        auto [dec_full_mu, dec_full_sd] = predict_decode(total, expected_d, kv_model);

        const double cc_mu = r_mu + cp_mu + dec_full_mu;
        const double cc_sd = std::sqrt(r_sd * r_sd + cp_sd * cp_sd + dec_full_sd * dec_full_sd);
        const double cc_lcb = cc_mu - beta * cc_sd;

        auto [hp_mu, hp_sd] = htp_prefill.predict(prefill_feat(h, p));
        auto [ho_full_mu, ho_full_sd] = htp_to_cpu.predict({1.0, double(total)});
        const double hc_full_mu = hp_mu + ho_full_mu + dec_full_mu;
        const double hc_full_sd = std::sqrt(hp_sd * hp_sd + ho_full_sd * ho_full_sd + dec_full_sd * dec_full_sd);
        const double hc_full_lcb = hc_full_mu - beta * hc_full_sd;

        const int logical_h = (h_logical >= 0) ? std::max(h, h_logical) : h;
        const int logical_total = logical_h + p;
        const int k_min = kv_model.valid ? kv_model.quality_k_min : 1536;
        const double ratio_min = kv_model.valid ? kv_model.quality_keep_ratio_min : 0.25;
        const int align = kv_model.valid ? kv_model.align_pad : 256;
        const int min_evict = kv_model.valid ? kv_model.min_evict_savings_tokens : 512;
        const double min_gain = kv_model.valid ? kv_model.min_gain_ms : 100.0;
        const int ratio_raw = static_cast<int>(std::ceil(ratio_min * double(logical_total)));
        const int eff_sink = std::max(sink_tokens, pinned_tokens);
        const int k_floor = std::max({k_min, eff_sink + recent_tokens, pad_align_up(ratio_raw, align)});

        M1JointDecision out;
        out.keep_tokens = total;
        out.h_active = total;
        out.k_floor = k_floor;
        out.cc_mean_ms = cc_mu;
        out.cc_std_ms = cc_sd;
        out.cc_lcb_ms = cc_lcb;
        out.hc_full_mean_ms = hc_full_mu;
        out.hc_full_std_ms = hc_full_sd;
        out.hc_full_lcb_ms = hc_full_lcb;

        bool hc_compact_eligible = false;
        int best_k = total;
        if (kv_enabled && kv_model.valid && total > k_floor + min_evict) {
            std::vector<int> cands;
            for (int c : kv_model.candidates) {
                if (c >= k_floor && c + min_evict <= total) cands.push_back(c);
            }
            for (int off : {0, 512, 1024, 2048}) {
                int c = k_floor + off;
                if (c + min_evict <= total) cands.push_back(c);
            }
            std::sort(cands.begin(), cands.end());
            cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
            if (!cands.empty()) {
                out.prior_tax_ms = kv_model.htp_prior_tax.eval(double(p));
                out.evict_ms = kv_model.evict.eval(double(total));
                for (int k : cands) {
                    auto [ho_k_mu, ho_k_sd] = htp_to_cpu.predict({1.0, double(k)});
                    auto [dec_k_mu, dec_k_sd] = predict_decode(k, expected_d, kv_model);
                    const double cand_mu = hp_mu + out.prior_tax_ms + out.evict_ms + ho_k_mu + dec_k_mu;
                    const double cand_sd = std::sqrt(hp_sd * hp_sd + ho_k_sd * ho_k_sd + dec_k_sd * dec_k_sd);
                    if (!out.hc_compact_valid || cand_mu < out.hc_compact_mean_ms) {
                        out.hc_compact_valid = true;
                        out.hc_compact_mean_ms = cand_mu;
                        out.hc_compact_std_ms = cand_sd;
                        out.hc_compact_lcb_ms = cand_mu - beta * cand_sd;
                        best_k = k;
                        out.handoff_saving_ms = ho_full_mu - ho_k_mu;
                        out.decode_saving_ms = dec_full_mu - dec_k_mu;
                    }
                }
                if (out.hc_compact_valid && (hc_full_mu - out.hc_compact_mean_ms) > min_gain) {
                    hc_compact_eligible = true;
                }
            }
        }

        if (!htp_allowed) {
            out.reason = "fixed_backend";
            return out;
        }
        if (p < routing_p_min) {
            out.reason = "below_routing_p_min";
            return out;
        }
        if (!htp_ready) {
            out.reason = "htp_not_ready";
            return out;
        }

        struct Cand {
            std::string path;
            std::string route;
            bool kv_admit;
            int h_active;
            double mean_ms;
            double lcb_ms;
        };
        std::vector<Cand> cands = {
            {"cc", "cpu", false, total, cc_mu, cc_lcb},
            {"hc_full", "htp", false, total, hc_full_mu, hc_full_lcb},
        };
        if (hc_compact_eligible) {
            cands.push_back({"hc_compact", "htp", true, best_k, out.hc_compact_mean_ms, out.hc_compact_lcb_ms});
        }

        auto greedy_it = std::min_element(cands.begin(), cands.end(),
            [](const Cand &a, const Cand &b) { return a.mean_ms < b.mean_ms; });
        if (!exploration_enabled) {
            out.path = greedy_it->path;
            out.route = greedy_it->route;
            out.kv_admit = greedy_it->kv_admit;
            out.keep_tokens = greedy_it->h_active;
            out.h_active = greedy_it->h_active;
            out.reason = "m1_greedy_mean";
            return out;
        }

        auto lcb_it = std::min_element(cands.begin(), cands.end(),
            [](const Cand &a, const Cand &b) { return a.lcb_ms < b.lcb_ms; });
        const bool explored = (lcb_it->path != greedy_it->path);
        auto chosen = lcb_it;
        std::string reason = explored ? "m1_lcb_explore" : "m1_lcb_agree";
        if (explored) {
            const double mean_gap = lcb_it->mean_ms - greedy_it->mean_ms;
            const bool enough_data =
                cpu_prefill.n_obs >= min_obs_before_guard && htp_prefill.n_obs >= min_obs_before_guard;
            if (enough_data && mean_gap > max_explore_penalty_ms) {
                chosen = greedy_it;
                reason = "m1_safe_guard_greedy";
            } else {
                out.explored = true;
            }
        }
        out.path = chosen->path;
        out.route = chosen->route;
        out.kv_admit = chosen->kv_admit;
        out.keep_tokens = chosen->h_active;
        out.h_active = chosen->h_active;
        out.reason = reason;
        return out;
    }

    bool observe(
            int h,
            int p,
            const std::string &route,
            int h_active,
            double restore_ms,
            double prepare_ms,
            double prefill_ms,
            double handoff_ms,
            double decode_ms,
            int predicted_n,
            bool collect_prior) {
        if (!online_update_enabled) return false;
        bool updated = false;
        if (prepare_ms > 0.0) {
            updated |= cpu_to_htp.update({1.0, double(h)}, prepare_ms);
        }
        if (route == "cpu") {
            if (restore_ms >= 0.0) updated |= cpu_restore.update({1.0, double(h)}, restore_ms);
            if (prefill_ms >= 0.0 && p > 0) updated |= cpu_prefill.update(prefill_feat(h, p), prefill_ms);
        } else if (route == "htp") {
            if (prefill_ms >= 0.0 && p > 0 && !collect_prior) {
                updated |= htp_prefill.update(prefill_feat(h, p), prefill_ms);
            }
            if (handoff_ms >= 0.0) {
                updated |= htp_to_cpu.update({1.0, double(h_active)}, handoff_ms);
            }
        }
        if (decode_ms >= 0.0 && predicted_n > 0) {
            updated |= decode_per_tok.update({1.0, double(h_active)}, decode_ms / double(predicted_n));
        }
        return updated;
    }
};

}  // namespace a2o_m1
