#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

struct KvLinearTerm {
    double intercept_ms = 0.0;
    double slope_ms_per_token = 0.0;

    double eval(double x) const {
        if (!std::isfinite(intercept_ms) || !std::isfinite(slope_ms_per_token) || !std::isfinite(x)) {
            throw std::runtime_error("nonfinite kv cost input");
        }
        return std::max(0.0, intercept_ms + slope_ms_per_token * x);
    }
};

struct KvReduceCostTable {
    bool valid = false;
    bool handoff_saving_enabled = false;
    int expected_decode_tokens = 96;
    int quality_k_min = 1536;
    double quality_keep_ratio_min = 0.25;
    int align_pad = 256;
    int min_evict_savings_tokens = 512;
    double min_gain_ms = 100.0;
    std::vector<int> candidates = {1536, 2048, 2560, 3072, 4096};
    KvLinearTerm decode_per_tok;
    KvLinearTerm handoff;
    KvLinearTerm evict;
    KvLinearTerm htp_prior_tax;
    KvLinearTerm cpu_prior_tax;
};

struct KvReducePlan {
    bool admit = false;
    int keep_tokens = 0;
    int k_floor = 0;
    int h_total = 0;
    double expected_gain_ms = 0.0;
    double handoff_saving_ms = 0.0;
    double decode_saving_ms = 0.0;
    double prior_tax_ms = 0.0;
    double evict_ms = 0.0;
    std::string reason = "disabled";
};

inline int pad_align_up(int n, int align) {
    if (align <= 1) return n;
    return ((n + align - 1) / align) * align;
}

inline KvReducePlan plan_kv_reduce(
        const KvReduceCostTable &model,
        int h,
        int p,
        int expected_d,
        bool is_htp,
        int h_logical = -1,
        int pinned_tokens = 0) {
    if (!model.valid) {
        throw std::runtime_error("kv reduce cost model unavailable");
    }
    if (h < 0 || p < 0 || expected_d < 0 || pinned_tokens < 0) {
        throw std::runtime_error("negative token count in plan_kv_reduce");
    }
    const int total = h + p;
    const int logical_h = (h_logical >= 0) ? std::max(h, h_logical) : h;
    const int logical_total = logical_h + p;
    const int ratio_raw = static_cast<int>(std::ceil(model.quality_keep_ratio_min * double(logical_total)));
    const int k_floor = std::max({model.quality_k_min, pinned_tokens, pad_align_up(ratio_raw, model.align_pad)});
    if (total <= k_floor + model.min_evict_savings_tokens) {
        KvReducePlan p_floor;
        p_floor.admit = false;
        p_floor.keep_tokens = total;
        p_floor.k_floor = k_floor;
        p_floor.h_total = total;
        p_floor.reason = "below_quality_or_evict_floor";
        return p_floor;
    }

    std::vector<int> cands;
    cands.reserve(model.candidates.size() + 4);
    for (int c : model.candidates) {
        if (c >= k_floor && c + model.min_evict_savings_tokens <= total) cands.push_back(c);
    }
    for (int off : {0, 512, 1024, 2048}) {
        int c = k_floor + off;
        if (c + model.min_evict_savings_tokens <= total) cands.push_back(c);
    }
    std::sort(cands.begin(), cands.end());
    cands.erase(std::unique(cands.begin(), cands.end()), cands.end());

    if (cands.empty()) {
        KvReducePlan p_none;
        p_none.admit = false;
        p_none.keep_tokens = total;
        p_none.k_floor = k_floor;
        p_none.h_total = total;
        p_none.reason = "no_valid_keep_candidate";
        return p_none;
    }

    const double tax_ms = is_htp ? model.htp_prior_tax.eval(double(p)) : model.cpu_prior_tax.eval(double(p));
    const double evict_ms = model.evict.eval(double(total));
    const double dec_full = model.decode_per_tok.eval(double(total));
    const bool use_hnd_saving = is_htp && model.handoff_saving_enabled;
    const double hnd_full = use_hnd_saving ? model.handoff.eval(double(total)) : 0.0;

    bool have_best = false;
    KvReducePlan best;
    for (int k : cands) {
        const double dec_k = model.decode_per_tok.eval(double(k));
        const double hnd_k = use_hnd_saving ? model.handoff.eval(double(k)) : 0.0;
        const double hnd_saving = hnd_full - hnd_k;
        const double dec_saving = double(expected_d) * (dec_full - dec_k);
        const double gain = hnd_saving + dec_saving - tax_ms - evict_ms;
        if (!have_best || gain > best.expected_gain_ms) {
            have_best = true;
            best.keep_tokens = k;
            best.k_floor = k_floor;
            best.h_total = total;
            best.expected_gain_ms = gain;
            best.handoff_saving_ms = hnd_saving;
            best.decode_saving_ms = dec_saving;
            best.prior_tax_ms = tax_ms;
            best.evict_ms = evict_ms;
        }
    }

    best.admit = best.expected_gain_ms > model.min_gain_ms;
    best.reason = best.admit ? "cost_table_admitted" : "negative_net_gain";
    return best;
}

template <typename JsonT>
inline KvLinearTerm parse_kv_linear_term(const JsonT &obj) {
    KvLinearTerm term;
    term.intercept_ms = obj.at("intercept_ms").template get<double>();
    term.slope_ms_per_token = obj.at("slope_ms_per_token").template get<double>();
    if (!std::isfinite(term.intercept_ms) || !std::isfinite(term.slope_ms_per_token)) {
        throw std::runtime_error("nonfinite linear term in kv_reduce_cost_v1");
    }
    return term;
}

template <typename JsonT>
inline KvReduceCostTable parse_kv_reduce_cost_table(const JsonT &m) {
    if (m.value("kind", std::string("")) != "kv_reduce_cost_v1") {
        throw std::runtime_error("unsupported kv cost model kind");
    }
    KvReduceCostTable t;
    t.valid = true;
    t.handoff_saving_enabled = m.value("handoff_saving_enabled", false);
    t.expected_decode_tokens = m.value("expected_decode_tokens", 96);
    t.quality_k_min = m.value("quality_k_min", 1536);
    t.quality_keep_ratio_min = m.value("quality_keep_ratio_min", 0.25);
    t.align_pad = m.value("align_pad", 256);
    t.min_evict_savings_tokens = m.value("min_evict_savings_tokens", 512);
    t.min_gain_ms = m.value("min_gain_ms", 100.0);
    if (m.contains("candidates")) {
        t.candidates = m.at("candidates").template get<std::vector<int>>();
    }
    t.decode_per_tok = parse_kv_linear_term(m.at("decode_per_tok"));
    t.handoff = parse_kv_linear_term(m.at("handoff"));
    t.evict = parse_kv_linear_term(m.at("evict"));
    t.htp_prior_tax = parse_kv_linear_term(m.at("htp_prior_tax"));
    t.cpu_prior_tax = parse_kv_linear_term(m.at("cpu_prior_tax"));
    return t;
}

template <typename JsonT>
inline JsonT kv_reduce_plan_to_json(const KvReducePlan &p) {
    return JsonT{
        {"admit", p.admit},
        {"keep_tokens", p.keep_tokens},
        {"k_floor", p.k_floor},
        {"H_total", p.h_total},
        {"expected_gain_ms", p.expected_gain_ms},
        {"handoff_saving_ms", p.handoff_saving_ms},
        {"decode_saving_ms", p.decode_saving_ms},
        {"prior_tax_ms", p.prior_tax_ms},
        {"evict_ms", p.evict_ms},
        {"reason", p.reason},
    };
}

struct RouteLinearCostTable {
    bool valid = false;
    int routing_p_min = 32;
    size_t h_min = 0;
    size_t h_max = 0;
    size_t p_min = 0;
    size_t p_max = 0;
    KvLinearTerm cpu_restore;
    KvLinearTerm cpu_prefill;
    KvLinearTerm htp_prefill;
    KvLinearTerm htp_to_cpu;
    KvLinearTerm cpu_to_htp;
};

struct JointRouteKvPlan {
    std::string route = "cpu";
    std::string reason = "fixed_backend";
    bool extrapolated = false;
    int h_eff = 0;
    int p = 0;
    int expected_d = 0;
    double cc_ms = 0.0;
    double hc_ms = 0.0;
    double cpu_no_kv_total_ms = 0.0;
    double htp_no_kv_total_ms = 0.0;
    double cpu_best_total_ms = 0.0;
    double htp_best_total_ms = 0.0;
    double selected_total_ms = 0.0;
    KvReducePlan kv_plan;
    KvReducePlan cpu_kv_plan;
    KvReducePlan htp_kv_plan;
};

template <typename JsonT>
inline RouteLinearCostTable parse_route_linear_cost_table(const JsonT &m) {
    if (m.value("kind", std::string("")) != "linear_hp_v1") {
        throw std::runtime_error("unsupported route cost model kind");
    }
    RouteLinearCostTable r;
    r.valid = true;
    r.routing_p_min = m.value("routing_p_min", 32);
    r.h_min = m.value("h_min", size_t(0));
    r.h_max = m.value("h_max", size_t(0));
    r.p_min = m.value("p_min", size_t(0));
    r.p_max = m.value("p_max", size_t(0));
    r.cpu_restore = parse_kv_linear_term(m.at("cpu_restore"));
    r.cpu_prefill = parse_kv_linear_term(m.at("cpu_prefill"));
    r.htp_prefill = parse_kv_linear_term(m.at("htp_prefill"));
    r.htp_to_cpu = parse_kv_linear_term(m.at("htp_to_cpu"));
    r.cpu_to_htp = parse_kv_linear_term(m.at("cpu_to_htp"));
    return r;
}

inline JointRouteKvPlan plan_joint_route_kv(
        const RouteLinearCostTable &route_model,
        const KvReduceCostTable &kv_model,
        int h_eff,
        int p,
        int expected_d,
        bool htp_allowed,
        bool kv_enabled,
        const std::string &default_route = "cpu",
        int h_logical = -1,
        int pinned_tokens = 0) {
    if (h_eff < 0 || p < 0 || expected_d < 0 || pinned_tokens < 0) {
        throw std::runtime_error("negative token count in plan_joint_route_kv");
    }
    const int total = h_eff + p;
    JointRouteKvPlan out;
    out.h_eff = h_eff;
    out.p = p;
    out.expected_d = expected_d;

    double hnd_full = 0.0;
    if (route_model.valid) {
        out.cc_ms = route_model.cpu_restore.eval(double(h_eff)) + route_model.cpu_prefill.eval(double(p));
        hnd_full = route_model.htp_to_cpu.eval(double(total));
        out.hc_ms = route_model.htp_prefill.eval(double(p)) + hnd_full;
        out.extrapolated =
            size_t(h_eff) < route_model.h_min ||
            size_t(total) > route_model.h_max ||
            size_t(p) < route_model.p_min ||
            (route_model.p_max > 0 && size_t(p) > route_model.p_max);
    } else if (kv_model.valid) {
        hnd_full = kv_model.handoff.eval(double(total));
        out.hc_ms = hnd_full;
    }

    const double dec_full_cost = kv_model.valid ? double(expected_d) * kv_model.decode_per_tok.eval(double(total)) : 0.0;
    out.cpu_no_kv_total_ms = out.cc_ms + dec_full_cost;
    out.htp_no_kv_total_ms = out.hc_ms + dec_full_cost;
    out.cpu_best_total_ms = out.cpu_no_kv_total_ms;
    out.htp_best_total_ms = out.htp_no_kv_total_ms;

    if (kv_enabled && kv_model.valid) {
        out.cpu_kv_plan = plan_kv_reduce(kv_model, h_eff, p, expected_d, false, h_logical, pinned_tokens);
        if (out.cpu_kv_plan.admit) {
            out.cpu_best_total_ms = out.cpu_no_kv_total_ms - out.cpu_kv_plan.expected_gain_ms;
        }
        out.htp_kv_plan = plan_kv_reduce(kv_model, h_eff, p, expected_d, true, h_logical, pinned_tokens);
        if (out.htp_kv_plan.admit) {
            out.htp_best_total_ms = out.htp_no_kv_total_ms - out.htp_kv_plan.expected_gain_ms;
        }
    }

    if (!htp_allowed) {
        out.route = default_route;
        const bool is_htp = (default_route == "htp");
        out.kv_plan = is_htp ? out.htp_kv_plan : out.cpu_kv_plan;
        out.selected_total_ms = is_htp ? out.htp_best_total_ms : out.cpu_best_total_ms;
        out.reason = "fixed_backend";
        return out;
    }

    if (route_model.valid && p < route_model.routing_p_min) {
        out.route = "cpu";
        out.kv_plan = out.cpu_kv_plan;
        out.selected_total_ms = out.cpu_best_total_ms;
        out.reason = "below_routing_p_min";
        return out;
    }

    if (out.htp_best_total_ms < out.cpu_best_total_ms) {
        out.route = "htp";
        out.kv_plan = out.htp_kv_plan;
        out.selected_total_ms = out.htp_best_total_ms;
        out.reason = (kv_enabled && kv_model.valid && out.htp_kv_plan.admit) ? "joint_route_kv_cost" : "linear_hp_cost";
    } else {
        out.route = "cpu";
        out.kv_plan = out.cpu_kv_plan;
        out.selected_total_ms = out.cpu_best_total_ms;
        out.reason = (kv_enabled && kv_model.valid && out.cpu_kv_plan.admit) ? "joint_route_kv_cost" : "linear_hp_cost";
    }
    return out;
}
