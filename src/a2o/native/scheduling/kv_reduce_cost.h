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

inline KvReducePlan plan_kv_reduce(const KvReduceCostTable &model, int h, int p, int expected_d, bool is_htp) {
    if (!model.valid) {
        throw std::runtime_error("kv reduce cost model unavailable");
    }
    if (h < 0 || p < 0 || expected_d < 0) {
        throw std::runtime_error("negative token count in plan_kv_reduce");
    }
    const int total = h + p;
    const int ratio_raw = static_cast<int>(std::ceil(model.quality_keep_ratio_min * double(total)));
    const int k_floor = std::max(model.quality_k_min, pad_align_up(ratio_raw, model.align_pad));
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
    cands.reserve(model.candidates.size() + 1);
    for (int c : model.candidates) {
        if (c >= k_floor && c + model.min_evict_savings_tokens <= total) cands.push_back(c);
    }
    if (k_floor + model.min_evict_savings_tokens <= total) {
        cands.push_back(k_floor);
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
    const double hnd_full = is_htp ? model.handoff.eval(double(total)) : 0.0;

    bool have_best = false;
    KvReducePlan best;
    for (int k : cands) {
        const double dec_k = model.decode_per_tok.eval(double(k));
        const double hnd_k = is_htp ? model.handoff.eval(double(k)) : 0.0;
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
