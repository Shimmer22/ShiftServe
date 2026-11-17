#include "scheduling/kv_reduce_cost.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

static KvReduceCostTable sample_table() {
    KvReduceCostTable t;
    t.valid = true;
    t.expected_decode_tokens = 96;
    t.quality_k_min = 1536;
    t.quality_keep_ratio_min = 0.25;
    t.align_pad = 256;
    t.min_evict_savings_tokens = 512;
    t.min_gain_ms = 100.0;
    t.candidates = {1536, 2048, 2560, 3072, 4096};
    t.decode_per_tok = {41.74211604447541, 0.006359819563889574};
    t.handoff = {102.90720255782043, 0.09779201752845793};
    t.evict = {0.3391706124856704, 0.000274793336113983};
    t.htp_prior_tax = {655.6884210187457, 1.1448914769712997};
    t.cpu_prior_tax = {612.7050909996033, 0.0};
    return t;
}

int main() {
    auto t = sample_table();

    auto p_short = plan_kv_reduce(t, 1024, 256, 96, false);
    assert(!p_short.admit && p_short.reason == "below_quality_or_evict_floor" && p_short.keep_tokens == 1280);

    auto p_2k = plan_kv_reduce(t, 2048, 256, 96, false);
    assert(!p_2k.admit && p_2k.reason == "negative_net_gain" && p_2k.keep_tokens == 1536);

    auto p_low_d = plan_kv_reduce(t, 4096, 256, 32, true);
    assert(!p_low_d.admit && p_low_d.reason == "negative_net_gain");

    auto p_4k = plan_kv_reduce(t, 4096, 256, 96, false);
    assert(p_4k.admit && p_4k.keep_tokens == 1536 && p_4k.k_floor == 1536);

    auto p_6k = plan_kv_reduce(t, 6144, 256, 96, true);
    assert(p_6k.admit && p_6k.keep_tokens == 1792 && p_6k.k_floor == 1792);

    auto p_8k = plan_kv_reduce(t, 8192, 256, 96, true);
    assert(p_8k.admit && p_8k.keep_tokens == 2304 && p_8k.k_floor == 2304);

    // Logical H_anchor + pinned_tokens prevents k_floor collapse across compact turns.
    auto p_compact = plan_kv_reduce(t, 1536, 512, 96, false, 8192, 0);
    assert(!p_compact.admit && p_compact.reason == "below_quality_or_evict_floor" && p_compact.k_floor == 2304);
    auto p_pinned = plan_kv_reduce(t, 4096, 512, 96, false, 4096, 3800);
    assert(p_pinned.k_floor == 3800);

    // Emit CSV-like values for cross-language parity verification.
    RouteLinearCostTable rm;
    rm.valid = true;
    rm.routing_p_min = 32;
    rm.h_min = 128;
    rm.h_max = 1536;
    rm.p_min = 1;
    rm.p_max = 384;
    rm.cpu_restore = {50.0, 0.05};
    rm.cpu_prefill = {20.0, 12.0};
    rm.htp_prefill = {80.0, 8.0};
    rm.htp_to_cpu = {102.90720255782043, 0.09779201752845793};
    rm.cpu_to_htp = {100.0, 0.1};
    // Joint case: H_eff=6144, P=256, D=96 -> HTP+KV wins over CPU+KV
    auto jp_htp = plan_joint_route_kv(rm, t, 6144, 256, 96, true, true, "cpu");
    assert(jp_htp.route == "htp" && jp_htp.reason == "joint_route_kv_cost" && jp_htp.kv_plan.admit && jp_htp.kv_plan.keep_tokens == 1792);
    // Below routing_p_min -> CPU+KV fallback
    auto jp_small_p = plan_joint_route_kv(rm, t, 6144, 16, 96, true, true, "cpu");
    assert(jp_small_p.route == "cpu" && jp_small_p.reason == "below_routing_p_min" && jp_small_p.kv_plan.admit);

    std::cout << std::setprecision(15)
              << p_4k.keep_tokens << "," << p_4k.expected_gain_ms << ","
              << p_8k.keep_tokens << "," << p_8k.expected_gain_ms << ","
              << jp_htp.route << "," << jp_htp.selected_total_ms << ","
              << jp_small_p.route << "," << jp_small_p.selected_total_ms << "\n";
    return 0;
}
