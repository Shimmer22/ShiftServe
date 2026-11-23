#include "scheduling/greedy_return.h"
#include "scheduling/m1_explorer.h"
#include "state/preparation_lane.h"
#include <cassert>
#include <future>
#include <limits>

void test_pending_prefers_expected_cost() {
    a2o_m1::M1JointExplorer explorer;
    explorer.cpu_prefill = {3, {5000, 0, 0}, {16000000, 0, 0}, 1};
    explorer.htp_prefill = {3, {2500, 0, 0}, {1, 0, 0}, 1};
    const auto ready = explorer.select_joint(1600, 100, 0, {}, true, false, true);
    assert(ready.path == "cc" && ready.explored && ready.cc_mean_ms > ready.hc_full_mean_ms);
    const auto pending = explorer.select_joint(1600, 100, 0, {}, true, false, true, 1600, 0, 4, 100, true, false);
    assert(pending.path == "hc_full" && !pending.explored);
    assert(plan_greedy_return(false, 50, 450, pending.cc_mean_ms, pending.hc_full_mean_ms).continue_prepare);
    assert(!plan_greedy_return(false, 50, 4500, pending.cc_mean_ms, pending.hc_full_mean_ms).continue_prepare);
}

void test_compact_looks_ahead_with_reusable_history() {
    a2o_m1::M1JointExplorer explorer;
    KvReduceCostTable model;
    model.valid = true;
    model.min_evict_savings_tokens = 256;
    model.htp_prior_tax = {200, 0};
    const auto current_only = explorer.select_joint(2213, 326, 1, model, true, true, true, 2213, 252, 4, 326, false);
    assert(current_only.path == "hc_full" && !current_only.hc_compact_eligible);
    const auto next_round = explorer.select_joint(2213, 326, 1, model, true, true, true, 2213, 252, 4, 326);
    assert(next_round.path == "hc_compact" && next_round.hc_compact_eligible);
    assert(next_round.single_turn_gain_ms < model.min_gain_ms);
    assert(next_round.delta_next_prefill_ms > 0 && next_round.effective_gain_ms > model.min_gain_ms);
    const auto short_history = explorer.select_joint(128, 2000, 1, model, true, true, true, 128, 128, 4, 2000);
    assert(!short_history.hc_compact_valid && !short_history.kv_admit);
    const auto small_evict = explorer.select_joint(1300, 450, 1, model, true, true, true, 1300, 252, 4, 450);
    assert(!small_evict.hc_compact_valid && !small_evict.kv_admit);
}

int main() {
    test_pending_prefers_expected_cost();
    test_compact_looks_ahead_with_reusable_history();
    auto p=plan_greedy_return(false,2,200,30000,20000);
    assert(p.continue_prepare && p.remaining_ms==198 && p.continue_cost_ms==20198 && p.wait_budget_ms==10000);
    assert(!plan_greedy_return(false,2,200,20198,20000).continue_prepare); // 打平选CPU
    assert(!plan_greedy_return(false,0,1000,500,100).continue_prepare);
    assert(plan_greedy_return(true,0,1000,500,100).continue_prepare);
    assert(plan_greedy_return(false,500,100,500,100).remaining_ms==0); // 预测已到期但仍pending
    bool invalid=false;
    try {plan_greedy_return(false,0,std::numeric_limits<double>::quiet_NaN(),500,100);} catch(...) {invalid=true;}
    assert(invalid);
    PreparationLane lane;
    auto gate=std::make_shared<PreparationBarrier>();
    assert(lane.start(1,[gate](const auto &valid){gate->enter();return valid();}));
    gate->wait_entered();
    assert(!lane.wait_ready(2,10)); // 不能等待/绑定旧版本
    assert(!lane.wait_ready(1,1)); // 有界超时，不把pending当ready
    gate->release();assert(lane.wait_ready(1,1000));lane.claim(1);assert(!lane.occupied());
    assert(lane.start(2,[](const auto &)->bool{throw std::runtime_error("backend failed");}));
    bool failed=false;
    try {lane.wait_ready(2,1000);} catch(...) {failed=true;}
    assert(failed); // 设备失败必须传播，不能伪装为放弃策略
    try {lane.close();} catch(...) {}
}
