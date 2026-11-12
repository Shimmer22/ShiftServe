#include "scheduling/greedy_return.h"
#include "state/preparation_lane.h"
#include <cassert>
#include <future>
#include <limits>

int main() {
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
