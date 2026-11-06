#include "scheduling/pending_wait.h"
#include <cassert>
#include <limits>

int main() {
    auto p=plan_pending_wait(true,2600,1900,180,100,20);
    assert(p.eligible && p.budget_ms==100);
    assert(!plan_pending_wait(true,1900,2600,180,100,20).eligible);
    assert(!plan_pending_wait(true,2100,1900,180,100,20).eligible); // 恰好吃完余量也不准入
    assert(!plan_pending_wait(false,2600,1900,180,100,20).eligible);
    assert(!plan_pending_wait(true,2600,1900,std::numeric_limits<double>::quiet_NaN(),100,20).eligible);
    assert(!plan_pending_wait(true,2600,1900,180,0,20).eligible);
    auto short_budget=plan_pending_wait(true,100,20,30,100,20);
    assert(short_budget.eligible && short_budget.budget_ms==60); // 预算不超过经济余量
    std::promise<void> done;auto ready=done.get_future();done.set_value();
    assert(await_pending(ready,p));ready.get();
    std::promise<void> incomplete;auto pending=incomplete.get_future();
    auto tiny=plan_pending_wait(true,2600,1900,180,1,20);
    assert(!await_pending(pending,tiny)); // 超时仍未完成，不可选HTP
    assert(pending.wait_for(std::chrono::milliseconds(0))==std::future_status::timeout);
    incomplete.set_value();pending.get();
}
