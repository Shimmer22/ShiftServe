#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

struct GreedyReturnPlan {
    bool continue_prepare=false;
    double remaining_ms=0,continue_cost_ms=0,wait_budget_ms=0;
    std::string reason;
};

// 返回时仅使用独立预测和已经流逝的时间，不读取本次准备的未来完成时间。
inline GreedyReturnPlan plan_greedy_return(bool ready,double elapsed_ms,double prepare_ms,
                                          double cpu_cost_ms,double htp_cost_ms) {
    if(!std::isfinite(elapsed_ms)||!std::isfinite(prepare_ms)||
       !std::isfinite(cpu_cost_ms)||!std::isfinite(htp_cost_ms)||
       elapsed_ms<0||prepare_ms<0||cpu_cost_ms<=0||htp_cost_ms<=0)
        throw std::runtime_error("invalid greedy return prediction");
    GreedyReturnPlan p;
    p.remaining_ms=ready?0:std::max(0.0,prepare_ms-elapsed_ms);
    p.continue_cost_ms=p.remaining_ms+htp_cost_ms;
    p.continue_prepare=p.continue_cost_ms<cpu_cost_ms;
    p.wait_budget_ms=p.continue_prepare && !ready?cpu_cost_ms-htp_cost_ms:0;
    p.reason=p.continue_prepare?(ready?"greedy_ready_htp":"greedy_continue_prepare"):"greedy_abandon_prepare";
    return p;
}
