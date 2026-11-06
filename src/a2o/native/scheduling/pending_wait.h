#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <string>

struct PendingWaitPlan {
    bool eligible=false;
    double budget_ms=0;
    std::string reason="wait_disabled";
};

// 不读取运行中准备线程的完成时间或进度。upper由独立校准提供，
// 保守按全额准备计准入成本；cap限制试等，margin保留估计误差余量。
inline PendingWaitPlan plan_pending_wait(bool qualified, double cc, double hc,
                                         double upper, double cap, double margin) {
    PendingWaitPlan p;
    if(!qualified) {p.reason="candidate_unqualified";return p;}
    if(!std::isfinite(cc)||!std::isfinite(hc)||!std::isfinite(upper)||
       !std::isfinite(cap)||!std::isfinite(margin)||cc<=0||hc<0||upper<=0||cap<=0||margin<0) {
        p.reason="invalid_wait_cost";return p;
    }
    if(hc+upper+margin>=cc) {p.reason="insufficient_cost_headroom";return p;}
    p.eligible=true;p.budget_ms=std::min(cap,cc-hc-margin);p.reason="bounded_pending_wait";
    return p;
}

inline bool await_pending(std::future<void>& worker, const PendingWaitPlan& p) {
    return p.eligible && worker.valid() &&
        worker.wait_for(std::chrono::duration<double,std::milli>(p.budget_ms))==std::future_status::ready;
}
