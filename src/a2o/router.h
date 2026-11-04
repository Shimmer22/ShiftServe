#pragma once
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct RouteCost {
    int h=0, p=0, d=0, samples=0;
    std::string path;
    double future_ms=0, entry_ms=0;
};
struct RouteDecision {
    std::string backend="cpu", reason="coverage_missing", availability;
    double cpu_ms=std::numeric_limits<double>::quiet_NaN();
    double htp_ms=std::numeric_limits<double>::quiet_NaN();
};

// 首版限于 CPU 权威状态、CPU decode 的 CC/HC；精确匹配，不外推。
class CostRouter {
public:
    std::string source;
    int context=0, batch=0, queue=0, opbatch=0;
    std::vector<RouteCost> rows;
    static CostRouter load(const std::string & path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("router table unavailable");
        CostRouter result; std::string line, magic;
        if (!std::getline(input,line)) throw std::runtime_error("empty router table");
        std::istringstream header(line);
        if (!(header >> magic >> result.source >> result.context >> result.batch >> result.queue >> result.opbatch)
                || magic != "a2o-router-v1") throw std::runtime_error("invalid router table header");
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            RouteCost r; std::istringstream in(line);
            if (!(in >> r.h >> r.p >> r.d >> r.path >> r.future_ms >> r.entry_ms >> r.samples)
                    || r.h<0 || r.p<=0 || r.d<=0 || r.samples<0 || (r.samples<3 && result.source!="mechanism-fixture")
                    || (r.path!="CC" && r.path!="HC")
                    || !std::isfinite(r.future_ms) || r.future_ms<0
                    || !std::isfinite(r.entry_ms) || r.entry_ms<0)
                throw std::runtime_error("invalid router cost row");
            std::string trailing;
            if (in >> trailing) throw std::runtime_error("unexpected router cost fields");
            for (const auto & old: result.rows)
                if (old.h==r.h && old.p==r.p && old.d==r.d && old.path==r.path)
                    throw std::runtime_error("duplicate router cost row");
            result.rows.push_back(r);
        }
        return result;
    }
    RouteDecision choose(int h, int p, int d, const std::string & availability) const {
        RouteDecision result; result.availability=availability;
        const RouteCost * cc=nullptr, *hc=nullptr;
        for (const auto & r:rows) if (r.h==h && r.p==p && r.d==d) {
            if(r.path=="CC") cc=&r; else if(r.path=="HC") hc=&r;
        }
        if (cc) result.cpu_ms=cc->future_ms;
        // 不可抢占 API 没有可靠剩余时间；首版不等待未就绪候选。
        if (availability!="ready") { result.reason="candidate_"+availability; return result; }
        if (!cc || !hc) return result;
        result.htp_ms=hc->future_ms; // 历史准备已完成，不能重复计费。
        if (result.htp_ms < result.cpu_ms) { result.backend="htp"; result.reason="htp_lower_cost"; }
        else result.reason="cpu_lower_cost";
        return result;
    }
    // Tool 开始时未知 P，仅在同一 H/D 有合格且可能获益的输入时准备。
    bool should_prepare(int h, int d) const {
        for (const auto & hc:rows) if (hc.h==h && hc.d==d && hc.path=="HC")
            for (const auto & cc:rows) if (cc.h==h && cc.p==hc.p && cc.d==d && cc.path=="CC"
                                          && hc.future_ms<cc.future_ms) return true;
        return false;
    }
};
