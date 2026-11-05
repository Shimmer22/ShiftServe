#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

// 会话内状态，不持久化；source/执行条件/H/P/D/路径精确隔离。
struct OnlineCostKey {
    std::string source, regime, path;
    int h=0, p=0, d=0;
    bool operator==(const OnlineCostKey & b) const {
        return h==b.h && p==b.p && d==b.d && source==b.source && regime==b.regime && path==b.path;
    }
};
struct OnlineCostHash {
    size_t operator()(const OnlineCostKey & k) const {
        size_t h=0;
        auto mix=[&](size_t v) {h^=v+0x9e3779b9+(h<<6)+(h>>2);};
        for(const auto * s:{&k.source,&k.regime,&k.path}) mix(std::hash<std::string>{}(*s));
        for(int n:{k.h,k.p,k.d}) mix(std::hash<int>{}(n));
        return h;
    }
};
struct OnlineCostConfig {
    bool enabled=false;
    double alpha=.2, ttl_ms=300000;
    size_t min_samples=3, every=8, capacity=256;
};

class OnlineRouteCosts {
public:
    explicit OnlineRouteCosts(OnlineCostConfig config={}):config_(config) {
        if(!std::isfinite(config.alpha) || config.alpha<=0 || config.alpha>1
                || !std::isfinite(config.ttl_ms) || config.ttl_ms<=0
                || !config.min_samples || !config.every || !config.capacity)
            throw std::runtime_error("invalid online cost configuration");
    }
    bool enabled() const {return config_.enabled;}
    // 冷桶按有效样本数决定采集；失败/短输出不会消耗冷桶名额。
    bool sample(const OnlineCostKey & key,double now) {
        if(!config_.enabled) return false;
        auto & s=state(key,now);
        if(s.samples<config_.min_samples) return true;
        if(++s.skipped>=config_.every) {s.skipped=0;return true;}
        return false;
    }
    bool update(const OnlineCostKey & key,double future_ms,double now) {
        if(!config_.enabled || !std::isfinite(future_ms) || future_ms<0) return false;
        auto & s=state(key,now);
        s.mean=s.samples?s.mean+config_.alpha*(future_ms-s.mean):future_ms;
        ++s.samples;s.updated=now;s.touch=++serial_;return true;
    }
    double predict(const OnlineCostKey & key,double now) const {
        auto it=values_.find(key);
        if(!config_.enabled || it==values_.end() || expired(it->second,now)
                || it->second.samples<config_.min_samples) return std::numeric_limits<double>::quiet_NaN();
        return it->second.mean;
    }
    size_t samples(const OnlineCostKey & key,double now) const {
        auto it=values_.find(key);
        return it==values_.end() || expired(it->second,now)?0:it->second.samples;
    }
    size_t size() const {return values_.size();}
private:
    struct State {double mean=0,updated=0;size_t samples=0,skipped=0;uint64_t touch=0;};
    OnlineCostConfig config_;
    std::unordered_map<OnlineCostKey,State,OnlineCostHash> values_;
    uint64_t serial_=0;
    bool expired(const State & s,double now) const {return now-s.updated>config_.ttl_ms;}
    State & state(const OnlineCostKey & key,double now) {
        auto it=values_.find(key);
        if(it==values_.end()) {
            if(values_.size()>=config_.capacity) {
                auto oldest=values_.begin();
                for(auto p=values_.begin();p!=values_.end();++p)
                    if(p->second.touch<oldest->second.touch) oldest=p;
                values_.erase(oldest);
            }
            it=values_.emplace(key,State{}).first;
            it->second.updated=now;it->second.touch=++serial_;
        }
        if(expired(it->second,now)) {it->second=State{};it->second.updated=now;it->second.touch=++serial_;}
        return it->second;
    }
};

// 只有第 D 次 evaluation 完成时读额外一次时钟；EOG 本身若被评估也计一次。
class DecodeHorizon {
public:
    DecodeHorizon(bool sampled,int target,double start):sampled_(sampled),target_(target),start_(start) {}
    template<class Clock> void evaluated(int count,Clock clock) {
        if(sampled_ && count==target_) prefix_ms_=clock()-start_;
    }
    double prefix_ms() const {return prefix_ms_;}
private:
    bool sampled_;
    int target_;
    double start_,prefix_ms_=std::numeric_limits<double>::quiet_NaN();
};
