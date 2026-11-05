#pragma once
#include "backend/execution-preparation.h"
#include <algorithm>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

// 必须由同一条 PreparationLane 独占 HTP context。析构前先 join lane。
class ExecutionPreparation {
public:
    ExecutionPreparation(llama_context *ctx,bool enabled,a2o_execution_control_fn control=nullptr)
        :ctx_(ctx),enabled_(enabled),control_(control) {
        if(enabled_) {
            if(!control_) control_=(a2o_execution_control_fn)dlsym(RTLD_DEFAULT,"a2o_execution_control_v1");
            if(!control_) throw std::runtime_error("private execution-preparation runtime unavailable");
        }
    }
    bool enabled() const {return enabled_;}
    void discard() {
        if(enabled_ && control_(ctx_,A2O_EXEC_DISCARD,nullptr,&last_))
            throw std::runtime_error("execution plan discard failed");
    }
    a2o_execution_stats prepare(uint64_t version,int64_t pos,uint32_t tokens,uint64_t prefault_bytes) {
        if(!enabled_) return {};
        a2o_execution_request r{};r.abi=A2O_EXEC_ABI;r.size=sizeof(r);r.version=version;r.next_pos=pos;r.n_tokens=tokens;
        r.flags=A2O_EXEC_DESCRIPTORS|A2O_EXEC_MAP|(prefault_bytes?A2O_EXEC_PREFAULT:0);
        r.descriptor_budget=8ull<<20;r.map_budget=4ull<<30;r.prefault_budget=prefault_bytes;
        if(control_(ctx_,A2O_EXEC_PREPARE,&r,&last_)) throw std::runtime_error(std::string("execution plan preparation failed: ")+last_.reason);
        if(last_.status!=A2O_EXEC_READY) throw std::runtime_error("execution plan did not become ready");
        return last_;
    }
    a2o_execution_stats bind(uint64_t version,int64_t pos,uint32_t tokens) {
        if(!enabled_) return {};
        a2o_execution_request r{};r.abi=A2O_EXEC_ABI;r.size=sizeof(r);r.version=version;r.next_pos=pos;r.n_tokens=tokens;
        if(control_(ctx_,A2O_EXEC_BIND,&r,&last_)) throw std::runtime_error("execution plan bind failed");
        return last_;
    }
    // 调用不可抢占；取消后由同一 worker 在 API 返回边界释放，不与前台争用 context。
    template<class Valid>
    bool prepare_cancellable(Valid valid,uint64_t version,int64_t pos,uint32_t tokens,uint64_t prefault_bytes,
                             a2o_execution_stats *stats=nullptr) {
        if(!valid()) return false;
        a2o_execution_stats prepared{};
        try {
            prepared=prepare(version,pos,tokens,prefault_bytes);
        } catch(...) {discard();throw;}
        if(!valid()) {discard();return false;}
        if(stats) *stats=prepared;
        return true;
    }
    a2o_execution_stats query() {
        if(enabled_ && control_(ctx_,A2O_EXEC_QUERY,nullptr,&last_)) throw std::runtime_error("execution plan query failed");
        return last_;
    }
    ~ExecutionPreparation() {try {discard();} catch(...) {}}
private:
    llama_context *ctx_;
    bool enabled_;
    a2o_execution_control_fn control_=nullptr;
    a2o_execution_stats last_{};
};
