#include "scheduling/execution_preparation.h"
#include "state/preparation_lane.h"
#include <cassert>
#include <cstdio>
#include <future>

// ABI 假后端：不分配图/模型，只在调用内部阻塞并模拟预算错误。
struct llama_context {
    std::mutex mutex;
    std::condition_variable cv;
    bool block=false,entered=false,released=false,inflight=false,allocated=false;
    uint64_t descriptor_bytes=1,map_bytes=1;
    size_t calls=0,discards=0;
    a2o_execution_stats stats{};
    ~llama_context() {assert(!inflight && !allocated);}
};
static int control(llama_context *ctx,uint32_t action,const a2o_execution_request *request,a2o_execution_stats *out) {
    std::unique_lock<std::mutex> lock(ctx->mutex);
    if(action==A2O_EXEC_PREPARE) {
        ++ctx->calls;ctx->inflight=ctx->entered=ctx->allocated=true;ctx->cv.notify_all();
        if(ctx->block) ctx->cv.wait(lock,[&] {return ctx->released;});
        ctx->inflight=false;
        if(ctx->descriptor_bytes>request->descriptor_budget || ctx->map_bytes>request->map_budget) {
            ctx->stats.status=A2O_EXEC_FAILED;
            snprintf(ctx->stats.reason,sizeof(ctx->stats.reason),"budget_exceeded");
            *out=ctx->stats;return -1;
        }
        ctx->stats.status=A2O_EXEC_READY;
    } else if(action==A2O_EXEC_DISCARD) {
        assert(!ctx->inflight);++ctx->discards;ctx->allocated=false;ctx->stats.status=A2O_EXEC_EMPTY;
    }
    *out=ctx->stats;return 0;
}

static void cancel_during_api() {
    llama_context ctx;ctx.block=true;
    ExecutionPreparation plan(&ctx,true,control);
    PreparationLane lane;
    assert(lane.start(1,[&](const auto &valid) {return plan.prepare_cancellable(valid,1,256,64,0);}));
    {std::unique_lock<std::mutex> lock(ctx.mutex);ctx.cv.wait(lock,[&] {return ctx.entered;});}
    lane.retire();
    // API 仍未返回：前台可继续 CPU/发布，不 join、不 discard 正在用的上下文。
    assert(lane.availability(2)=="busy");
    assert(!lane.start(2,[](const auto &) {return true;}));
    assert(ctx.inflight && ctx.discards==0 && lane.reaped()==0);
    std::promise<void> closing;
    auto closed=std::async(std::launch::async,[&] {closing.set_value();lane.close();});
    closing.get_future().wait();
    assert(closed.wait_for(std::chrono::seconds(0))==std::future_status::timeout);
    {std::lock_guard<std::mutex> lock(ctx.mutex);ctx.released=true;ctx.cv.notify_all();}
    closed.get();
    assert(!ctx.allocated && !ctx.inflight && ctx.discards==1 && lane.reaped()==1);
    ctx.block=false;
    assert(lane.start(2,[&](const auto &valid) {return plan.prepare_cancellable(valid,2,256,64,0);}));
    lane.wait_for_test();lane.claim(2); // 回收完成后下一轮才可复用。
    assert(plan.query().status==A2O_EXEC_READY);
    plan.discard();
    assert(lane.started()==lane.reaped());
}

static void budget_and_failure_cleanup() {
    llama_context ctx;
    ExecutionPreparation plan(&ctx,true,control);
    assert(!plan.prepare_cancellable([] {return false;},1,256,64,0) && ctx.calls==0);
    for(int kind=0;kind<2;++kind) {
        ctx.descriptor_bytes=kind==0?(8ull<<20)+1:1;
        ctx.map_bytes=kind==1?(4ull<<30)+1:1;
        bool failed=false;
        try {plan.prepare_cancellable([] {return true;},1,256,64,0);}
        catch(const std::runtime_error &e) {failed=std::string(e.what()).find("budget_exceeded")!=std::string::npos;}
        assert(failed && !ctx.allocated && !ctx.inflight);
    }
    // worker 内的错误也向 lane 使用者传播，已分配的计划先释放。
    {
        ctx.block=true;ctx.entered=ctx.released=false;
        PreparationLane lane;
        assert(lane.start(3,[&](const auto &valid) {return plan.prepare_cancellable(valid,3,256,64,0);}));
        {std::unique_lock<std::mutex> lock(ctx.mutex);ctx.cv.wait(lock,[&] {return ctx.entered;});}
        {std::lock_guard<std::mutex> lock(ctx.mutex);ctx.released=true;ctx.cv.notify_all();}
        bool failed=false;
        try {lane.close();} catch(const std::runtime_error &) {failed=true;}
        assert(failed && !ctx.allocated);
    }
    ctx.block=false;
    ctx.descriptor_bytes=ctx.map_bytes=1;
    assert(plan.prepare_cancellable([] {return true;},4,256,64,0));
    assert(ctx.allocated);plan.discard();
}

int main(int argc,char **argv) {
    assert(argc==2);std::string name=argv[1];
    if(name=="cancel") cancel_during_api();
    else if(name=="failure") budget_and_failure_cleanup();
    else return 2;
}
