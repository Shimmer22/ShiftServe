// 编入私有 llama-context.cpp，位于所有头文件之后。无 llama_context 字段扩展。
#include "a2o-execution-preparation.h"
#include "llama-kv-cache.h"
#include "../ggml/src/ggml-impl.h"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <cstdio>
extern "C" int a2o_sched_execution_prepare_v1(void *,uint32_t,const a2o_execution_request *,a2o_execution_stats *,a2o_backend_prepare_callback);
static int a2o_backend_dispatch(void *opaque,void *graph,uint32_t action,
        const a2o_execution_request *r,a2o_execution_stats *s) {
    auto backend=(ggml_backend_t)opaque;auto device=ggml_backend_get_device(backend);
    auto reg=device?ggml_backend_dev_backend_reg(device):nullptr;
    using fn_type=int(*)(ggml_backend_t,ggml_cgraph *,uint32_t,const a2o_execution_request *,a2o_execution_stats *);
    auto fn=reg?(fn_type)ggml_backend_reg_get_proc_address(reg,"a2o_backend_execution_prepare_v1"):nullptr;
    if(!fn) return 1;
    return fn(backend,(ggml_cgraph*)graph,action,r,s);
}
static int a2o_sched_prepare(void *sched,uint32_t action,const a2o_execution_request *r,a2o_execution_stats *s) {
    return a2o_sched_execution_prepare_v1(sched,action,r,s,a2o_backend_dispatch);
}
struct a2o_context_plan {
    std::mutex mutex;
    a2o_execution_request request{};
    a2o_execution_stats stats{};
    std::unique_ptr<llama_batch_allocr> balloc;
    llama_memory_context_ptr preview;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<int8_t> output;
    bool bound=false;
};
static std::mutex a2o_contexts_mutex;
static std::unordered_map<llama_context*,std::shared_ptr<a2o_context_plan>> a2o_contexts;
static std::atomic<bool> a2o_any_context{false};
static std::shared_ptr<a2o_context_plan> a2o_context_find(llama_context *ctx,bool create) {
    if(!create && !a2o_any_context.load()) return {};
    std::lock_guard<std::mutex> lock(a2o_contexts_mutex);
    auto it=a2o_contexts.find(ctx);if(it!=a2o_contexts.end()) return it->second;
    if(!create) return {};
    auto p=std::make_shared<a2o_context_plan>();a2o_contexts.emplace(ctx,p);a2o_any_context.store(true);return p;
}
static void a2o_context_forget(llama_context *ctx) {
    std::shared_ptr<a2o_context_plan> plan;
    {std::lock_guard<std::mutex> lock(a2o_contexts_mutex);
     auto it=a2o_contexts.find(ctx);if(it==a2o_contexts.end()) return;
     plan=it->second;a2o_contexts.erase(it);a2o_any_context.store(!a2o_contexts.empty());}
    a2o_sched_prepare(ctx->get_sched(),A2O_EXEC_DISCARD,nullptr,&plan->stats);
}
static void a2o_reason(a2o_context_plan &p,uint32_t status,const char *text) {
    p.stats.status=status;snprintf(p.stats.reason,sizeof(p.stats.reason),"%s",text);
}
// 在真实 ubatch 已 apply、但尚未构图/写输入前检查；未知输入从未在准备阶段执行。
static bool a2o_execution_before_ubatch(llama_context *ctx,const llama_ubatch &ubatch) {
    auto p=a2o_context_find(ctx,false);if(!p) return false;
    std::lock_guard<std::mutex> lock(p->mutex);
    if(p->stats.status!=A2O_EXEC_READY && p->stats.status!=A2O_EXEC_BOUND) return false;
    bool match=p->bound && ubatch.token && !ubatch.embd && ubatch.n_seqs_unq==1 &&
        ubatch.seq_id_unq && ubatch.seq_id_unq[0]==0 && ubatch.n_tokens==p->request.n_tokens && ubatch.pos &&
        ubatch.pos[0]==p->request.next_pos && ubatch.output;
    for(uint32_t i=0;match && i<ubatch.n_tokens;i++)
        match=ubatch.pos[i]==p->request.next_pos+i && ubatch.output[i]==(i+1==ubatch.n_tokens);
    if(!match) {
        a2o_reason(*p,A2O_EXEC_MISS,"unbound_or_input_layout_changed");p->bound=false;
        a2o_sched_prepare(ctx->get_sched(),A2O_EXEC_DISCARD,nullptr,&p->stats);
        // 使普通 process_ubatch 路径重建；不可复用持有 preview mctx 的输入对象。
        return true; // 强制普通构图分支销毁旧输入；preview 在此之前保持存活。
    }
    return false;
}
static void a2o_execution_graph_decision(llama_context *ctx,bool reuse,uint64_t uid) {
    auto p=a2o_context_find(ctx,false);if(!p) return;
    std::lock_guard<std::mutex> lock(p->mutex);
    if(p->stats.status==A2O_EXEC_BOUND && p->bound) {
        if(reuse && uid==p->stats.graph_uid) {p->stats.graph_hits++;a2o_reason(*p,A2O_EXEC_CONSUMED,"graph_reused");}
        else {a2o_reason(*p,A2O_EXEC_MISS,reuse?"allocated_graph_uid_changed":"runtime_graph_rebuilt");a2o_sched_prepare(ctx->get_sched(),A2O_EXEC_DISCARD,nullptr,&p->stats);}
        p->bound=false;
    }
    // preview 必须至少活到 can_reuse 已将输入 mctx 重绑，或 reset 已销毁旧输入对象之后。
}
int llama_context::a2o_execution_control(uint32_t action,const a2o_execution_request *r,a2o_execution_stats *out) {
    if(!out) return -1;
    auto p=a2o_context_find(this,action==A2O_EXEC_PREPARE);
    if(!p) {*out={};out->abi=A2O_EXEC_ABI;out->size=sizeof(*out);return action==A2O_EXEC_QUERY || action==A2O_EXEC_DISCARD?0:-2;}
    std::lock_guard<std::mutex> lock(p->mutex);
    if(action==A2O_EXEC_QUERY) {
        *out=p->stats;out->descriptor_hits=0;out->descriptor_misses=0;
        return a2o_sched_prepare(sched.get(),A2O_EXEC_QUERY,nullptr,out);
    }
    if(action==A2O_EXEC_DISCARD) {
        synchronize();a2o_sched_prepare(sched.get(),A2O_EXEC_DISCARD,nullptr,&p->stats);
        gf_res_prev->reset();ggml_backend_sched_reset(sched.get());
        p->preview.reset();p->balloc.reset();p->bound=false;a2o_reason(*p,A2O_EXEC_EMPTY,"discarded");*out=p->stats;return 0;
    }
    if(!r || r->abi!=A2O_EXEC_ABI || r->size!=sizeof(*r)) return -3;
    if(action==A2O_EXEC_BIND) {
        const auto start=ggml_time_us();
        bool match=p->stats.status==A2O_EXEC_READY && r->version==p->request.version &&
            r->next_pos==p->request.next_pos && r->n_tokens==p->request.n_tokens &&
            memory && memory->seq_pos_max(0)+1==r->next_pos && !sched_need_reserve;
        if(match) {p->bound=true;a2o_reason(*p,A2O_EXEC_BOUND,"shape_and_version_match");}
        else {
            a2o_reason(*p,A2O_EXEC_MISS,"shape_or_version_changed");p->bound=false;
            a2o_sched_prepare(sched.get(),A2O_EXEC_DISCARD,nullptr,&p->stats);
            gf_res_prev->reset();ggml_backend_sched_reset(sched.get());p->preview.reset();p->balloc.reset();
        }
        p->stats.bind_ms=(ggml_time_us()-start)/1000.;*out=p->stats;return 0;
    }
    if(action!=A2O_EXEC_PREPARE) return -4;
    p->stats={};p->stats.abi=A2O_EXEC_ABI;p->stats.size=sizeof(p->stats);p->stats.version=r->version;
    try {
        if(model.arch!=LLM_ARCH_QWEN2VL || !memory || cparams.pipeline_parallel || cparams.embeddings ||
           cparams.n_seq_max!=1 || !sampling.samplers.empty() || !r->n_tokens || r->n_tokens>cparams.n_ubatch ||
           r->n_tokens>cparams.n_batch || r->next_pos<0 || (uint64_t)r->next_pos>n_ctx() || r->n_tokens>n_ctx()-(uint64_t)r->next_pos ||
           memory->seq_pos_max(0)+1!=r->next_pos || graph_reuse_disable || !r->descriptor_budget || !r->map_budget)
            throw std::runtime_error("unsupported prediction context or shape");
        synchronize();
        auto update=memory->init_update(this,false);
        if(!update || update->get_status()!=LLAMA_MEMORY_STATUS_NO_UPDATE)
            throw std::runtime_error("pending memory update must be handled before prediction");
        a2o_sched_prepare(sched.get(),A2O_EXEC_DISCARD,nullptr,&p->stats);
        sched_reserve();gf_res_prev->reset();ggml_backend_sched_reset(sched.get());
        p->preview.reset();p->balloc.reset();p->request=*r;p->bound=false;
        p->tokens.assign(r->n_tokens,0);p->positions.resize((size_t)r->n_tokens*model.hparams.n_pos_per_embd());p->output.assign(r->n_tokens,0);p->output.back()=1;
        for(uint32_t axis=0;axis<model.hparams.n_pos_per_embd();axis++)
            for(uint32_t i=0;i<r->n_tokens;i++) p->positions[axis*r->n_tokens+i]=r->next_pos+i;
        llama_batch dummy{};dummy.n_tokens=r->n_tokens;dummy.token=p->tokens.data();dummy.pos=p->positions.data();dummy.logits=p->output.data();
        p->balloc=std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd());
        if(!p->balloc->init(dummy,model.vocab,memory.get(),model.hparams.n_embd_inp(),1,false))
            throw std::runtime_error("preview batch allocation failed");
        p->preview=memory->init_batch(*p->balloc,cparams.n_ubatch,false);
        if(!p->preview || p->preview->get_status()!=LLAMA_MEMORY_STATUS_SUCCESS)
            throw std::runtime_error("preview cannot fit into current KV");
        // Qwen2VL 使用普通 llama_kv_cache；其它 memory 类型在上面的架构准入中拒绝。
        static_cast<llama_kv_cache_context*>(p->preview.get())->a2o_preview_graph();
        const auto &ubatch=p->preview->get_ubatch();
        const auto save_outputs=n_outputs;n_outputs=1;
        auto start=ggml_time_us();
        auto params=graph_params(gf_res_prev.get(),ubatch,p->preview.get(),LLM_GRAPH_TYPE_DECODER);
        ggml_cgraph *graph=nullptr;
        try {graph=model.build_graph(params);} catch(...) {n_outputs=save_outputs;throw;}
        n_outputs=save_outputs;p->stats.graph_ms=(ggml_time_us()-start)/1000.;
        if(!graph) throw std::runtime_error("prediction graph build failed");
        p->stats.graph_nodes=graph->n_nodes;
        start=ggml_time_us();
        if(!ggml_backend_sched_alloc_graph(sched.get(),graph)) throw std::runtime_error("prediction buffer allocation failed");
        p->stats.graph_uid=graph->uid; // scheduler 分图会更新 uid，必须在 alloc_graph 之后记录。
        // 预留 CPU 输出空间；不写入 token、position 或 mask 张量。
        if(output_reserve(1)<1) throw std::runtime_error("prediction output allocation failed");
        p->stats.allocate_ms=(ggml_time_us()-start)/1000.;start=ggml_time_us();
        if(a2o_sched_prepare(sched.get(),A2O_EXEC_PREPARE,r,&p->stats))
            throw std::runtime_error(p->stats.reason[0]?p->stats.reason:"backend descriptor preparation failed");
        p->stats.backend_ms=(ggml_time_us()-start)/1000.;a2o_reason(*p,A2O_EXEC_READY,"prepared_without_compute");*out=p->stats;return 0;
    } catch(const std::exception &error) {
        a2o_reason(*p,A2O_EXEC_FAILED,error.what());
        a2o_sched_prepare(sched.get(),A2O_EXEC_DISCARD,nullptr,&p->stats);
        gf_res_prev->reset();ggml_backend_sched_reset(sched.get());p->preview.reset();p->balloc.reset();p->bound=false;
        *out=p->stats;return -5;
    }
}
extern "C" LLAMA_API int a2o_execution_control_v1(llama_context *ctx,uint32_t action,
        const a2o_execution_request *r,a2o_execution_stats *out) {
    if(!ctx) return -1;
    try {return ctx->a2o_execution_control(action,r,out);}
    catch(const std::exception &e) {if(out) {out->status=A2O_EXEC_FAILED;snprintf(out->reason,sizeof(out->reason),"%s",e.what());}return -6;}
}
