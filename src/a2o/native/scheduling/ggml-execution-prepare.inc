// 编入私有 ggml-backend.cpp。直接访问已分配的真实 split，扩展解析交给 libllama 回调。
#include "a2o-execution-preparation.h"
extern "C" GGML_API int a2o_sched_execution_prepare_v1(void *opaque,uint32_t action,
        const a2o_execution_request *request,a2o_execution_stats *stats,a2o_backend_prepare_callback callback) {
    auto *sched=(ggml_backend_sched_t)opaque;
    if(!sched || !stats || !callback) return -1;
    bool found=false;
    if(action!=A2O_EXEC_PREPARE) {
        for(int i=0;i<sched->n_backends;i++) {
            const int result=callback(sched->backends[i],nullptr,action,request,stats);
            if(result<0) return result;
            if(result==0) found=true;
        }
        return found?0:-3;
    }
    if(!request || sched->n_copies!=1 || sched->callback_eval) return -4;
    for(int i=0;i<sched->n_splits;i++) {
        auto &split=sched->splits[i];
        const int result=callback(sched->backends[split.backend_id],&split.graph,action,request,stats);
        if(result<0) return result;
        if(result==0) found=true;
        else for(int n=0;n<split.graph.n_nodes;n++) {
            auto *node=split.graph.nodes[n];
            if(!ggml_op_is_empty(node->op) && !ggml_is_empty(node) && (node->flags&GGML_TENSOR_FLAG_COMPUTE)) stats->cpu_nodes++;
        }
    }
    return found && stats->htp_nodes?0:-6;
}
