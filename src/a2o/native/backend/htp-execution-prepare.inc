// 编入私有 ggml-hexagon.cpp，位于 op_is_compute 之后；准备入口从不 enqueue/flush_batch。
#include "a2o-execution-preparation.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <array>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unistd.h>

struct a2o_tensor_stamp {
    uint32_t node_index;
    int32_t source;
    const ggml_tensor *object;
    const void *data;
    ggml_backend_buffer_t buffer;
    int type,op;
    std::array<int64_t,4> ne;
    std::array<size_t,4> nb;
    std::array<const ggml_tensor*,GGML_MAX_SRC> src;
    std::array<unsigned char,GGML_MAX_OP_PARAMS> params;
};
static a2o_tensor_stamp a2o_stamp(const ggml_tensor *t) {
    a2o_tensor_stamp s{};s.object=t;s.data=t->data;s.buffer=t->buffer;s.type=t->type;s.op=t->op;
    std::copy(t->ne,t->ne+4,s.ne.begin());std::copy(t->nb,t->nb+4,s.nb.begin());
    std::copy(t->src,t->src+GGML_MAX_SRC,s.src.begin());
    memcpy(s.params.data(),t->op_params,s.params.size());return s;
}
static bool a2o_same_stamp(const a2o_tensor_stamp &old,const ggml_tensor *t) {
    auto s=a2o_stamp(t);
    return old.object==s.object && old.data==s.data && old.buffer==s.buffer && old.type==s.type &&
        old.op==s.op && old.ne==s.ne && old.nb==s.nb && old.src==s.src && old.params==s.params;
}
struct a2o_prepared_split {
    std::vector<std::unique_ptr<ggml_hexagon_opbatch>> batches;
    std::vector<a2o_tensor_stamp> stamps;
    std::vector<const ggml_tensor*> nodes;
};
struct a2o_prepared_session {
    std::unordered_map<uint64_t,std::unique_ptr<a2o_prepared_split>> splits;
    std::set<std::pair<int,size_t>> mapped;
    std::set<ggml_backend_buffer_t> touched;
    uint64_t descriptor_hits=0,descriptor_misses=0;
};
static std::atomic<bool> a2o_any_session{false};
static std::mutex a2o_sessions_mutex;
static std::unordered_map<ggml_hexagon_session*,std::shared_ptr<a2o_prepared_session>> a2o_sessions;
static std::shared_ptr<a2o_prepared_session> a2o_session_plan(ggml_hexagon_session *sess,bool create) {
    if(!create && !a2o_any_session.load()) return {};
    std::lock_guard<std::mutex> lock(a2o_sessions_mutex);
    auto it=a2o_sessions.find(sess);
    if(it!=a2o_sessions.end()) return it->second;
    if(!create) return {};
    auto plan=std::make_shared<a2o_prepared_session>();a2o_sessions.emplace(sess,plan);a2o_any_session.store(true);return plan;
}
static void a2o_forget_session(ggml_hexagon_session *sess) {
    std::lock_guard<std::mutex> lock(a2o_sessions_mutex);a2o_sessions.erase(sess);a2o_any_session.store(!a2o_sessions.empty());
}
static void a2o_prefault_buffer(a2o_prepared_session &plan,ggml_backend_buffer_t buffer,
        const a2o_execution_request &r,a2o_execution_stats &stats) {
    if(!(r.flags&A2O_EXEC_PREFAULT) || !r.prefault_budget || !plan.touched.insert(buffer).second ||
       ggml_backend_buffer_get_usage(buffer)!=GGML_BACKEND_BUFFER_USAGE_COMPUTE) return;
    auto *base=(volatile unsigned char*)ggml_backend_buffer_get_base(buffer);
    size_t size=ggml_backend_buffer_get_size(buffer);
    const size_t page=(size_t)sysconf(_SC_PAGESIZE);
    if(!base || !page || page>(1u<<20)) throw std::runtime_error("invalid compute buffer page geometry");
    // 仅写临时 compute buffer；不写权重、普通KV或任意用户映射。预算按触及页字节计。
    for(size_t offset=0;offset<size && stats.prefault_bytes<r.prefault_budget;) {
        const size_t bytes=std::min(page-((uintptr_t)(base+offset)%page),size-offset);
        if(bytes>r.prefault_budget-stats.prefault_bytes) break;
        base[offset]=0;stats.prefault_bytes+=bytes;offset+=bytes;
    }
}
static int a2o_backend_execution_prepare(ggml_backend_t backend,ggml_cgraph *graph,uint32_t action,
        const a2o_execution_request *request,a2o_execution_stats *stats) {
    auto *sess=static_cast<ggml_hexagon_session*>(backend->context);
    if(!stats) return -1;
    if(action==A2O_EXEC_DISCARD) {a2o_forget_session(sess);return 0;}
    auto plan=a2o_session_plan(sess,action==A2O_EXEC_PREPARE);
    if(action==A2O_EXEC_QUERY) {
        if(plan) {stats->descriptor_hits+=plan->descriptor_hits;stats->descriptor_misses+=plan->descriptor_misses;}
        return 0;
    }
    if(action!=A2O_EXEC_PREPARE || !request || !graph || !(request->flags&A2O_EXEC_DESCRIPTORS) ||
       !sess->op_batch->empty() || sess->op_pending.load()!=0 || (opt_opstage&(HTP_OPSTAGE_QUEUE|HTP_OPSTAGE_COMPUTE))!=(HTP_OPSTAGE_QUEUE|HTP_OPSTAGE_COMPUTE)) return -2;
    try {
        if(plan->splits.count(graph->uid)) return -3;
        auto split=std::make_unique<a2o_prepared_split>();
        auto make_batch=[&] {return std::make_unique<ggml_hexagon_opbatch>(sess,sess->op_batch->n_ops_max,sess->op_batch->b_vmem_max);};
        auto batch=make_batch();uint64_t bytes=0;
        for(int i=0;i<graph->n_nodes;i++) {
            auto *node=graph->nodes[i];split->nodes.push_back(node);
            if(!op_is_compute(node)) continue;
            stats->htp_nodes++;
            if(!batch->fit_op(node)) {
                if(batch->empty()) return -4;
                split->batches.push_back(std::move(batch));batch=make_batch();
                if(!batch->fit_op(node)) return -5;
            }
            auto remember=[&](const ggml_tensor *t,int source) {
                if(!t || !t->buffer || !t->data) throw std::runtime_error("unallocated descriptor tensor");
                auto *sbuf=static_cast<ggml_hexagon_shared_buffer*>(t->buffer->context);
                if(!sbuf || sbuf->sess!=sess || !sbuf->mapped) throw std::runtime_error("tensor belongs to another HTP session");
                auto stamp=a2o_stamp(t);stamp.node_index=i;stamp.source=source;split->stamps.push_back(stamp);
                if((request->flags&A2O_EXEC_MAP) && plan->mapped.insert({sbuf->fd,sbuf->size}).second) {
                    if(sbuf->size>request->map_budget-stats->mapped_bytes) throw std::runtime_error("mapping budget exceeded");
                    if(htp_iface_mmap(sess->handle,sbuf->fd,sbuf->size)) throw std::runtime_error("HTP page mapping failed");
                    stats->mapped_buffers++;stats->mapped_bytes+=sbuf->size;
                }
                a2o_prefault_buffer(*plan,t->buffer,*request,*stats);
            };
            remember(node,-1);
            for(unsigned j=0;j<HTP_OP_MAX_INPUTS && node->src[j];j++) remember(node->src[j],j);
            batch->add_op(op_remap_to_htp(node),node);stats->descriptor_ops++;
            // 预算含持久 stamp 和保留的 descriptor 容量，不只按活跃元素估算。
            bytes=split->stamps.size()*sizeof(a2o_tensor_stamp)+(split->batches.size()+1)*(
                sizeof(htp_buf_desc)*batch->n_bufs_max+sizeof(htp_tensor)*batch->n_tens_max+sizeof(htp_op_desc)*batch->n_ops_max);
            if(stats->descriptor_bytes>request->descriptor_budget || bytes>request->descriptor_budget-stats->descriptor_bytes)
                throw std::runtime_error("descriptor budget exceeded");
        }
        if(!batch->empty()) split->batches.push_back(std::move(batch));
        stats->descriptor_bytes+=bytes;stats->descriptor_batches+=split->batches.size();
        plan->splits.emplace(graph->uid,std::move(split));return 0;
    } catch(const std::exception &error) {
        snprintf(stats->reason,sizeof(stats->reason),"%s",error.what());a2o_forget_session(sess);return -6;
    }
}
// 返回0=无匹配模板，1=成功执行模板，-1=提交失败；此函数仅从实际 graph_compute 调用。
static int a2o_compute_prepared(ggml_backend_t backend,ggml_cgraph *graph) {
    auto *sess=static_cast<ggml_hexagon_session*>(backend->context);
    auto plan=a2o_session_plan(sess,false);if(!plan) return 0;
    auto it=plan->splits.find(graph->uid);if(it==plan->splits.end()) {plan->descriptor_misses++;return 0;}
    auto &split=*it->second;
    bool same=split.nodes.size()==(size_t)graph->n_nodes;
    for(size_t i=0;same && i<split.nodes.size();i++) same=split.nodes[i]==graph->nodes[i];
    // 只解引用当前图提供的活指针；缓存地址不相同时直接失配，避免触碰旧图。
    if(same) for(const auto &stamp:split.stamps) {
        if(stamp.node_index>=(uint32_t)graph->n_nodes) {same=false;break;}
        auto *current=graph->nodes[stamp.node_index];
        if(stamp.source>=0) current=current->src[stamp.source];
        if(current!=stamp.object || !a2o_same_stamp(stamp,current)) {same=false;break;}
    }
    if(!same || !sess->op_batch->empty()) {plan->descriptor_misses++;plan->splits.erase(it);return 0;}
    for(auto &batch:split.batches) {
        htp_opbatch_req req{};dspqueue_buffer dbuf{};
        if(!sess->op_queue->push(req,dbuf,batch.get(),false)) {
            sess->flush_pending(false);
            if(!sess->op_queue->push(req,dbuf,batch.get(),false)) return -1;
        }
        sess->op_pending++;
        if(dspqueue_write(sess->queue,0,1,&dbuf,sizeof(req),(uint8_t*)&req,DSPQUEUE_TIMEOUT)) return -1;
        plan->descriptor_hits+=batch->n_ops;
    }
    sess->flush_pending(true);return 1;
}
