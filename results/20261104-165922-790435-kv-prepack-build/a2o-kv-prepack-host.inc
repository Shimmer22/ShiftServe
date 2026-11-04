struct a2o_kv_descriptor {uint32_t fd,size,offset,kind,capacity,history;};
// 函数由 ggml_backend_reg_get_proc_address 显式取得；仅私有 bridge 提供。
static int a2o_kv_control(ggml_backend_reg_t reg,int history,int validate,uint64_t *stats) {
    auto hreg=static_cast<ggml_hexagon_registry*>(reg->context);
    auto sess=static_cast<ggml_hexagon_session*>(hreg->devices[0].context);
    sess->flush();htp_iface_pmu_conf conf={};
    if(stats) {
        ggml_hexagon_shared_buffer buf(sess,128);
        conf.events[0]=2;conf.events[1]=buf.fd;conf.events[2]=buf.size;
        int rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);
        if(!rc) {
            memcpy(stats,buf.base,6*sizeof(uint64_t));
            auto trace=(const uint64_t*)buf.base;
            if(opt_profile==1 && trace[6]) GGML_LOG_INFO("a2o-kv-layout: vtcm=%llu K_chunks=%llu V_chunks=%llu K_nc=%llu V_nc=%llu\n",
                (unsigned long long)trace[6],(unsigned long long)trace[7],(unsigned long long)trace[8],
                (unsigned long long)trace[9],(unsigned long long)trace[10]);
        }
        // history<0：保留计数后释放 tiles，CPU decode 不再持有额外 DSP DDR。
        if(!rc && history<0) {
            conf={};rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);
        }
        return rc;
    }
    int rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);
    if(rc || !history) return rc;
    std::vector<a2o_kv_descriptor> descriptors;
    for(auto t:sess->a2o_kv_tensors) {
        if(t->type!=GGML_TYPE_F16 || t->ne[0]!=256 || t->ne[2]!=1) return -1;
        auto buf=static_cast<ggml_hexagon_shared_buffer*>(t->buffer->context);
        descriptors.push_back({(uint32_t)buf->fd,(uint32_t)buf->size,
            (uint32_t)((uint8_t*)t->data-buf->base),(uint32_t)(strncmp(t->name,"cache_v_",8)==0),
            (uint32_t)t->ne[1],(uint32_t)history});
    }
    if(descriptors.size()!=72) return -2;
    ggml_hexagon_shared_buffer batch(sess,8+descriptors.size()*sizeof(a2o_kv_descriptor));
    auto header=(uint32_t*)batch.base;header[0]=descriptors.size();header[1]=validate;
    memcpy(header+2,descriptors.data(),descriptors.size()*sizeof(a2o_kv_descriptor));
    conf={};conf.events[0]=3;conf.events[1]=batch.fd;conf.events[2]=batch.size;
    return htp_iface_profiler(sess->handle,0x41324f50u,&conf);
}
