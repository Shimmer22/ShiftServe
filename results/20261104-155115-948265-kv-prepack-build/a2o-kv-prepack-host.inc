// 函数由 ggml_backend_reg_get_proc_address 显式取得；仅私有 bridge 提供。
static int a2o_kv_control(ggml_backend_reg_t reg,int history,int validate,uint64_t *stats) {
    auto hreg=static_cast<ggml_hexagon_registry*>(reg->context);
    auto sess=static_cast<ggml_hexagon_session*>(hreg->devices[0].context);
    sess->flush();htp_iface_pmu_conf conf={};
    if(stats) {
        ggml_hexagon_shared_buffer buf(sess,128);
        conf.events[0]=2;conf.events[1]=buf.fd;conf.events[2]=buf.size;
        int rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);
        if(!rc) memcpy(stats,buf.base,6*sizeof(uint64_t));
        // history<0：保留计数后释放 tiles，CPU decode 不再持有额外 DSP DDR。
        if(!rc && history<0) {
            conf={};rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);
        }
        return rc;
    }
    int rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);
    if(rc || !history) return rc;
    int count=0;
    for(auto t:sess->a2o_kv_tensors) {
        if(t->type!=GGML_TYPE_F16 || t->ne[0]!=256 || t->ne[2]!=1) return -1;
        auto buf=static_cast<ggml_hexagon_shared_buffer*>(t->buffer->context);
        conf={};conf.events[0]=1;conf.events[1]=buf->fd;conf.events[2]=buf->size;
        conf.events[3]=(uint8_t*)t->data-buf->base;conf.events[4]=strncmp(t->name,"cache_v_",8)==0;
        conf.events[5]=t->ne[1];conf.events[6]=history;conf.events[7]=validate;
        rc=htp_iface_profiler(sess->handle,0x41324f50u,&conf);if(rc) return rc;count++;
    }
    return count==72?0:-2;
}
