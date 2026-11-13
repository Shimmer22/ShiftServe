// 仅注入私有ggml-backend.cpp，不更改算子、设备分配或同步行为。
#include <map>
#include <string>
#include <sstream>
static thread_local bool a2o_attr_active=false;
static thread_local int a2o_attr_copy_depth=0,a2o_attr_compute_depth=0;
struct a2o_attr_value {int64_t us=0,calls=0,bytes=0,nodes=0;std::string metadata;};
static thread_local std::map<std::string,a2o_attr_value> a2o_attr_values;
struct a2o_attr_scope {
    std::string key;int64_t start=0;bool active=false;
    a2o_attr_scope(std::string k,bool enabled=true,int64_t bytes=0,int64_t nodes=0):key(std::move(k)),active(a2o_attr_active&&enabled){
        if(active){start=ggml_time_us();auto &v=a2o_attr_values[key];v.calls++;v.bytes+=bytes;v.nodes+=nodes;}
    }
    ~a2o_attr_scope(){if(active)a2o_attr_values[key].us+=ggml_time_us()-start;}
};
struct a2o_attr_compute {
    a2o_attr_scope scope;
    static std::string label(ggml_backend_t backend,ggml_cgraph *graph){
        if(!a2o_attr_active)return "";std::string name=ggml_backend_name(backend);std::map<std::string,int> ops;
        for(int i=0;i<graph->n_nodes;++i)if(!ggml_op_is_empty(graph->nodes[i]->op))ops[ggml_op_name(graph->nodes[i]->op)]++;
        std::string label="COMPUTE/"+name+"/";for(auto &p:ops){if(label.back()!='/')label+=",";label+=p.first;}
        return label;
    }
    a2o_attr_compute(ggml_backend_t b,ggml_cgraph*g):scope(label(b,g),true,0,g->n_nodes){a2o_attr_compute_depth++;}
    ~a2o_attr_compute(){a2o_attr_compute_depth--;}
};
struct a2o_attr_copy {
    a2o_attr_scope scope;
    a2o_attr_copy(size_t bytes):scope("COPY",a2o_attr_copy_depth==0&&a2o_attr_compute_depth==0,bytes){a2o_attr_copy_depth++;}
    ~a2o_attr_copy(){a2o_attr_copy_depth--;}
};
extern "C" __attribute__((visibility("default"))) void a2o_attr_begin(int enabled){a2o_attr_values.clear();a2o_attr_active=enabled;}
static std::string a2o_json_quote(const std::string &v){std::string s="\"";for(char c:v){if(c=='\\'||c=='\"')s+='\\';if(c=='\n'){s+="\\n";continue;}s+=c;}return s+'\"';}
static std::string a2o_tensor_meta(const ggml_tensor *t){
    std::ostringstream s;s<<"{\"name\":"<<a2o_json_quote(t->name)<<",\"type\":"<<a2o_json_quote(ggml_type_name(t->type))<<",\"ne\":[";
    for(int i=0;i<4;++i){if(i)s<<',';s<<t->ne[i];}
    s<<"],\"logical_bytes\":"<<ggml_nbytes(t)<<"}";return s.str();
}
extern "C" __attribute__((visibility("default"))) int a2o_cpu_node_enabled(){return a2o_attr_active;}
extern "C" __attribute__((visibility("default"))) void a2o_cpu_node_record(const ggml_tensor *node,int64_t us,int fused){
    if(!a2o_attr_active)return;
    std::string key=std::string("CPU_NODE/")+ggml_op_name(node->op)+"/"+node->name;
    auto &v=a2o_attr_values[key];v.us+=us;v.calls++;v.nodes+=1+fused;
    if(v.metadata.empty()){
        std::ostringstream s;s<<"{\"output\":"<<a2o_tensor_meta(node)<<",\"fused_following_nodes\":"<<fused<<",\"sources\":[";bool first=true;
        for(auto *src:node->src)if(src){if(!first)s<<',';first=false;s<<a2o_tensor_meta(src);}
        s<<"]}";v.metadata=s.str();
    }
}
extern "C" __attribute__((visibility("default"))) const char *a2o_attr_end(){
    a2o_attr_active=false;static thread_local std::string output;std::ostringstream s;s<<"{";bool first=true;
    for(auto &p:a2o_attr_values){if(!first)s<<",";first=false;s<<a2o_json_quote(p.first)<<":{\"us\":"<<p.second.us<<",\"calls\":"<<p.second.calls<<",\"bytes\":"<<p.second.bytes<<",\"nodes\":"<<p.second.nodes;if(!p.second.metadata.empty())s<<",\"metadata\":"<<p.second.metadata;s<<"}";}
    s<<"}";output=s.str();return output.c_str();
}
