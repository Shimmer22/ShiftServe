#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
using J=nlohmann::ordered_json;
using llama_token=int;
using llama_pos=int;
struct llama_context {std::vector<uint8_t> state;};
struct SessionSnapshot {std::vector<uint8_t> state; int next_pos=0,evaluated_tokens=0;};
int context_size=8192,batch_size=128,evaluated=0;
double now_ms(){return 0;}
void require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
std::string read(const std::string &file){std::ifstream in(file,std::ios::binary);return {std::istreambuf_iterator<char>(in),{}};}
llama_context *llama_get_memory(llama_context *ctx){return ctx;}
void llama_memory_clear(llama_context *ctx,bool){ctx->state.clear();}
void restore_session(llama_context *ctx,const SessionSnapshot &snapshot){ctx->state=snapshot.state;}
void evaluate(llama_context *ctx,const std::vector<int> &ids,int &pos){
    for(int id:ids){ctx->state.push_back(id);++pos;++evaluated;}
}
std::vector<uint8_t> export_seq(llama_context *ctx){return ctx->state;}
#include "session/replay_cache.h"
int main(int argc,char **argv){
    assert(argc==2);std::string directory=argv[1];llama_context ctx;J report;
    FrozenReplayCache cache("model-build-a");
    auto a=cache.get(&ctx,{1,2,3},directory,report);
    assert(evaluated==3 && report["source"]=="bootstrap");
    ctx.state={99};
    auto b=cache.get(&ctx,{1,2,3},directory,report);
    assert(evaluated==3 && report["source"]=="memory" && a.state==b.state);
    // The caller restores fixtures before measurement; a hit returns saved bytes.
    auto c=cache.get(&ctx,{1,2,3,4},directory,report);
    assert(evaluated==4 && report["extended_prefix_tokens"]==3);
    assert(c.state==std::vector<uint8_t>({1,2,3,4}));
    cache.get(&ctx,{7,8},directory,report);
    assert(evaluated==6 && report["extended_prefix_tokens"]==0);
    FrozenReplayCache restarted("model-build-a");
    auto d=restarted.get(&ctx,{1,2,3},directory,report);
    assert(evaluated==6 && report["source"]=="disk" && d.state==a.state);
    FrozenReplayCache different_build("model-build-b");
    different_build.get(&ctx,{1,2,3},directory,report);
    assert(evaluated==9 && !report["hit"].get<bool>());
    // Same-length corruption must fail before importing state.
    for(const auto &entry:std::filesystem::directory_iterator(directory)){
        if(entry.path().extension()!=".json")continue;
        auto metadata=J::parse(read(entry.path().string()));
        if(metadata["metadata"]["identity"]!="model-build-a")continue;
        std::fstream file(entry.path().string()+".state",std::ios::in|std::ios::out|std::ios::binary);
        file.put('x');
    }
    FrozenReplayCache corrupted("model-build-a");bool rejected=false;
    try{corrupted.get(&ctx,{1,2,3},directory,report);}catch(const std::runtime_error &){rejected=true;}
    assert(rejected);
}
