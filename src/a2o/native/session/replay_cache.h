#pragma once
#include <filesystem>
#include <iomanip>

// One hot snapshot in RAM; optional persistent snapshots in a private directory.
// Hashes select files only: exact token/config equality is checked on every hit.
class FrozenReplayCache {
    std::vector<llama_token> ids_;
    SessionSnapshot snapshot_;
    std::string identity_;
    static std::string checksum(const std::vector<uint8_t> &bytes) {
        uint64_t hash=14695981039346656037ULL;
        for(auto byte:bytes){hash^=byte;hash*=1099511628211ULL;}
        std::ostringstream out;out<<std::hex<<hash;return out.str();
    }
public:
    explicit FrozenReplayCache(std::string identity):identity_(std::move(identity)) {}
    bool matches(const std::vector<llama_token> &ids) const {return ids==ids_ && !snapshot_.state.empty();}
    const SessionSnapshot &get(llama_context *ctx,const std::vector<llama_token> &ids,
                               const std::string &directory,J &report,KvVerification verification) {
        double begin=now_ms();
        report={{"hit",false},{"source","bootstrap"},{"prefix_tokens",ids.size()}};
        if(ids==ids_ && !snapshot_.state.empty()) {
            report["hit"]=true;report["source"]="memory";
            report["setup_ms"]=now_ms()-begin;return snapshot_;
        }
        // Include executable build, model identity and context in persistent key.
        J metadata={{"format",2},{"identity",identity_},{"context",context_size},
                    {"batch",batch_size},{"prefix_ids",ids}};
        auto serialized=metadata.dump();uint64_t hash=14695981039346656037ULL;
        for(unsigned char c:serialized){hash^=c;hash*=1099511628211ULL;}
        std::ostringstream key;key<<std::hex<<hash;
        std::filesystem::path file;
        if(!directory.empty()) {
            std::filesystem::create_directories(directory);
            file=std::filesystem::path(directory)/(key.str()+".json");
            if(std::filesystem::exists(file)) {
                auto stored=J::parse(read(file.string()));
                require(stored.at("metadata")==metadata,"snapshot metadata mismatch");
                const auto statefile=file.string()+".state";
                const size_t bytes=stored.at("state_bytes").get<size_t>();
                require(std::filesystem::file_size(statefile)==bytes,"snapshot size mismatch");
                SessionSnapshot s;s.state.resize(bytes);s.next_pos=ids.size();
                std::ifstream in(statefile,std::ios::binary);
                require(bool(in.read(reinterpret_cast<char *>(s.state.data()),s.state.size()))
                        && in.peek()==std::ifstream::traits_type::eof(),"snapshot read failed");
                s.evaluated_tokens=ids.size();
                require(checksum(s.state)==stored.at("state_checksum").get<std::string>(),"snapshot checksum mismatch");
                restore_session(ctx,s,nullptr,verification);
                ids_=ids;snapshot_=std::move(s);
                report["hit"]=true;report["source"]="disk";
                report["setup_ms"]=now_ms()-begin;return snapshot_;
            }
        }
        // When sorted by trajectory, extend an exact prefix instead of starting
        // from token zero. Recurrent state cannot be cropped on a mismatch.
        size_t reused=0;
        if(!ids_.empty() && ids.size()>=ids_.size() &&
           std::equal(ids_.begin(),ids_.end(),ids.begin())) {
            restore_session(ctx,snapshot_,nullptr,verification);reused=ids_.size();
        } else llama_memory_clear(llama_get_memory(ctx),true);
        llama_pos pos=reused;
        evaluate(ctx,std::vector<llama_token>(ids.begin()+reused,ids.end()),pos);
        SessionSnapshot s;s.state=export_seq(ctx);s.next_pos=ids.size();s.evaluated_tokens=ids.size();
        ids_=ids;snapshot_=std::move(s);
        if(!directory.empty()) {
            auto statefile=file.string()+".state",tmp=statefile+".tmp";
            {std::ofstream f(tmp,std::ios::binary);f.write(reinterpret_cast<const char*>(snapshot_.state.data()),snapshot_.state.size());require(bool(f),"snapshot write failed");}
            std::filesystem::rename(tmp,statefile);
            {std::ofstream f(file.string()+".tmp");f<<J{{"metadata",metadata},{"state_bytes",snapshot_.state.size()},{"state_checksum",checksum(snapshot_.state)}}.dump();require(bool(f),"snapshot metadata write failed");}
            std::filesystem::rename(file.string()+".tmp",file);
        }
        report["extended_prefix_tokens"]=reused;
        report["state_bytes"]=snapshot_.state.size();
        report["setup_ms"]=now_ms()-begin;return snapshot_;
    }
};
