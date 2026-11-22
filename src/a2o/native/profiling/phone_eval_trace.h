#pragma once
#include <array>
#include <filesystem>
#include <fstream>
#include <sched.h>

struct PhoneGraphTrace {
    using Begin = void (*)();
    using End = void (*)(int64_t *);
    bool enabled;
    std::array<int64_t,6> totals{};

    explicit PhoneGraphTrace(bool enabled) : enabled(enabled) {}
    template<class Evaluate>
    void measure(Evaluate evaluate) {
        if(!enabled){evaluate();return;}
        static const auto begin=reinterpret_cast<Begin>(dlsym(RTLD_DEFAULT,"llama_a2o_graph_profile_begin"));
        static const auto end=reinterpret_cast<End>(dlsym(RTLD_DEFAULT,"llama_a2o_graph_profile_end"));
        require(begin && end,"runtime does not support graph profiling");
        std::array<int64_t,6> counts{};
        begin();
        try { evaluate(); }
        catch(...) { end(counts.data()); throw; }
        end(counts.data());
        for(size_t i=0;i<totals.size();++i)totals[i]+=counts[i];
    }
    J json() const {
        return {{"build_us",totals[0]},{"dispatch_us",totals[1]},{"sync_us",totals[2]},
                {"builds",totals[3]},{"graphs",totals[4]},{"callback_graphs",totals[5]}};
    }
};

static J phone_clock_sample() {
    namespace fs=std::filesystem;
    const auto integer=[](const fs::path & file)->J {
        long value;
        std::ifstream input(file);
        if(input>>value)return value;
        return nullptr;
    };
    J cpu=J::object(), thermal=J::object(), accelerator=J::object(), affinity=J::array();
    for(int policy:{0,2,5,7}){
        const auto name="policy"+std::to_string(policy);
        cpu[name]=integer(fs::path("/sys/devices/system/cpu/cpufreq")/name/"scaling_cur_freq");
    }
    std::error_code ec;
    for(fs::directory_iterator it("/sys/class/thermal",ec),end; it!=end && !ec; it.increment(ec)){
        const auto zone=it->path();
        if(zone.filename().string().compare(0,12,"thermal_zone")!=0)continue;
        std::ifstream type_file(zone/"type");
        std::string type;
        if(std::getline(type_file,type)){
            const auto value=integer(zone/"temp");
            if(!value.is_null())thermal[zone.filename().string()]={{"type",type},{"value",value}};
        }
    }
    ec.clear();
    for(fs::directory_iterator it("/sys/class/devfreq",ec),end; it!=end && !ec; it.increment(ec)){
        std::string name=it->path().filename().string();
        std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return std::tolower(c);});
        for(const char * component:{"cdsp","adsp","nsp","htp","npu"}){
            if(name.find(component)!=std::string::npos){accelerator[name]=integer(it->path()/"cur_freq");break;}
        }
    }
    cpu_set_t mask;
    if(sched_getaffinity(0,sizeof(mask),&mask)==0){
        for(int cpu_id=0;cpu_id<CPU_SETSIZE;++cpu_id)if(CPU_ISSET(cpu_id,&mask))affinity.push_back(cpu_id);
    }
    return {{"cpu_policy_khz",cpu},{"thermal_sysfs_raw",thermal},{"accelerator_devfreq_hz",accelerator},
            {"devfreq_access_error",ec ? J(ec.message()) : J(nullptr)},{"request_thread_affinity",affinity}};
}
