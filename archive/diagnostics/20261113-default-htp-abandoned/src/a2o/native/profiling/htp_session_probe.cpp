// 只枚举已请求的CPU/HTP后端及有效会话，不加载模型、不执行生成。
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
int main() {
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name("HTP");
    if (!reg) { std::puts("HTP registry unavailable"); return 2; }
    int requested = std::getenv("GGML_HEXAGON_NDEV") ? std::atoi(std::getenv("GGML_HEXAGON_NDEV")) : 1;
    size_t count = ggml_backend_reg_dev_count(reg);
    std::printf("requested=%d registry_count=%zu\n", requested, count);
    bool passed = count == size_t(requested);
    for (size_t i=0; i<count; ++i) {
        auto dev = ggml_backend_reg_dev_get(reg,i);
        if (!dev) { std::printf("session[%zu]=unavailable\n",i); passed=false; }
        else std::printf("session[%zu]=%s\n",i,ggml_backend_dev_name(dev));
    }
    std::fflush(stdout);
    return passed ? 0 : 3;
}
