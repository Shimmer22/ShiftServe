// 只读时钟诊断：核对 steady clock 是否包含设备实际经过的时间。
#include <chrono>
#include <cstdio>
#include <time.h>
#include <unistd.h>
static double seconds(clockid_t id) { timespec t{}; if(clock_gettime(id,&t))return -1;return t.tv_sec+1e-9*t.tv_nsec; }
int main() {
    for(int i=0;i<2;++i) {
        std::printf("{\"steady_s\":%.9f,\"monotonic_s\":%.9f,\"boottime_s\":%.9f,\"realtime_s\":%.9f}\n",
            std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(),
            seconds(CLOCK_MONOTONIC),seconds(CLOCK_BOOTTIME),seconds(CLOCK_REALTIME));std::fflush(stdout);
        if(!i) {timespec wait{5,0};while(nanosleep(&wait,&wait)){};}
    }
}
