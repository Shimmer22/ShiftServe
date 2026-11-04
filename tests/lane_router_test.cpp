#include "preparation_lane.h"
#include "router.h"
#include <cassert>
#include <atomic>
#include <future>

int main(int argc, char ** argv) {
    PreparationLane lane;
    auto barrier=std::make_shared<PreparationBarrier>();
    auto payload=std::make_shared<int>(42); std::weak_ptr<int> lifetime=payload;
    std::atomic<int> writes{0};
    assert(lane.start(1,[barrier,payload,&writes](const auto & valid) {
        barrier->enter(); if(!valid()) return false; ++writes; return *payload==42;
    }));
    payload.reset(); barrier->wait_entered();
    lane.retire();
    assert(lane.availability(2)=="busy" && !lifetime.expired());
    for(int round=2;round<100;++round) {
        // CPU 前台在屏障释放之前推进；不创建第二个 worker/context。
        assert(!lane.start(round,[](const auto &) {return true;}));
        assert(lane.started()==1 && writes==0);
    }
    barrier->release(); lane.close();
    assert(lifetime.expired() && writes==0 && lane.reaped()==1);
    assert(lane.start(100,[](const auto &) {return true;}));
    lane.wait_for_test();
    assert(lane.availability(99)=="stale");
    assert(lane.availability(100)=="ready"); lane.claim(100);
    assert(!lane.occupied() && lane.reaped()==2);

    // 关闭时可阻塞，但错误必须传播；析构仅做最后的安全清理。
    std::promise<void> error_entered;
    assert(lane.start(101,[&](const auto &) -> bool {
        error_entered.set_value(); throw std::runtime_error("import error");
    }));
    error_entered.get_future().wait();
    bool error=false;
    try {lane.close();} catch(const std::runtime_error &) {error=true;}
    assert(error);

    CostRouter router;
    router.rows={{1024,16,4,3,"CC",100,0},{1024,16,4,3,"HC",200,50},
                 {1024,64,4,3,"CC",300,0},{1024,64,4,3,"HC",200,50}};
    assert(router.choose(1024,16,4,"ready").reason=="cpu_lower_cost");
    assert(router.choose(1024,64,4,"ready").backend=="htp");
    for(auto unavailable:{"pending","busy","stale","cancelled","absent"})
        assert(router.choose(1024,64,4,unavailable).backend=="cpu");
    assert(router.choose(1024,17,4,"ready").reason=="coverage_missing");
    assert(!router.choose(1024,17,4,"pending").cost_covered);
    assert(router.choose(1024,17,4,"absent").reason=="coverage_missing");
    assert(router.choose(1024,64,5,"ready").reason=="coverage_missing");
    assert(router.should_prepare(1024,4) && !router.should_prepare(2048,4));
    if(argc==2) {
        auto measured=CostRouter::load(argv[1]);
        assert(measured.choose(4096,16,4,"ready").reason=="cpu_lower_cost");
        assert(measured.choose(4096,64,4,"ready").reason=="htp_lower_cost");
        assert(measured.choose(1024,16,4,"ready").reason=="coverage_missing");
        std::string fixture=std::string(argv[1])+".fixture";
        { std::ofstream out(fixture); out << "a2o-router-v1\tmechanism-fixture\t5120\t128\t1\t1\n1024\t16\t4\tCC\t100\t0\t0\n"; }
        bool rejected=false;
        try {CostRouter::load(fixture);} catch(const std::runtime_error &) {rejected=true;}
        assert(rejected && CostRouter::load(fixture,true).rows.size()==1);
    }
}
