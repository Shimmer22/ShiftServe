#include "scheduling/router.h"
#include <cassert>

static OnlineCostKey key() {return {"offline-source","ordinary","HC",256,32,4};}
static OnlineCostConfig config() {OnlineCostConfig c;c.enabled=true;return c;}

static void routing() {
    CostRouter router;router.source="offline-source";
    router.rows={{256,32,4,3,"CC",48,0},{256,32,4,3,"HC",38,40}};
    OnlineRouteCosts online(config());
    auto decision=router.choose(256,32,4,"ready",&online,0);
    assert(decision.backend=="htp" && decision.htp_source=="offline");
    for(int n=1;n<=3;n++) {
        assert(online.sample(key(),n));
        online.update(key(),70,n);
        decision=router.choose(256,32,4,"ready",&online,n);
        assert(decision.backend==(n<3?"htp":"cpu"));
    }
    assert(decision.cpu_ms==48 && decision.cpu_source=="offline");
    assert(decision.htp_ms==70 && decision.htp_source=="online_corrected");
    assert(router.rows[1].future_ms==38 && online.samples({"offline-source","ordinary","CC",256,32,4},3)==0);
    assert(router.should_prepare(256,4) && !router.should_prepare(256,4,&online,3));
    assert(router.choose(256,32,4,"pending",&online,3).reason=="candidate_pending");
    auto unseen=key();unseen.p=33;
    for(int n=0;n<3;n++) online.update(unseen,1,4);
    assert(!router.choose(256,33,4,"ready",&online,4).cost_covered);
}

static void horizon_and_sampling() {
    int reads=0;auto clock=[&] {++reads;return 109.;};
    DecodeHorizon horizon(true,4,100);
    for(int count=1;count<=8;++count) horizon.evaluated(count,clock);
    assert(reads==1 && horizon.prefix_ms()==9); // 完整输出更长也只取前 D。
    DecodeHorizon early_eog(true,4,100),disabled(false,4,100);
    for(int count=1;count<=2;++count) early_eog.evaluated(count,clock);
    for(int count=1;count<=8;++count) disabled.evaluated(count,clock);
    assert(reads==1 && !std::isfinite(early_eog.prefix_ms()) && !std::isfinite(disabled.prefix_ms()));
    OnlineRouteCosts online(config());
    assert(online.sample(key(),0));
    assert(!online.update(key(),early_eog.prefix_ms(),0) && online.samples(key(),0)==0);
    for(int request=1;request<=11;++request) {
        bool sampled=online.sample(key(),request);
        assert(sampled==(request<=3 || request==11));
        if(sampled) online.update(key(),request<=3?10:20,request);
    }
    assert(online.predict(key(),11)==12); // EMA alpha=.2；不是整体输出耗时。
}

static void state_bounds() {
    auto cfg=config();cfg.min_samples=1;cfg.capacity=2;cfg.ttl_ms=10;
    OnlineRouteCosts online(cfg);
    auto a=key(),b=key(),c=key();b.p=64;c.p=128;
    online.update(a,10,0);online.update(b,20,1);
    assert(!std::isfinite(online.predict(a,11)) && online.sample(a,11));
    online.update(a,30,11);online.update(c,40,12);
    assert(online.size()==2 && !std::isfinite(online.predict(b,12)) && online.predict(a,12)==30);
    assert(!online.update(a,-1,12) && !online.update(a,INFINITY,12));
    for(int dimension=0;dimension<7;++dimension) {
        auto other=a;
        if(dimension==0) other.source="other-model-runtime-table";
        if(dimension==1) other.regime="plan_hit";
        if(dimension==2) other.regime="plan_miss";
        if(dimension==3) other.path="CC";
        if(dimension==4) ++other.h;
        if(dimension==5) ++other.p;
        if(dimension==6) ++other.d;
        assert(!std::isfinite(online.predict(other,12)));
    }
    OnlineRouteCosts disabled;
    assert(!disabled.sample(a,0) && !disabled.update(a,1,0) && disabled.size()==0);
}

int main(int argc,char **argv) {
    assert(argc==2);std::string name=argv[1];
    if(name=="routing") routing();
    else if(name=="horizon") horizon_and_sampling();
    else if(name=="state") state_bounds();
    else return 2;
}
