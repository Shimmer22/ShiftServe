#include "state/async_preparation.h"
#include <atomic>
#include <cassert>
#include <future>
#include <stdexcept>

int main() {
    // Pending selection must return while the worker is deliberately held.
    std::atomic<int> writes{0};
    AsyncPreparation early(7, [&](const auto & valid) { if(!valid()) return false; ++writes; return true; }, true);
    assert(early.bind(7).reason == "pending");
    assert(writes == 0);
    early.release(); early.wait();
    assert(early.bind(7).backend == "htp");
    assert(early.bind(8).reason == "stale");
    early.join(); assert(writes == 1);
    early.cancel();
    assert(early.bind(7).backend == "cpu" && early.state() == "cancelled");

    AsyncPreparation cancelled(1, [&](const auto &) { ++writes; return true; }, true);
    cancelled.cancel(); cancelled.wait(); cancelled.join();
    assert(cancelled.bind(1).reason == "cancelled" && writes == 1);

    // Cancellation during a noninterruptible call is acknowledged at the next checkpoint.
    std::promise<void> entered, resume;
    auto gate = resume.get_future();
    AsyncPreparation running(2, [&](const auto & valid) {
        entered.set_value(); gate.wait();
        if(!valid()) return false;
        ++writes; return true;
    });
    entered.get_future().wait(); running.cancel(); resume.set_value(); running.join();
    assert(running.state() == "cancelled" && writes == 1);

    AsyncPreparation failed(3, [](const auto &) -> bool { throw std::runtime_error("import failed"); });
    bool wait_error=false, bind_error=false, join_error=false;
    try { failed.wait(); } catch(const std::runtime_error &) { wait_error=true; }
    try { failed.bind(3); } catch(const std::runtime_error &) { bind_error=true; }
    try { failed.join(); } catch(const std::runtime_error &) { join_error=true; }
    assert(wait_error && bind_error && join_error);
}
