#include "../src/a2o/native/session/kv_window.h"
#include <cassert>

int main() {
    const SessionKvWindow cold(true, false, 0, 1200, 20, 4, 32);
    assert(!cold.can_reduce && cold.recent == 1200 && cold.keep_floor == 1200);

    const SessionKvWindow long_observation(true, true, 2490, 158, 2145, 4, 32);
    assert(long_observation.can_reduce && long_observation.recent == 158);
    assert(long_observation.keep_floor == 2303);

    const SessionKvWindow short_observation(true, true, 40, 12, 4, 4, 32);
    assert(short_observation.recent == 32 && short_observation.keep_floor == 36);

    const SessionKvWindow short_history(true, true, 2, 120, 2, 4, 32);
    assert(short_history.keep_floor == 122);
    const SessionKvWindow disabled(false, true, 2490, 158, 2145, 4, 32);
    assert(!disabled.can_reduce);
}
