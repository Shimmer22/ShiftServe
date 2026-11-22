#pragma once
#include <algorithm>

struct SessionKvWindow {
    bool can_reduce;
    int recent;
    int keep_floor;

    SessionKvWindow(bool enabled, bool reuse, int h, int p, int pinned, int sink, int recent_limit)
        : can_reduce(enabled && reuse),
          recent(std::max(p, recent_limit)),
          keep_floor(std::min(h + p, std::max(pinned, sink) + recent)) {}
};
