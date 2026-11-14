#include "nepenthes_trace.hpp"

#include <cassert>
#include <cstdint>

static std::uint64_t seen = 0;
static std::uint64_t last_duration = 0;
static std::uint16_t last_flags = 0;

static void collect(const nepenthes::Event & event) noexcept {
    ++seen;
    last_duration = event.duration_ns;
    last_flags = event.flags;
}

int main() {
    using namespace nepenthes;

    emit(EventType::request_start, "disabled");
    assert(seen == 0);

    set_sink(&collect);
    emit(EventType::cache_hit, "expert", 7, 4096, flag_device, 42);
    assert(seen == 1);
    assert(last_flags == flag_device);

    {
        ScopedEvent phase(EventType::sync, "worker_wait", 3, 0, 42);
    }
    assert(seen == 3);
    assert(last_flags == phase_end);
    (void) last_duration;

    set_sink(nullptr);
    emit(EventType::request_end, "disabled");
    assert(seen == 3);
}
