#include "nepenthes_trace.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <thread>

static std::uint64_t seen = 0;
static std::uint64_t last_duration = 0;
static std::uint16_t last_flags = 0;
static std::atomic<bool> callback_entered{false};
static std::atomic<bool> release_callback{false};

static void collect(const nepenthes::Event & event) noexcept {
    ++seen;
    last_duration = event.duration_ns;
    last_flags = event.flags;
}

static void blocking_collect(const nepenthes::Event &) noexcept {
    callback_entered.store(true, std::memory_order_release);
    while (!release_callback.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

int main() {
    using namespace nepenthes;

    assert(request_id_from_task_id(-1) == 0);
    assert(request_id_from_task_id(0) == 1);
    assert(request_id_from_task_id(41) == 42);

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

    // Shutdown must wait for a callback that has already entered the sink.
    set_sink(&blocking_collect);
    std::thread producer([] {
        emit(EventType::kernel, "shutdown-race");
    });
    while (!callback_entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    std::atomic<bool> disable_done{false};
    std::thread disabler([&] {
        set_sink(nullptr);
        disable_done.store(true, std::memory_order_release);
    });
    while (sink_slot().load(std::memory_order_acquire) != nullptr) {
        std::this_thread::yield();
    }
    assert(!disable_done.load(std::memory_order_acquire));

    release_callback.store(true, std::memory_order_release);
    producer.join();
    disabler.join();
    assert(disable_done.load(std::memory_order_acquire));
    assert(active_emitters().load(std::memory_order_acquire) == 0);
}
