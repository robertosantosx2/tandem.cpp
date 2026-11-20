#include "nepenthes_nvtx.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<std::string> marker_messages;
static std::vector<nepenthes::Event> marker_events;

extern "C" void nvtxMarkA(const char * message) {
    marker_messages.emplace_back(message ? message : "");
}

static void collect(const nepenthes::Event & event) noexcept {
    marker_events.push_back(event);
}

int main() {
    char message[96]{};
    assert(nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "start", 42));
    assert(std::strcmp(message, "nepenthes.batch.start token=42") == 0);
    assert(nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "end", 42));
    assert(std::strcmp(message, "nepenthes.batch.end token=42") == 0);
    assert(!nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "other", 42));
    assert(!nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "start", 0));
    assert(!nepenthes::nvtx_detail::format_batch_message(message, 8, "start", 42));

    const char * setting = std::getenv("NEPENTHES_NVTX");
    const bool enabled = setting != nullptr && std::strcmp(setting, "1") == 0;
    nepenthes::set_sink(&collect);
    {
        nepenthes::BatchNvtxScope scope(42);
    }
    nepenthes::BatchNvtxScope invalid(0);
    nepenthes::set_sink(nullptr);

    if (enabled) {
        assert(marker_messages.size() == 2);
        assert(marker_messages[0] == "nepenthes.batch.start token=42");
        assert(marker_messages[1] == "nepenthes.batch.end token=42");
        assert(marker_events.size() == 2);
        assert(marker_events[0].type == nepenthes::EventType::nvtx_marker);
        assert(marker_events[0].correlation_token == 42);
        assert(std::strcmp(marker_events[0].label, "batch.start") == 0);
        assert(marker_events[0].value1 >= marker_events[0].value0);
        assert(marker_events[1].type == nepenthes::EventType::nvtx_marker);
        assert(marker_events[1].correlation_token == 42);
        assert(std::strcmp(marker_events[1].label, "batch.end") == 0);
        assert(marker_events[1].value1 >= marker_events[1].value0);
    } else {
        assert(marker_messages.empty());
        assert(marker_events.empty());
    }
    return 0;
}
