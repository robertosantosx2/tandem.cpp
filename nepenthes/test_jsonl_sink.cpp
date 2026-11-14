#include "nepenthes_jsonl_sink.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

int main() {
    const std::string path = "nepenthes-sink-test.jsonl";
    const std::string run_id = "run\"line\nbreak";
    std::remove(path.c_str());

    {
        nepenthes::JsonlSink sink(path, run_id);

        for (std::uint64_t i = 0; i < 256; ++i) {
            nepenthes::Event event{};
            event.kind = nepenthes::EventKind::kernel;
            event.object_id = i + 1;
            event.bytes = 4096;
            event.layer = 3;
            nepenthes::Trace::emit(event);
        }

        // Exercise the multi-producer queue while keeping the total below capacity.
        std::vector<std::thread> producers;
        for (std::uint64_t t = 0; t < 4; ++t) {
            producers.emplace_back([t] {
                for (std::uint64_t i = 0; i < 1000; ++i) {
                    nepenthes::Event event{};
                    event.kind = nepenthes::EventKind::transfer;
                    event.request_id = t + 1;
                    event.object_id = i + 1;
                    event.bytes = 1024;
                    nepenthes::Trace::emit(event);
                }
            });
        }
        for (auto& producer : producers) producer.join();

        sink.close();
        sink.close(); // close is idempotent
        assert(sink.dropped() == 0);
        assert(sink.write_errors() == 0);
        assert(!nepenthes::Trace::enabled());
    }

    std::ifstream input(path);
    assert(input.good());
    std::string line;
    std::size_t count = 0;
    while (std::getline(input, line)) {
        assert(line.find("\"source\":\"tandem.cpp\"") != std::string::npos);
        assert(line.find("\"run_id\":\"run\\\"line\\nbreak\"") != std::string::npos);
        if (count < 256) {
            assert(line.find("\"event_kind\":\"kernel\"") != std::string::npos);
        }
        ++count;
    }
    input.close();
    std::remove(path.c_str());
    assert(count == 4256);
    return 0;
}
