#include "nepenthes_jsonl_sink.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

int main() {
    const std::string path = "nepenthes-jsonl-test.jsonl";
    std::remove(path.c_str());
    {
        nepenthes::JsonlSink sink(path, "run\"line\nbreak");
        for (std::uint64_t i = 0; i < 256; ++i) {
            nepenthes::emit(nepenthes::EventType::kernel, "cuda.kernel", i, 4096);
        }
        std::vector<std::thread> producers;
        for (std::uint64_t t = 0; t < 4; ++t) {
            producers.emplace_back([t] {
                for (std::uint64_t i = 0; i < 1000; ++i) {
                    nepenthes::emit(nepenthes::EventType::transfer, "moe.transfer", t, 1024,
                                    nepenthes::phase_instant, t + 1);
                }
            });
        }
        for (auto & producer : producers) producer.join();
        sink.close();
        sink.close();
        assert(sink.dropped() == 0);
        assert(sink.write_errors() == 0);
    }

    std::ifstream input(path);
    assert(input.good());
    std::string line;
    std::size_t count = 0;
    while (std::getline(input, line)) {
        // Every record must occupy one physical line and end with its JSON brace.
        // This catches a literal backslash+n being written instead of a newline.
        assert(!line.empty());
        assert(line.back() == '}');
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
