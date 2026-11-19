#include "nepenthes_jsonl_sink.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
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
    std::string summary_line;
    std::size_t count = 0;
    while (std::getline(input, line)) {
        assert(!line.empty());
        assert(line.back() == '}');
        assert(line.find("\"source\":\"tandem.cpp\"") != std::string::npos);
        assert(line.find("\"run_id\":\"run\\\"line\\nbreak\"") != std::string::npos);
        if (line.find("\"event_kind\":\"sink_summary\"") != std::string::npos) {
            assert(summary_line.empty());
            assert(count == 4256);
            summary_line = line;
        } else {
            assert(summary_line.empty()); // the summary must be final
            if (count < 256) {
                assert(line.find("\"event_kind\":\"kernel\"") != std::string::npos);
            }
        }
        ++count;
    }
    input.close();
    assert(count == 4257); // 4,256 events plus exactly one final summary
    assert(!summary_line.empty());
    assert(summary_line.find("\"events_drained\":4256") != std::string::npos);
    assert(summary_line.find("\"dropped_events\":0") != std::string::npos);
    assert(summary_line.find("\"write_errors\":0") != std::string::npos);
    if (std::getenv("NEPENTHES_KEEP_TEST_TRACE") == nullptr) {
        std::remove(path.c_str());
    }
    return 0;
}
