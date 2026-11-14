#include "nepenthes_jsonl_sink.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>

int main() {
    const std::string path = "nepenthes-sink-test.jsonl";
    std::remove(path.c_str());
    {
        nepenthes::JsonlSink sink(path);
        for (std::uint64_t i = 0; i < 256; ++i) {
            nepenthes::Event event{};
            event.kind = nepenthes::EventKind::kernel;
            event.object_id = i + 1;
            event.bytes = 4096;
            event.layer = 3;
            nepenthes::Trace::emit(event);
        }
        sink.close();
        assert(sink.dropped() == 0);
    }
    std::ifstream input(path);
    assert(input.good());
    std::string line;
    std::size_t count = 0;
    while (std::getline(input, line)) {
        assert(line.find("\"event_kind\":\"kernel\"") != std::string::npos);
        assert(line.find("\"source\":\"tandem.cpp\"") != std::string::npos);
        ++count;
    }
    std::remove(path.c_str());
    assert(count == 256);
    return 0;
}
