#include "nepenthes_nvtx.hpp"

#include <cassert>
#include <cstring>

int main() {
    char message[96]{};
    assert(nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "start", 42));
    assert(std::strcmp(message, "nepenthes.batch.start token=42") == 0);
    assert(nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "end", 42));
    assert(std::strcmp(message, "nepenthes.batch.end token=42") == 0);
    assert(!nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "other", 42));
    assert(!nepenthes::nvtx_detail::format_batch_message(message, sizeof(message), "start", 0));
    assert(!nepenthes::nvtx_detail::format_batch_message(message, 8, "start", 42));

    // With NEPENTHES_NVTX unset, scope construction/destruction is a no-op and
    // must not require NVTX headers, a link-time library, or a live sink.
    nepenthes::BatchNvtxScope disabled(42);
    nepenthes::BatchNvtxScope invalid(0);
    return 0;
}
