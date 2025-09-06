#include <cstdint>

extern "C" uint32_t count_nonzero(const uint8_t* data, uint32_t len) {
    if (data == nullptr) {
        return 0u;
    }
    uint32_t count = 0u;
    for (uint32_t i = 0; i < len; ++i) {
        count += (data[i] != 0u) ? 1u : 0u;
    }
    return count;
}
