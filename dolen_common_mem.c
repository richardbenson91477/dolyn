#include "dolen_common_mem.h"


void *a_calloc(size_t size) {
    if (! size) {
        return NULL;
    }

    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size)) {
        return NULL;
    }

    memset(ptr, 0, size);
    return ptr;
}

uint64_t read_le64(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

