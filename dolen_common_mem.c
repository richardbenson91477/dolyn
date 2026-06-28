#include "dolen_common_mem.h"


void *a_calloc(size_t size) {
    if (! size) {
        return NULL;
    }

    void *_p = NULL;
    if (posix_memalign(&_p, 64, size)) {
        return NULL;
    }

    memset(_p, 0, size);
    return _p;
}

uint64_t read_le64(const uint8_t bytes[8]) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

