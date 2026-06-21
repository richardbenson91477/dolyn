// I/O Utilities
#ifndef DOLEN_COMMON_MEM_H
#define DOLEN_COMMON_MEM_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


void *a_calloc(size_t size);

uint64_t read_le64(const uint8_t bytes[8]);


#endif // DOLEN_COMMON_MEM_H

