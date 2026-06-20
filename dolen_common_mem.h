// I/O Utilities
#ifndef DOLEN_COMMON_MEM_H
#define DOLEN_COMMON_MEM_H

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdarg.h>
#include <errno.h>
#include <limits.h>
#include <sys/types.h>


void *a_calloc(size_t size);

uint64_t read_le64(const uint8_t bytes[8]);


#endif // DOLEN_COMMON_MEM_H

