// I/O Utilities
#ifndef DOLEN_COMMON_IO_H
#define DOLEN_COMMON_IO_H

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

#include "ext/json.h"


extern char *log_path;


long time_in_ms(void);

void log_msg(FILE *stream, const char *format, ...);

void read_msg(char *buf, size_t buf_len);

float get_json_float_val(JsonValue *v, float def);

int seek_abs(FILE *f, uint64_t offset);


#endif // DOLEN_COMMON_IO_H

