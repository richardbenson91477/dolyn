// I/O Utilities
#ifndef DOLEN_COMMON_IO_H
#define DOLEN_COMMON_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdarg.h>
#include <errno.h>

#include "ext/json.h"


#define MAGIC_Q3 (0x30335751514c4f44) // DOLQQW30
#define MAGIC_Q3_5 (0x35335751514c4f44) // DOLQQW35
#define MAGIC_G4 (0x344d4547514c4f44) // DOLQGEM4
#define MAGIC_IG4_1 (0x31344749514c4f44) //DOLQIG41


extern char *log_path;


long time_in_ms(void);

void log_msg(FILE *stream, const char *format, ...);

void read_msg(char *buf, size_t buf_len);

float get_json_float_val(JsonValue *v, float def);

int seek_abs(FILE *f, uint64_t offset);


#endif // DOLEN_COMMON_IO_H

