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


#define MAGIC_MS (0x5254534d514c4f44) // DOLQMSTR
#define MAGIC_Q2 (0x30325751514c4f44) // DOLQQW20
#define MAGIC_Q3 (0x30335751514c4f44) // DOLQQW30
#define MAGIC_Q3_5 (0x35335751514c4f44) // DOLQQW35
#define MAGIC_G4 (0x344d4547514c4f44) // DOLQGEM4
#define MAGIC_IG4_1 (0x31344749514c4f44) // DOLQIG41
#define MAGIC_L3 (0x4d414c4c514c4f44) // DOLQLLAM

extern char *_log_path;


int64_t time_in_ms();

void log_msg(FILE *_file, const char *_format, ...);

void read_msg(char *_buf, size_t buf_len);

char *read_file(const char *_path_s);

float get_json_float_val(JsonValue *_json_val, float def);

int32_t seek_abs(FILE *_file, uint64_t offset);


#endif // DOLEN_COMMON_IO_H

