#include "dolen_common_io.h"

char *log_path = NULL;

long time_in_ms(void) {
    struct timespec time;

    clock_gettime(CLOCK_REALTIME, &time);

    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void log_msg(FILE *stream, const char *format, ...) {
    va_list args1;
    va_start(args1, format);

    if (log_path) {
        va_list args2;
        va_copy(args2, args1);

        FILE *log_file = fopen(log_path, "a");

        if (log_file) {
            vfprintf(log_file, format, args2);
            fclose(log_file);
        }
        else {
            fprintf(stderr, "ERROR: can't open log file\n");
            exit(EXIT_FAILURE);
        }

        va_end(args2);
    }

    if (stream) {
        vfprintf(stream, format, args1);
        fflush(stream);
    }

    va_end(args1);
}

void read_msg(char *buf, size_t buf_len) {
    if (! buf_len) {
        return;
    }

    char *p = buf;
    size_t rem = buf_len;

    while (rem > 1) {
        if (fgets(p, rem, stdin) == NULL) {
            break;
        }

        size_t len = strlen(p);
        if (len < 2) {
            break;
        }

        if ((p[len - 2] == '\\') &&
                (p[len - 1] == '\n')) {
            p[len - 2] = '\n';
            p += len - 1;
            rem -= len - 1;
        }
        else {
            break;
        }
    }

    log_msg(NULL, "%s", buf);
}

float get_json_float_val(JsonValue *v, float def) {
    if (! v) {
        return def;
    }
    if (v->type == JSON_NUMBER) {
        return (float)v->data.number;
    }
    if (v->type == JSON_STRING) {
        return (float)atof(v->data.string);
    }
    return def;
}

int seek_abs(FILE *f, uint64_t offset) {
    if (offset > (uint64_t)INT64_MAX) {
        log_msg(stderr, "ERROR: File offset is too large\n");
        return -1;
    }
    if (fseeko(f, (off_t)offset, SEEK_SET)) {
        log_msg(stderr, "ERROR: Seek failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

