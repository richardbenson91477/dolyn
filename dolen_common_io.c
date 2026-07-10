#include "dolen_common_io.h"
#include "dolen_common_mem.h"


char *_log_path = NULL;


int64_t time_in_ms() {
    struct timespec time;

    clock_gettime(CLOCK_REALTIME, &time);

    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void log_msg(FILE *_file, const char *_format, ...) {
    va_list args1;
    va_start(args1, _format);

    if (_log_path) {
        va_list args2;
        va_copy(args2, args1);

        FILE *_log_file = fopen(_log_path, "a");

        if (_log_file) {
            vfprintf(_log_file, _format, args2);
            fclose(_log_file);
        }
        else {
            fprintf(stderr, "ERROR: can't open log file\n");
            exit(EXIT_FAILURE);
        }

        va_end(args2);
    }

    if (_file) {
        vfprintf(_file, _format, args1);
        fflush(_file);
    }

    va_end(args1);
}

void read_msg(char *_buf, size_t buf_len) {
    if (! buf_len) {
        return;
    }

    char *_p = _buf;
    size_t rem = buf_len;

    while (rem > 1) {
        if (fgets(_p, rem, stdin) == NULL) {
            break;
        }

        size_t len = strlen(_p);
        if (len < 2) {
            break;
        }

        if ((_p[len - 2] == '\\') &&
                (_p[len - 1] == '\n')) {
            _p[len - 2] = '\n';
            _p += len - 1;
            rem -= len - 1;
        }
        else {
            break;
        }
    }

    log_msg(NULL, "%s", _buf);
}

char *read_file(const char *_path_s) {
    FILE *_file = fopen(_path_s, "r");
    if (! _file) {
        log_msg(stderr, "ERROR: Couldn't open prompt file %s\n", _path_s);
        return(NULL);
    }

    fseek(_file, 0, SEEK_END);
    int64_t f_len = ftell(_file);
    fseek(_file, 0, SEEK_SET);

    if (f_len < 0) {
        log_msg(stderr, "ERROR: Failed to determine size of prompt file %s\n", _path_s);
        fclose(_file);
        return NULL;
    }

    char *_in_s = (char *)a_calloc(f_len + 1);
    if (! _in_s) {
        log_msg(stderr, "ERROR: Memory allocation failed in read_file\n");
        fclose(_file);
        return(NULL);
    }

    size_t read_bytes = fread(_in_s, 1, f_len, _file);
    _in_s[read_bytes] = '\0';
    fclose(_file);

    return _in_s;
}

float get_json_float_val(JsonValue *_json_val, float def) {
    if (! _json_val) {
        return def;
    }
    if (_json_val->type == JSON_NUMBER) {
        return (float)_json_val->data.number;
    }
    if (_json_val->type == JSON_STRING) {
        return (float)atof(_json_val->data.string);
    }
    return def;
}

int32_t seek_abs(FILE *_file, uint64_t offset) {
    if (offset > (uint64_t)INT64_MAX) {
        log_msg(stderr, "ERROR: File offset is too large\n");
        return -1;
    }
    if (fseeko(_file, (off_t)offset, SEEK_SET)) {
        log_msg(stderr, "ERROR: Seek failed: %s\n", strerror(errno));
        return -1;
    }
    return 0;
}

