// Quantized Tensor
#ifndef DOLEN_COMMON_QTENSOR_H
#define DOLEN_COMMON_QTENSOR_H

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


#define GROUP_SIZE 64 // Group Size


typedef enum {
    Q_TYPE_F32 = 0,
    Q_TYPE_F16 = 1,
    Q_TYPE_Q8  = 2
} q_type_t;

typedef struct {
    void *data;     // Points to float*, _Float16*, or int8_t* depending on type
    float *s;       // Scales for Q8 (NULL for F32/F16)
    int rows;
    int cols;
    q_type_t type;
} qtensor;


void dequantize_row(float *output, const qtensor *qt, int row_idx);

void matmul_qt(float *output, const float *input, const qtensor *qt);

void quantize_vec(qtensor *xq, const float *x, int n);

void matmul_qq(float *output, const qtensor *x, const qtensor *w);

void free_qt(qtensor *qt);

void free_qt_array(qtensor *arr, int n);

void read_qt(FILE *f, qtensor *qt);

void write_qt(FILE *f, qtensor *qt);


#endif // DOLEN_COMMON_QTENSOR_H
