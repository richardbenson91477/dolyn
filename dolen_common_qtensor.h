// Quantized Tensor
#ifndef DOLEN_COMMON_QTENSOR_H
#define DOLEN_COMMON_QTENSOR_H

#include <math.h>
#include <stdio.h>
#include <limits.h>


#define GROUP_SIZE 64 // Group Size


typedef enum {
    Q_TYPE_F32 = 0,
    Q_TYPE_F16 = 1,
    Q_TYPE_Q8 = 2,
    Q_TYPE_Q6 = 3,
    Q_TYPE_Q4 = 4,
} q_type_t;

typedef struct {
    void *_data;
    float *_scales;
    int rows;
    int cols;
    q_type_t type;
} qtensor;


void dequantize_row(float *_output, const qtensor *_qt, int row_idx);

void matmul_qt(float *_output, const float *_input, const qtensor *_qt);

void quantize_vec(qtensor *_xq, const float *_x, int n);

void matmul_qq(float *_output, const qtensor *_x, const qtensor *_w);

void free_qt(qtensor *_qt);

void free_qt_array(qtensor *_qt_arr, int arr_n);

void read_qt(FILE *_file, qtensor *_qt);


#endif // DOLEN_COMMON_QTENSOR_H

