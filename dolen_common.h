#ifndef DOLEN_COMMON_H
#define DOLEN_COMMON_H

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


// Quantized Tensor

#define GS 64 // Group Size

typedef struct {
    int8_t *q;
    float *s;
    int rows;
    int cols;
} qtensor;

void dequantize_row(float *output, const qtensor *qt, int row_idx);
void matmul_qt(float *output, const float *input, const qtensor *qt);
void quantize_vec(qtensor *xq, const float *x, int n);
void matmul_qq(float *output, const qtensor *x, const qtensor *w);
void free_qt(qtensor *qt);
void free_qt_array(qtensor *arr, int n);
void read_qt(FILE *f, qtensor *qt);
void write_qt(FILE *f, qtensor *qt);


// Math Utilities

void softmax(float *x, int size);
float silu(float x);
float sigmoid(float x);
float softplus(float x);
void l2norm(float *x, int size);
float matmul_scalar(float *x, float *w, int n);


// Tokenizer

typedef struct {
    char *str;
    int id;
} token_map;

typedef struct {
    char **vocab;
    float *vocab_scores;
    token_map *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
    token_map *special_tokens;
    int n_special_tokens;
} Tokenizer;

int compare_tokens(const void *a, const void *b);
int str_lookup(char *str, token_map *sorted_vocab, int vocab_size);
void encode_segment(Tokenizer *t, char *text, int *tokens, int *tokens_n);
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *tokens_n);
char *decode(Tokenizer *t, int token, bool debug);
void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size, token_map *special_tokens);
void free_tokenizer(Tokenizer *t);


// Sampler

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probs, int n);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
int sample_mult(float *probs, int n, float coin);
int compare_prob(const void *a, const void *b);
int sample_topp(float *probs, int n, float topp, ProbIndex *probindex, float coin);
int sample(Sampler *sampler, float *logits);
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler *sampler);


// I/O Utilities

extern char *log_path;

long time_in_ms(void);
void log_msg(FILE *stream, const char *format, ...);
void read_msg(char *buf, size_t buf_len);
void *a_calloc(size_t size);

// Common Model Interface

#define PROMPT_N_MAX_DEFAULT 32768
#define TEMP_DEFAULT 0.2
#define TOP_P_DEFAULT 0.95


typedef struct {
    void *model;

    float *(*forward)(void *model, int token, int pos);

    void  (*free_model)(void *model);

    int seq_n_max;

    int vocab_size;
    int im_end_id;
    token_map *special_tokens;

} model_iface;


void generate_common(model_iface *model_i, Tokenizer *tokenizer, Sampler *sampler,
        char *prompt, int steps_n_max);
void chat_common(model_iface *model_i, Tokenizer *tokenizer, Sampler *sampler,
        char *system_prompt, char *init_prompt, int prompt_n_max, int steps_n_max, bool _debug);
void error_usage(const char *prog_name);
int common_main(int argc, char *argv[],
        model_iface *(*init_fn)(const char *model_path, int seq_n_max),
        const char *prog_name);


#endif // DOLEN_COMMON_H

