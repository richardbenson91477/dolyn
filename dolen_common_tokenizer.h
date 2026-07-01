#ifndef DOLEN_COMMON_TOKENIZER_H
#define DOLEN_COMMON_TOKENIZER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


typedef struct {
    char *_str_s;
    int32_t id;
} token_map;

typedef struct {
    char **__vocab;
    token_map *_vocab_sorted;
    int32_t vocab_size;
    uint32_t max_token_length;
    unsigned char _byte_pieces_s[512];
    int32_t bos_id;
    int32_t eos_id;
    int32_t im_end_id;
    int32_t is_sorted;
} tokenizer;


int32_t compare_tokens(const void *_a, const void *_b);

int32_t str_lookup(char *_str_s, token_map *_vocab_sorted, int32_t vocab_size);

void encode_segment(tokenizer *_tokenizer, char *_text_s, int32_t *_tokens, int32_t *_tokens_n);

void encode(tokenizer *_tokenizer, char *_text_s, int32_t bos_id, int8_t eos_id, int32_t *_tokens, int32_t *_tokens_n);

char *decode(tokenizer *_tokenizer, int32_t token);

void build_tokenizer(tokenizer *_tokenizer, const char *_tokenizer_path_s, int32_t vocab_size);

void free_tokenizer(tokenizer *_tokenizer);

int32_t tokenizer_write_to_file(FILE *_file, const tokenizer *_tokenizer);

int32_t tokenizer_read_from_file(FILE *_file, int32_t vocab_size, tokenizer *_tokenizer);


#endif // DOLEN_COMMON_TOKENIZER_H

