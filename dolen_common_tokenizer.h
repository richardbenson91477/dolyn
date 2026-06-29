#ifndef DOLEN_COMMON_TOKENIZER_H
#define DOLEN_COMMON_TOKENIZER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


typedef struct {
    char *_str_s;
    int id;
} token_map;

typedef struct {
    char **__vocab;
    token_map *_vocab_sorted;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char _byte_pieces_s[512];
    int bos_id;
    int eos_id;
    int im_end_id;
    int is_sorted;
} tokenizer;


int compare_tokens(const void *_a, const void *_b);

int str_lookup(char *_str_s, token_map *_vocab_sorted, int vocab_size);

void encode_segment(tokenizer *_tokenizer, char *_text_s, int *_tokens, int *_tokens_n);

void encode(tokenizer *_tokenizer, char *_text_s, int bos_id, int8_t eos_id, int *_tokens, int *_tokens_n);

char *decode(tokenizer *_tokenizer, int token);

void build_tokenizer(tokenizer *_tokenizer, const char *_tokenizer_path_s, int vocab_size);

void free_tokenizer(tokenizer *_tokenizer);

int tokenizer_write_to_file(FILE *_file, const tokenizer *_tokenizer);

int tokenizer_read_from_file(FILE *_file, int vocab_size, tokenizer *_tokenizer);


#endif // DOLEN_COMMON_TOKENIZER_H

