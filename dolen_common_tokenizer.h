#ifndef DOLEN_COMMON_TOKENIZER_H
#define DOLEN_COMMON_TOKENIZER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


typedef struct {
    char *str;
    int id;
} token_map;

typedef struct {
    char **vocab;
    token_map *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
    token_map *special_tokens;
    int n_special_tokens;
    int bos_token_id;
    int eos_token_id;
    int im_end_id;
    int is_sorted;
} Tokenizer;


int compare_tokens(const void *a, const void *b);
int str_lookup(char *str, token_map *sorted_vocab, int vocab_size);
void encode_segment(Tokenizer *t, char *text, int *tokens, int *tokens_n);
void encode(Tokenizer *t, char *text, int bos_token, int8_t eos, int *tokens, int *tokens_n);
char *decode(Tokenizer *t, int token, bool debug_);
void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size, token_map *special_tokens);
void free_tokenizer(Tokenizer *t);
int tokenizer_write_to_file(FILE *f, const Tokenizer *t);
int tokenizer_read_from_file(FILE *f, int vocab_size, Tokenizer *t);


#endif // DOLEN_COMMON_TOKENIZER_H
