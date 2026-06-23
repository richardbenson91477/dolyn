#include "dolen_common_tokenizer.h"
#include "dolen_common_io.h"
#include "dolen_common_mem.h"


int compare_tokens(const void *a, const void *b) {
    return strcmp(((token_map *)a)->str, ((token_map *)b)->str);
}

int str_lookup(char *str, token_map *sorted_vocab, int vocab_size) {
    token_map tok = { .str = str };
    token_map *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(token_map), compare_tokens);
    return res ? res->id : -1;
}

void encode_segment(Tokenizer *t, char *text, int *tokens, int *tokens_n) {
    if (text[0] == '\0') {
        return;
    }
    char *str_buf = a_calloc((t->max_token_length + 1) * sizeof(char));
    char *pos = text;
    while (*pos) {
        int best_len = 0, best_id = -1;
        for (int len = 1; (len <= t->max_token_length) &&
                (pos[len - 1]); len++) {
            if (((pos[len - 1] & 0xC0) == 0x80) &&
                    ((len < t->max_token_length) &&
                     (pos[len]))) {
                continue;
            }
            strncpy(str_buf, pos, len);
            str_buf[len] = '\0';
            int id = str_lookup(str_buf, t->sorted_vocab, t->vocab_size);
            if (id != -1) {
                best_len = len;
                best_id = id;
            }
        }
        if (best_id != -1) {
            tokens[(*tokens_n)++] = best_id;
            pos += best_len;
        }
        else {
            tokens[(*tokens_n)++] = (unsigned char)*pos + 3;
            pos++;
        }
    }
    free(str_buf);
}

void encode(Tokenizer *t, char *text, int bos_token, int8_t eos, int *tokens, int *tokens_n) {
    if (text == NULL) {
        log_msg(stderr, "ERROR: Cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (! t->sorted_vocab) {
        t->sorted_vocab = a_calloc(t->vocab_size * sizeof(token_map));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(token_map), compare_tokens);
    }

    *tokens_n = 0;
    char *input = text;
    if (bos_token > 0) {
        tokens[(*tokens_n)++] = bos_token;

        const char *bos_piece = t->vocab[bos_token];
        if (bos_piece) {
            size_t bos_len = strlen(bos_piece);
            if ((bos_len > 0) &&
                    (! strncmp(input, bos_piece, bos_len))) {
                input += bos_len;
            }
        }
    }

    char *segment = a_calloc(strlen(input) + 1);

    char *pos = input;
    while (*pos) {
        int found_special = 0;
        for (int i = 0; i < t->n_special_tokens; i++) {
            size_t len = strlen(t->special_tokens[i].str);
            if (! strncmp(pos, t->special_tokens[i].str, len)) {
                tokens[(*tokens_n)++] = t->special_tokens[i].id;
                pos += len;
                found_special = 1;
                break;
            }
        }
        if (! found_special) {
            size_t seg_len = 0;
            char *seg_start = pos;
            while (*pos) {
                int is_special_start = 0;
                for (int i = 0; i < t->n_special_tokens; i++) {
                    if (! strncmp(pos, t->special_tokens[i].str, strlen(t->special_tokens[i].str))) {
                        is_special_start = 1;
                        break;
                    }
                }
                if (is_special_start) {
                    break;
                }

                pos++;
                seg_len++;
            }
            if (seg_len > 0) {
                strncpy(segment, seg_start, seg_len);
                segment[seg_len] = '\0';

                encode_segment(t, segment, tokens, tokens_n);
            }
        }
    }

    if (eos) {
        tokens[(*tokens_n)++] = 2;
    }

    free(segment);
}

char *decode(Tokenizer *t, int token, bool debug_) {
    char *piece = t->vocab[token];

    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }

    if (debug_) {
        log_msg(stdout, "\nDEBUG: token: %u piece:", token);
        for (int c = 0; c < strlen(piece); c++) {
            log_msg(stdout, "<%x>", (unsigned char)(piece[c]));
        }
    }

    return piece;
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size, token_map *special_tokens) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)a_calloc(vocab_size * sizeof(char *));
    t->sorted_vocab = NULL;
    t->special_tokens = special_tokens;

    t->n_special_tokens = 0;
    if (special_tokens) {
        while (special_tokens[t->n_special_tokens].str) {
            t->n_special_tokens++;
        }
    }

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(tokenizer_path, "rb");
    if (! file) {
        log_msg(stderr, "ERROR: Couldn't open %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        log_msg(stderr, "ERROR: Failed read: max_token_length\n");
        exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) {
            log_msg(stderr, "ERROR: Failed read: len (%u)\n", i);
            exit(EXIT_FAILURE);
        }

        t->vocab[i] = (char *)a_calloc(len + 1);
        if (len > 0) {
            if (fread(t->vocab[i], len, 1, file) != 1) {
                log_msg(stderr, "ERROR: Failed read: vocab (%u)\n", i);
                exit(EXIT_FAILURE);
            }
        }
        t->vocab[i][len] = '\0';
    }

    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }

    free(t->vocab);
    free(t->sorted_vocab);
}

