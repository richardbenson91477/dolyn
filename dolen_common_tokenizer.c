#include "dolen_common_tokenizer.h"
#include "dolen_common_io.h"
#include "dolen_common_mem.h"


int compare_tokens(const void *_a, const void *_b) {
    return strcmp(((token_map *)_a)->_str_s, ((token_map *)_b)->_str_s);
}

int str_lookup(char *_str_s, token_map *_vocab_sorted, int vocab_size) {
    token_map tok = {._str_s = _str_s};
    token_map *res = bsearch(&tok, _vocab_sorted, vocab_size, sizeof(token_map), compare_tokens);
    return res ? res->id : -1;
}

void encode_segment(tokenizer *_tokenizer, char *_text_s, int *tokens, int *tokens_n) {
    if (_text_s[0] == '\0') {
        return;
    }
    char *_buf_s = a_calloc((_tokenizer->max_token_length + 1) * sizeof(char));
    char *_p = _text_s;
    while (*_p) {
        int best_len = 0, best_id = -1;
        for (int len = 1; (len <= _tokenizer->max_token_length) &&
                (_p[len - 1]); len++) {
            if (((_p[len - 1] & 0xC0) == 0x80) &&
                    ((len < _tokenizer->max_token_length) &&
                     (_p[len]))) {
                continue;
            }
            strncpy(_buf_s, _p, len);
            _buf_s[len] = '\0';
            int id = str_lookup(_buf_s, _tokenizer->_vocab_sorted, _tokenizer->vocab_size);
            if (id != -1) {
                best_len = len;
                best_id = id;
            }
        }
        if (best_id != -1) {
            tokens[(*tokens_n)++] = best_id;
            _p += best_len;
        }
        else {
            tokens[(*tokens_n)++] = (unsigned char)*_p + 3;
            _p++;
        }
    }
    free(_buf_s);
}

void encode(tokenizer *_tokenizer, char *_text_s, int bos_token, int8_t eos, int *tokens, int *tokens_n) {
    if (! _text_s) {
        log_msg(stderr, "ERROR: Cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (! _tokenizer->_vocab_sorted) {
        _tokenizer->_vocab_sorted = a_calloc(_tokenizer->vocab_size * sizeof(token_map));
        for (int i = 0; i < _tokenizer->vocab_size; i++) {
            _tokenizer->_vocab_sorted[i]._str_s = _tokenizer->__vocab[i];
            _tokenizer->_vocab_sorted[i].id = i;
        }
        qsort(_tokenizer->_vocab_sorted, _tokenizer->vocab_size, sizeof(token_map), compare_tokens);
        _tokenizer->is_sorted = 1;
    }

    *tokens_n = 0;
    char *_input_s = _text_s;
    if (bos_token > 0) {
        tokens[(*tokens_n)++] = bos_token;

        const char *bos_piece = _tokenizer->__vocab[bos_token];
        if (bos_piece) {
            size_t bos_len = strlen(bos_piece);
            if ((bos_len > 0) &&
                    (! strncmp(_input_s, bos_piece, bos_len))) {
                _input_s += bos_len;
            }
        }
    }

    char *segment = a_calloc(strlen(_input_s) + 1);

    char *_p = _input_s;
    while (*_p) {
        int found_special = 0;
        for (int i = 0; i < _tokenizer->token_special_n; i++) {
            size_t len = strlen(_tokenizer->_tokens_special[i]._str_s);
            if (! strncmp(_p, _tokenizer->_tokens_special[i]._str_s, len)) {
                tokens[(*tokens_n)++] = _tokenizer->_tokens_special[i].id;
                _p += len;
                found_special = 1;
                break;
            }
        }
        if (! found_special) {
            size_t seg_len = 0;
            char *seg_start = _p;
            while (*_p) {
                int is_special_start = 0;
                for (int i = 0; i < _tokenizer->token_special_n; i++) {
                    if (! strncmp(_p, _tokenizer->_tokens_special[i]._str_s, strlen(_tokenizer->_tokens_special[i]._str_s))) {
                        is_special_start = 1;
                        break;
                    }
                }
                if (is_special_start) {
                    break;
                }

                _p++;
                seg_len++;
            }
            if (seg_len > 0) {
                strncpy(segment, seg_start, seg_len);
                segment[seg_len] = '\0';

                encode_segment(_tokenizer, segment, tokens, tokens_n);
            }
        }
    }

    if (eos) {
        tokens[(*tokens_n)++] = 2;
    }

    free(segment);
}

char *decode(tokenizer *_tokenizer, int token, bool debug_) {
    char *piece = _tokenizer->__vocab[token];

    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)_tokenizer->_byte_pieces_s + byte_val * 2;
    }

    if (debug_) {
        log_msg(stdout, "\nDEBUG: token: %u piece:", token);
        for (int c = 0; c < strlen(piece); c++) {
            log_msg(stdout, "<%x>", (unsigned char)(piece[c]));
        }
    }

    return piece;
}

void build_tokenizer(tokenizer *_tokenizer, const char *_tokenizer_path_s, int vocab_size, token_map *_tokens_special) {
    _tokenizer->vocab_size = vocab_size;
    _tokenizer->__vocab = (char **)a_calloc(vocab_size * sizeof(char *));
    _tokenizer->_vocab_sorted = NULL;
    _tokenizer->is_sorted = 0;
    _tokenizer->_tokens_special = _tokens_special;

    _tokenizer->token_special_n = 0;
    if (_tokens_special) {
        while (_tokens_special[_tokenizer->token_special_n]._str_s) {
            _tokenizer->token_special_n++;
        }
    }

    for (int i = 0; i < 256; i++) {
        _tokenizer->_byte_pieces_s[i * 2] = (unsigned char)i;
        _tokenizer->_byte_pieces_s[i * 2 + 1] = '\0';
    }

    FILE *file = fopen(_tokenizer_path_s, "rb");
    if (! file) {
        log_msg(stderr, "ERROR: Couldn't open %s\n", _tokenizer_path_s);
        exit(EXIT_FAILURE);
    }

    if (fread(&_tokenizer->max_token_length, sizeof(int), 1, file) != 1) {
        log_msg(stderr, "ERROR: Failed read: max_token_length\n");
        exit(EXIT_FAILURE);
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, file) != 1) {
            log_msg(stderr, "ERROR: Failed read: len (%u)\n", i);
            exit(EXIT_FAILURE);
        }

        _tokenizer->__vocab[i] = (char *)a_calloc(len + 1);
        if (len > 0) {
            if (fread(_tokenizer->__vocab[i], len, 1, file) != 1) {
                log_msg(stderr, "ERROR: Failed read: vocab (%u)\n", i);
                exit(EXIT_FAILURE);
            }
        }
        _tokenizer->__vocab[i][len] = '\0';
    }

    fclose(file);
}

void free_tokenizer(tokenizer *_tokenizer) {
    if (! _tokenizer) {
        return;
    }

    for (int i = 0; i < _tokenizer->vocab_size; i++) {
        free(_tokenizer->__vocab[i]);
    }

    free(_tokenizer->__vocab);
    free(_tokenizer->_vocab_sorted);
}

int tokenizer_write_to_file(FILE *f, const tokenizer *_tokenizer) {
    if (fwrite(&_tokenizer->max_token_length, sizeof(int), 1, f) != 1) {
        return -1;
    }

    for (int i = 0; i < _tokenizer->vocab_size; i++) {
        int len = (int)strlen(_tokenizer->__vocab[i]);

        if (fwrite(&len, sizeof(int), 1, f) != 1) {
            return -1;
        }

        if (len > 0 && fwrite(_tokenizer->__vocab[i], len, 1, f) != 1) {
            return -1;
        }
    }
    return 0;
}

int tokenizer_read_from_file(FILE *f, int vocab_size, tokenizer *_tokenizer) {
    if (fread(&_tokenizer->max_token_length, sizeof(int), 1, f) != 1) {
        return -1;
    }

    _tokenizer->__vocab = (char **)a_calloc((size_t)vocab_size * sizeof(char *));
    _tokenizer->_vocab_sorted = NULL;
    _tokenizer->is_sorted = 0;
    _tokenizer->vocab_size = vocab_size;
    _tokenizer->_tokens_special = NULL;
    _tokenizer->token_special_n = 0;

    for (int i = 0; i < 256; i++) {
        _tokenizer->_byte_pieces_s[i * 2] = (unsigned char)i;
        _tokenizer->_byte_pieces_s[i * 2 + 1] = '\0';
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&len, sizeof(int), 1, f) != 1) {
            return -1;
        }

        _tokenizer->__vocab[i] = (char *)a_calloc((size_t)len + 1);

        if (len > 0 && fread(_tokenizer->__vocab[i], len, 1, f) != 1) {
            return -1;
        }

        _tokenizer->__vocab[i][len] = '\0';
    }

    return 0;
}

