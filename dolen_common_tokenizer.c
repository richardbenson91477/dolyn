#include "dolen_common_tokenizer.h"
#include "dolen_common_io.h"
#include "dolen_common_mem.h"


int32_t compare_tokens(const void *_a, const void *_b) {
    return strcmp(((token_map *)_a)->_str_s, ((token_map *)_b)->_str_s);
}

int32_t str_lookup(char *_str_s, token_map *_vocab_sorted, int32_t vocab_size) {
    token_map tok = {._str_s = _str_s};
    token_map *_tm_res = bsearch(&tok, _vocab_sorted, vocab_size, sizeof(token_map), compare_tokens);
    return _tm_res ? _tm_res->id : -1;
}

void encode_segment(tokenizer *_tokenizer, char *_text_s, int32_t *_tokens, int32_t *_tokens_n) {
    if (_text_s[0] == '\0') {
        return;
    }

    char *_buf_s = a_calloc((_tokenizer->max_token_length + 1) * sizeof(char));
    char *_p = _text_s;
    while (*_p) {
        int32_t best_len = 0;
        int32_t best_id = -1;
        for (int32_t len = 1; (len <= _tokenizer->max_token_length) && (_p[len - 1]); len++) {
            if (((_p[len - 1] & 0xC0) == 0x80) &&
                    ((len < _tokenizer->max_token_length) &&
                     (_p[len]))) {
                continue;
            }

            strncpy(_buf_s, _p, len);
            _buf_s[len] = '\0';

            int32_t id = str_lookup(_buf_s, _tokenizer->_vocab_sorted, _tokenizer->vocab_size);
            if (id != -1) {
                best_len = len;
                best_id = id;
            }
        }

        if (best_id != -1) {
            _tokens[(*_tokens_n)++] = best_id;
            _p += best_len;
        }
        else {
            char byte_str[8];
            snprintf(byte_str, sizeof(byte_str), "<0x%02X>", (unsigned char)*_p);
            int32_t id = str_lookup(byte_str, _tokenizer->_vocab_sorted, _tokenizer->vocab_size);
            
            if (id != -1) {
                _tokens[(*_tokens_n)++] = id;
            } else {
                _tokens[(*_tokens_n)++] = (unsigned char)*_p + 3;
            }
            _p++;
        }
    }

    free(_buf_s);
}

bool encode(tokenizer *_tokenizer, char *_text_s, int32_t bos_token, int8_t eos, int32_t *_tokens, int32_t *_tokens_n) {
    if (! _text_s) {
        log_msg(stderr, "ERROR: Cannot encode NULL text\n");
        return false;
    }

    *_tokens_n = 0;
    char *_input_s = _text_s;
    if (bos_token > 0) {
        _tokens[(*_tokens_n)++] = bos_token;

        const char *_bos_piece_s = _tokenizer->__vocab[bos_token];
        if (_bos_piece_s) {
            size_t bos_len = strlen(_bos_piece_s);
            if ((bos_len > 0) &&
                    (! strncmp(_input_s, _bos_piece_s, bos_len))) {
                _input_s += bos_len;
            }
        }
    }

    encode_segment(_tokenizer, _input_s, _tokens, _tokens_n);

    if (eos) {
        _tokens[(*_tokens_n)++] = _tokenizer->eos_id;
    }

    return true;
}

char *decode(tokenizer *_tokenizer, int32_t token) {
    char *_piece_s = _tokenizer->__vocab[token];

    unsigned char byte_val;
    if (sscanf(_piece_s, "<0x%02hhX>", &byte_val) == 1) {
        _piece_s = (char *)_tokenizer->_byte_pieces_s + byte_val * 2;
    }

    return _piece_s;
}

bool build_tokenizer(tokenizer *_tokenizer, const char *_tokenizer_path_s, int32_t vocab_size) {
    FILE *_file = fopen(_tokenizer_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: Couldn't open %s\n", _tokenizer_path_s);
        return false;
    }

    if (! tokenizer_read_from_file(_file, vocab_size, _tokenizer)) {
        fclose(_file);
        return false;
    }

    fclose(_file);
    return true;
}

void free_tokenizer(tokenizer *_tokenizer) {
    if (! _tokenizer) {
        return;
    }

    for (int32_t i = 0; i < _tokenizer->vocab_size; i++) {
        free(_tokenizer->__vocab[i]);
    }

    free(_tokenizer->__vocab);
    free(_tokenizer->_vocab_sorted);
}

bool tokenizer_write_to_file(FILE *_file, const tokenizer *_tokenizer) {
    if (fwrite(&_tokenizer->max_token_length, sizeof(int32_t), 1, _file) != 1) {
        return false;
    }

    for (int32_t i = 0; i < _tokenizer->vocab_size; i++) {
        int32_t len = (int32_t)strlen(_tokenizer->__vocab[i]);

        if (fwrite(&len, sizeof(int32_t), 1, _file) != 1) {
            return false;
        }

        if ((len > 0) &&
                (fwrite(_tokenizer->__vocab[i], len, 1, _file) != 1)) {
            return false;
        }
    }

    return true;
}

bool tokenizer_read_from_file(FILE *_file, int32_t vocab_size, tokenizer *_tokenizer) {
    if (fread(&_tokenizer->max_token_length, sizeof(int32_t), 1, _file) != 1) {
        log_msg(stderr, "ERROR: Failed read: max_token_length\n");
        return false;
    }

    _tokenizer->vocab_size = vocab_size;
    _tokenizer->__vocab = (char **)a_calloc((size_t)vocab_size * sizeof(char *));

    for (int32_t i = 0; i < 256; i++) {
        _tokenizer->_byte_pieces_s[i * 2] = (unsigned char)i;
        _tokenizer->_byte_pieces_s[i * 2 + 1] = '\0';
    }

    int32_t len;
    for (int32_t i = 0; i < vocab_size; i++) {
        if (fread(&len, sizeof(int32_t), 1, _file) != 1) {
            log_msg(stderr, "ERROR: Failed read: len (%u)\n", i);
            return false;
        }

        _tokenizer->__vocab[i] = (char *)a_calloc((size_t)len + 1);

        if ((len > 0) &&
                (fread(_tokenizer->__vocab[i], len, 1, _file) != 1)) {
            log_msg(stderr, "ERROR: Failed read: vocab (%u)\n", i);
            return false;
        }

        _tokenizer->__vocab[i][len] = '\0';
    }

    _tokenizer->_vocab_sorted = a_calloc(_tokenizer->vocab_size * sizeof(token_map));
    for (int32_t i = 0; i < _tokenizer->vocab_size; i++) {
        _tokenizer->_vocab_sorted[i]._str_s = _tokenizer->__vocab[i];
        _tokenizer->_vocab_sorted[i].id = i;
    }
    qsort(_tokenizer->_vocab_sorted, _tokenizer->vocab_size, sizeof(token_map), compare_tokens);

    return true;
}

