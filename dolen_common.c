#include "dolen_common.h"
#include <stdarg.h>


char *log_path = NULL;


void dequantize_row(float *output, const qtensor *qt, int row_idx) {
    if (row_idx >= qt->rows || row_idx < 0) {
        log_msg(stderr, "ERROR: Row index %d out of bounds (max %d)\n", row_idx, qt->rows);
        exit(EXIT_FAILURE);
    }
    int num_groups = (qt->cols + GS - 1) / GS;
    const float *row_s = qt->s + row_idx * num_groups;
    const int8_t *row_q = qt->q + row_idx * qt->cols;

    for (int g = 0; g < num_groups; g++) {
        int start = g * GS;
        int end = start + GS;
        if (end > qt->cols) {
            end = qt->cols;
        }

        float scale = row_s[g];
        for (int j = start; j < end; j++) {
            output[j] = (float)row_q[j] * scale;
        }
    }
}

void matmul_qt(float *restrict output, const float *restrict input, const qtensor *restrict qt) {
    int cols = qt->cols;
    int rows = qt->rows;
    int num_groups = (cols + GS - 1) / GS;
    int full_groups = cols / GS;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        const int8_t *row_q = qt->q + (size_t)i * cols;
        const float  *row_s = qt->s + (size_t)i * num_groups;

        for (int g = 0; g < full_groups; g++) {
            float group_sum = 0.0f;
            const int offset = g * GS;
            #pragma omp simd reduction(+:group_sum)
            for (int j = 0; j < GS; j++) {
                group_sum += input[offset + j] * (float)row_q[offset + j];
            }
            sum += group_sum * row_s[g];
        }

        int rem_start = full_groups * GS;
        if (rem_start < cols) {
            float group_sum = 0.0f;
            #pragma omp simd reduction(+:group_sum)
            for (int j = rem_start; j < cols; j++) {
                group_sum += input[j] * (float)row_q[j];
            }
            sum += group_sum * row_s[full_groups];
        }
        output[i] = sum;
    }
}

void quantize_vec(qtensor *xq, const float *x, int n) {
    int num_groups = (n + GS - 1) / GS;
    xq->rows = 1;
    xq->cols = n;
    
    #pragma omp parallel for schedule(static) if(num_groups > 32)
    for (int g = 0; g < num_groups; g++) {
        int start = g * GS;
        int end = start + GS < n ? start + GS : n;
        float wmax = 0.0f;
        for (int i = start; i < end; i++) {
            float v = fabsf(x[i]);
            if (v > wmax) wmax = v;
        }
        float scale = wmax < 1e-9f ? 1e-9f : wmax / 127.0f;
        xq->s[g] = scale;
        float inv_scale = 1.0f / scale;
        for (int i = start; i < end; i++) {
            xq->q[i] = (int8_t)roundf(x[i] * inv_scale);
        }
    }
}

void matmul_qq(float *restrict output, const qtensor *restrict x, const qtensor *restrict w) {
    int n = x->cols;
    int d = w->rows;
    int n_groups = (n + GS - 1) / GS;
    int full_groups = n / GS;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const int8_t *w_row = w->q + (size_t)i * n;
        const float *w_s = w->s + (size_t)i * n_groups;

        for (int g = 0; g < full_groups; g++) {
            int32_t acc = 0;
            const int offset = g * GS;

            #pragma omp simd reduction(+:acc)
            for (int k = 0; k < GS; k++) {
                acc += (int32_t)x->q[offset + k] * (int32_t)w_row[offset + k];
            }
            val += (float)acc * w_s[g] * x->s[g];
        }

        int rem_start = full_groups * GS;
        if (rem_start < n) {
            int32_t acc = 0;
            #pragma omp simd reduction(+:acc)
            for (int k = rem_start; k < n; k++) {
                acc += (int32_t)x->q[k] * (int32_t)w_row[k];
            }
            val += (float)acc * w_s[full_groups] * x->s[full_groups];
        }
        output[i] = val;
    }
}

void free_qt(qtensor *qt) {
    if (! qt) {
        return;
    }

    free(qt->q);
    free(qt->s);

    qt->q = NULL;
    qt->s = NULL;

    qt->rows = 0;
    qt->cols = 0;
}

void free_qt_array(qtensor *arr, int n) {
    if (! arr) {
        return;
    }

    for (int i = 0; i < n; i++) {
        free_qt(&arr[i]);
    }

    free(arr);
}

void read_qt(FILE *f, qtensor *qt) {
    fread(&qt->rows, sizeof(int), 1, f);
    fread(&qt->cols, sizeof(int), 1, f);

    int num_groups = (qt->cols + GS - 1) / GS;

    qt->q = (int8_t *)a_calloc((size_t)qt->rows * qt->cols * sizeof(int8_t));
    qt->s = (float *)a_calloc((size_t)qt->rows * num_groups * sizeof(float));

    fread(qt->q, sizeof(int8_t), (size_t)qt->rows * qt->cols, f);
    fread(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
}

void write_qt(FILE *f, qtensor *qt) {
    fwrite(&qt->rows, sizeof(int), 1, f);
    fwrite(&qt->cols, sizeof(int), 1, f);

    int num_groups = (qt->cols + GS - 1) / GS;

    fwrite(qt->q, sizeof(int8_t), (size_t)qt->rows * qt->cols, f);
    fwrite(qt->s, sizeof(float), (size_t)qt->rows * num_groups, f);
}

void softmax(float *x, int size) {
    float max_val = x[0];
    #pragma omp simd reduction(max:max_val)
    for (int i = 1; i < size; i++) {
        max_val = fmaxf(max_val, x[i]);
    }

    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    if (sum < 1e-10f) {
        float uniform = 1.0f / size;
        for (int i = 0; i < size; i++) {
            x[i] = uniform;
        }
        return;
    }

    float inv_sum = 1.0f / sum;

    #pragma omp simd
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

float silu(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float softplus(float x) {
    return logf(1.0f + expf(x));
}

void l2norm(float *x, int size) {
    float ss = 0.0f;
    #pragma omp simd reduction(+:ss)
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss + 1e-6f);
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        x[i] *= ss;
    }
}

float matmul_scalar(float *x, float *w, int n) {
    float val = 0.0f;

    #pragma omp simd reduction(+:val)
    for (int i = 0; i < n; i++) {
        val += w[i] * x[i];
    }
    return val;
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((token_map *)a)->str, ((token_map *)b)->str);
}

int str_lookup(char *str, token_map *sorted_vocab, int vocab_size) {
    token_map tok = { .str = str };
    token_map *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(token_map), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode_segment(Tokenizer *t, char *text, int *tokens, int *tokens_n) {
    if (text[0] == '\0') {
        return;
    }
    char *str_buf = a_calloc((t->max_token_length + 1) * sizeof(char));
    char *pos = text;
    while (*pos != '\0') {
        int best_len = 0, best_id = -1;
        for (int len = 1; (len <= t->max_token_length) && (pos[len-1] != '\0'); len++) {
            if (((pos[len-1] & 0xC0) == 0x80) && ((len < t->max_token_length) && (pos[len] != '\0'))) {
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
            tokens[(*tokens_n)++] = best_id; pos += best_len;
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

    if (t->sorted_vocab == NULL) {
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
            if (bos_len > 0 && strncmp(input, bos_piece, bos_len) == 0) {
                input += bos_len;
            }
        }
    }

    char *segment = a_calloc(strlen(input) + 1);

    char *pos = input;
    while (*pos != '\0') {
        int found_special = 0;
        for (int i = 0; i < t->n_special_tokens; i++) {
            size_t len = strlen(t->special_tokens[i].str);
            if (strncmp(pos, t->special_tokens[i].str, len) == 0) {
                tokens[(*tokens_n)++] = t->special_tokens[i].id;
                pos += len;
                found_special = 1;
                break;
            }
        }
        if (! found_special) {
            size_t seg_len = 0;
            char *seg_start = pos;
            while (*pos != '\0') {
                int is_special_start = 0;
                for (int i = 0; i < t->n_special_tokens; i++) {
                    if (strncmp(pos, t->special_tokens[i].str, strlen(t->special_tokens[i].str)) == 0) {
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

char *decode(Tokenizer *t, int token, bool _debug) {
    char *piece = t->vocab[token];

    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }

    if (_debug) {
        log_msg(stdout, "\nDEBUG: token: %u piece:", token);
        for (int c = 0; c < strlen(piece); c ++) {
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

int sample_argmax(float *probs, int n) {
    int max_i = 0;
    float max_p = probs[0];

    for (int i = 1; i < n; i++) {
        if (probs[i] > max_p) {
            max_i = i;
            max_p = probs[i];
        }
    }

    return max_i;
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probs, int n, float coin) {
    float cdf = 0.0f;

    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare_prob(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;

    if (a_->prob > b_->prob) {
        return -1;
    }

    if (a_->prob < b_->prob) {
        return 1;
    }

    return 0;
}

int sample_top(float *probs, int n, int topk, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);

    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probs[i];
            n0++;
        }
    }

    if (n0 == 0) {
        return sample_argmax(probs, n);
    }

    qsort(probindex, n0, sizeof(ProbIndex), compare_prob);

    if (topk > 0 && n0 > topk) {
        n0 = topk;
    }

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }

    return probindex[last_idx].index;
}

int sample(Sampler *sampler, float *logits) {
    int next;

    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        #pragma omp parallel for if (sampler->vocab_size > 4096)
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }

        softmax(logits, sampler->vocab_size);

        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        }
        else {
            next = sample_top(logits, sampler->vocab_size, sampler->topk, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, int topk, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topk = topk;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = a_calloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

long time_in_ms(void) {
    struct timespec time;

    clock_gettime(CLOCK_REALTIME, &time);

    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void log_msg(FILE *stream, const char *format, ...) {
    va_list args1;
    va_start(args1, format);

    if (log_path) {
        va_list args2;
        va_copy(args2, args1);

        FILE *log_file = fopen(log_path, "a");

        if (log_file != NULL) {
            vfprintf(log_file, format, args2);
            fclose(log_file);
        }
        else {
            fprintf(stderr, "ERROR: can't open log file\n");
            exit(EXIT_FAILURE);
        }

        va_end(args2);
    }

    if (stream) {
        vfprintf(stream, format, args1);
        fflush(stream);
    }

    va_end(args1);
}

void read_msg(char *buf, size_t buf_len) {
    if (buf_len == 0) {
        return;
    }
 
    char *p = buf;
    size_t rem = buf_len;
  
    while (rem > 1) {
        if (fgets(p, rem, stdin) == NULL) {
            break;
        }

        size_t len = strlen(p);
        if (len < 2) {
            break;
        }
 
        if ((p[len - 2] == '\\') && (p[len - 1] == '\n')) {
            p[len - 2] = '\n';
            p += len - 1;
            rem -= len - 1;
        } else {
            break;
        }
    }

    log_msg(NULL, "%s", buf);
}

void *a_calloc(size_t size) {
    if (size == 0) return NULL;

    void *ptr = NULL;
    if (posix_memalign(&ptr, 64, size) != 0) {
        return NULL;
    }

    memset(ptr, 0, size);
    return ptr;
}

void generate_common(model_iface *model_i, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps_n_max) {
    if (prompt == NULL) {
        prompt = "";
    }

    int prompt_tokens_n = 0;

    int *prompt_tokens = (int *)a_calloc((strlen(prompt) * 4 + 3) * sizeof(int));

    encode(tokenizer, prompt, model_i->bos_token_id, 0, prompt_tokens, &prompt_tokens_n);

    if (prompt_tokens_n < 1) {
        log_msg(stderr, "ERROR: Expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];

    int pos = 0;
    while (pos < steps_n_max) {
        float *logits = model_i->forward(model_i->model, token, pos);

        if (pos < (prompt_tokens_n - 1)) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }

        pos++;
        if (next == 1) {
            break;
        }

        char *piece = decode(tokenizer, next, false);
        log_msg(stdout, "%s", piece);

        token = next;

        if (start == 0) {
            start = time_in_ms();
        }
    }
    log_msg(stdout, "\n");

    if (pos > 1) {
        long end = time_in_ms();
        log_msg(stderr, "INFO: %f tokens per second.\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

static const chat_template CHAT_TEMPLATE_CHATML = {
    .first_turn_and_system =
        "<|im_start|>system\n%s<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
    .first_turn =
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
    .next_turn =
        "<|im_end|>\n"
        "<|im_start|>user\n%s<|im_end|>\n"
        "<|im_start|>assistant\n",
};

static char *render_chat_turn(const chat_template *tmpl, bool first_turn,
        const char *system_prompt, const char *prompt, int *rendered_len) {
    const char *format = NULL;
    int len;

    if (first_turn && system_prompt && system_prompt[0] != '\0') {
        format = tmpl->first_turn_and_system;
        len = snprintf(NULL, 0, format, system_prompt, prompt);
    } else {
        format = first_turn ? tmpl->first_turn : tmpl->next_turn;
        len = snprintf(NULL, 0, format, prompt);
    }

    if (!format || len < 0) {
        log_msg(stderr, "ERROR: Invalid chat template\n");
        exit(EXIT_FAILURE);
    }

    char *rendered = a_calloc((size_t)len + 1);
    if (!rendered) {
        log_msg(stderr, "ERROR: Failed to allocate rendered chat prompt\n");
        exit(EXIT_FAILURE);
    }

    if (first_turn && system_prompt && system_prompt[0] != '\0') {
        snprintf(rendered, (size_t)len + 1, format, system_prompt, prompt);
    } else {
        snprintf(rendered, (size_t)len + 1, format, prompt);
    }

    *rendered_len = len;
    return rendered;
}

static bool is_chat_stop_token(const model_iface *model_i, int token) {
    if (token == model_i->im_end_id) {
        return true;
    }
    if (model_i->eos_token_id > 0 && token == model_i->eos_token_id) {
        return true;
    }

    return token == 2;
}

void chat_common(model_iface *model_i, Tokenizer *tokenizer, Sampler *sampler,
        char *system_prompt, char *init_prompt, int prompt_n_max, int steps_n_max,
        bool _debug) {
    const chat_template *tmpl = model_i->chat_template;
    if (!tmpl) {
        tmpl = &CHAT_TEMPLATE_CHATML;
    }
    if (!tmpl->first_turn_and_system || !tmpl->first_turn || !tmpl->next_turn) {
        log_msg(stderr, "ERROR: Model supplied an incomplete chat template\n");
        exit(EXIT_FAILURE);
    }

    int rendered_len = 0;
    char *rendered_prompt = NULL;
    int prompt_tokens_n = 0;
    int *prompt_tokens = NULL;
    int user_idx;
    int8_t user_turn = 1;
    int8_t first_turn = 1;
    int next = 0;
    int token;
    int pos = 0;
    long start = 0;
    int generated_tokens = 0;

    char *prompt = (char *)a_calloc((prompt_n_max + 1) * sizeof(char));

    while (pos < steps_n_max) {
        if (user_turn) {
            if (first_turn && (init_prompt != NULL)) {
                strncpy(prompt, init_prompt, prompt_n_max);
                prompt[prompt_n_max] = '\0';
            }
            else {
                log_msg(stdout, "In: ");
                read_msg(prompt, prompt_n_max);
            }
            if (prompt[0] == '\0') {
                continue;
            }

            rendered_prompt = render_chat_turn(tmpl, first_turn,
                    system_prompt, prompt, &rendered_len);

            if (prompt_tokens) {
                free(prompt_tokens);
            }

            prompt_tokens = (int *)a_calloc(((size_t)rendered_len * 4 + 3) * sizeof(int));

            int bos_token = first_turn ? model_i->bos_token_id : 0;
            encode(tokenizer, rendered_prompt, bos_token, 0,
                    prompt_tokens, &prompt_tokens_n);

            free(rendered_prompt);
            rendered_prompt = NULL;

            user_idx = 0;
            user_turn = 0;
            first_turn = 0;
            generated_tokens = 0;
            start = time_in_ms();

            log_msg(stdout, "Out: ");
        }

        if (user_idx < prompt_tokens_n) {
            token = prompt_tokens[user_idx++];
        }
        else {
            token = next;
        }

        float *logits = model_i->forward(model_i->model, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= prompt_tokens_n) {
            if (is_chat_stop_token(model_i, next)) {
                log_msg(stdout, "\n");
                long end = time_in_ms();
                if ((generated_tokens > 0) && ((end - start) > 0)) {
                    log_msg(stderr, "\ntok/s: %.2f\n", generated_tokens / (double)(end - start) * 1000);
                }
                user_turn = 1;
            } else {
                char *piece = decode(tokenizer, next, _debug);
                log_msg(stdout, "%s", piece);
                generated_tokens++;
            }
        }
    }
    log_msg(stdout, "\n");

    if (prompt_tokens) {
        free(prompt_tokens);
    }

    free(prompt);
}

void error_usage(const char *prog_name) {
    log_msg(stderr, "Usage: %s [options]\n", prog_name);
    log_msg(stderr, "Options:\n");
    log_msg(stderr, " -m  | --model <str>:         model path, default: none\n");
    log_msg(stderr, " -t  | --temp <float>:        temperature in [0,inf], default: %f\n", TEMP_DEFAULT);
    log_msg(stderr, " -tp | --top_p <float>:       top-p value in [0,1] default: %f\n", TOP_P_DEFAULT);
    log_msg(stderr, " -k  | --top_k <int>:         top-k value, default: %d\n", TOP_K_DEFAULT);
    log_msg(stderr, " -s  | --seed <int>:          random seed, default: current time\n");
    log_msg(stderr, " -n  | --seq_n <int>:         maximum number of steps, default: model max\n");
    log_msg(stderr, " -pn | --prompt_n <int>:      prompt maximum length, default: %d\n", PROMPT_N_MAX_DEFAULT);
    log_msg(stderr, " -p  | --prompt <str>:        prompt, default: none\n");
    log_msg(stderr, " -pf | --prompt_file <str>:   path to a file containing the initial prompt, default: none\n");
    log_msg(stderr, " -tk | --tokenizer <str>:     path to tokenizer, default: \"tokenizer.bin\"\n");
    log_msg(stderr, " -M  | --mode <str>:          generate|chat, default: chat\n");
    log_msg(stderr, " -sp | --system_prompt <str>: system prompt, default: none\n");
    log_msg(stderr, " -d  | --debug <int>:         enable debug output, default: 0\n");
    log_msg(stderr, " -l  | --log <str>:           path to append all I/O to, default: none\n");
    log_msg(stderr, " -h  | --help:                print this help and exit\n");

    exit(EXIT_FAILURE);
}

int common_main(int argc, char *argv[], model_iface *(*init_fn)(const char *, int), const char *prog_name) {
    char *model_path = NULL;
    float temperature = TEMP_DEFAULT;
    int topk = TOP_K_DEFAULT;
    float topp = TOP_P_DEFAULT;
    unsigned long long rng_seed = 0;
    int seq_n_max = 0;
    int prompt_n_max = PROMPT_N_MAX_DEFAULT;
    char *prompt = NULL;
    char *prompt_file = NULL;
    char *tokenizer_path = "tokenizer.bin";
    char *mode = "chat";
    char *system_prompt = NULL;
    bool _debug = false;


    for (int i = 1; i < argc; i += 2) {
        if (argv[i][0] != '-') {
            error_usage(prog_name);
        }

        if ((! strcmp(argv[i], "-h")) || (! strcmp(argv[i], "--help"))) {
            error_usage(prog_name);
        }

        if ((i + 1) >= argc) {
            error_usage(prog_name);
        }


        if ((! strcmp(argv[i], "-m")) || (! strcmp(argv[i], "--model"))) {
            model_path = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-t")) || (! strcmp(argv[i], "--temp"))) {
            temperature = atof(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-k")) || (! strcmp(argv[i], "--top_k"))) {
            topk = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-tp")) || (! strcmp(argv[i], "--top_p"))) {
            topp = atof(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-s")) || (! strcmp(argv[i], "--seed"))) {
            rng_seed = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-n")) || (! strcmp(argv[i], "--seq_n"))) {
            seq_n_max = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-pn")) || (! strcmp(argv[i], "--prompt_n"))) {
            prompt_n_max = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-p")) || (! strcmp(argv[i], "--prompt"))) {
            prompt = a_calloc(strlen(argv[i + 1]) + 1);
            strcpy(prompt, argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-pf")) || (! strcmp(argv[i], "--prompt_file"))) {
            prompt_file = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-tk")) || (! strcmp(argv[i], "--tokenizer"))) {
            tokenizer_path = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-M")) || (! strcmp(argv[i], "--mode"))) {
            mode = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-sp")) || (! strcmp(argv[i], "--system_prompt"))) {
            system_prompt = a_calloc(strlen(argv[i + 1]) + 1);
            strcpy(system_prompt, argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-d")) || (! strcmp(argv[i], "--debug"))) {
            _debug = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-l")) || (! strcmp(argv[i], "--log"))) {
            log_path = argv[i + 1];
        }
        else {
            error_usage(prog_name);
        }
    }

    if (! model_path) {
        log_msg(stderr, "ERROR: Model path required.\n");
        exit(EXIT_FAILURE);
    }

    if (rng_seed == 0) {
        rng_seed = (unsigned int)time(NULL);
    }
    log_msg(stderr, "INFO: Using seed %lu\n", rng_seed);

    if (prompt_file) {
        if (prompt) {
            free(prompt);
            prompt = NULL;
        }
        FILE *pf = fopen(prompt_file, "r");
        if (!pf) {
            log_msg(stderr, "ERROR: Couldn't open prompt file %s\n", prompt_file);
            exit(EXIT_FAILURE);
        }
        fseek(pf, 0, SEEK_END);
        long f_len = ftell(pf);
        fseek(pf, 0, SEEK_SET);
        
        if (f_len < 0) {
            log_msg(stderr, "ERROR: Failed to determine size of prompt file %s\n", prompt_file);
            fclose(pf);
            exit(EXIT_FAILURE);
        }
        
        prompt = (char *)a_calloc(f_len + 1);
        if (!prompt) {
            log_msg(stderr, "ERROR: Memory allocation failed for prompt file\n");
            fclose(pf);
            exit(EXIT_FAILURE);
        }
        
        size_t read_bytes = fread(prompt, 1, f_len, pf);
        prompt[read_bytes] = '\0';
        fclose(pf);
    }

    model_iface *model_i = init_fn(model_path, seq_n_max);
    if (! model_i) {
        exit(EXIT_FAILURE);
    }

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model_i->vocab_size, model_i->special_tokens);

    Sampler sampler;
    build_sampler(&sampler, model_i->vocab_size, temperature, topk, topp, rng_seed);

    if (! memcmp(mode, "generate", strlen("generate") + 1)) {
        generate_common(model_i, &tokenizer, &sampler, prompt, model_i->seq_n_max);
    }
    else if (! memcmp(mode, "chat", strlen("chat") + 1)) {
        chat_common(model_i, &tokenizer, &sampler, system_prompt, prompt, prompt_n_max, model_i->seq_n_max, _debug);
    }
    else {
        log_msg(stderr, "ERROR: Unknown mode: %s\n", mode);
        error_usage(prog_name);
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);

    model_i->free_model(model_i->model);

    free(model_i);

    return 0;
}

