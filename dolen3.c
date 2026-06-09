#include "dolen3_common.h"


// Old config format (v1 binary files) \u2014 without rms_norm_eps
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int shared_classifier;
    float rope_theta;
    float rope_scaling_factor;
} config_qwen3_v1;

int load_quantized_qwen3(const char *filepath, Qwen3 *model_qwen3, int seq_n_max) {
    FILE *f = fopen(filepath, "rb");
    if (! f) {
        fprintf(stderr, "ERROR: Failed to open %s for reading\n", filepath);
        return -1;
    }
    
    memset(model_qwen3, 0, sizeof(Qwen3));

    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 || fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to read header from %s\n", filepath);
        fclose(f);
        return -1;
    }
    
    if (magic != 0x30335751) { // 'QW30'
        fprintf(stderr, "ERROR: Invalid magic number in %s\n", filepath);
        fclose(f);
        return -1;
    }
    
    if (version != 2) {
        fprintf(stderr, "ERROR: Unsupported version %d in %s\n", version, filepath);
        fclose(f);
        return -1;
    }

    config_qwen3 *p = &model_qwen3->config;

    if (fread(p, sizeof(config_qwen3), 1, f) != 1) {
        fprintf(stderr, "ERROR: Failed to read config from %s\n", filepath);
        fclose(f);
        return -1;
    }

    weights_qwen3 *w = &model_qwen3->weights;

    if (seq_n_max != 0) {
        p->seq_len = seq_n_max;
    }
    
    // Allocate memory for non-quantized tensors
    w->rms_att_weight = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_ffn_weight = (float *)a_calloc((size_t)p->n_layers * p->dim * sizeof(float));
    w->rms_final_weight = (float *)a_calloc((size_t)p->dim * sizeof(float));
    w->q_norm_weights = (float *)a_calloc((size_t)p->n_layers * p->head_dim * sizeof(float));
    w->k_norm_weights = (float *)a_calloc((size_t)p->n_layers * p->head_dim * sizeof(float));
    
    if (!w->rms_att_weight || !w->rms_ffn_weight || !w->rms_final_weight || 
        !w->q_norm_weights || !w->k_norm_weights) {
        fprintf(stderr, "ERROR: Failed to allocate memory for weights\n");
        fclose(f);
        return -1;
    }
    
    // Read quantized token embedding table
    read_qt(f, &w->token_embedding_table);
    
    if (fread(w->rms_att_weight, sizeof(float), (size_t)p->n_layers * p->dim, f) != (size_t)p->n_layers * p->dim) {
        fprintf(stderr, "ERROR: Failed to read rms_att_weight\n");
        fclose(f);
        return -1;
    }
    
    if (fread(w->rms_ffn_weight, sizeof(float), (size_t)p->n_layers * p->dim, f) != (size_t)p->n_layers * p->dim) {
        fprintf(stderr, "ERROR: Failed to read rms_ffn_weight\n");
        fclose(f);
        return -1;
    }
    
    if (fread(w->rms_final_weight, sizeof(float), (size_t)p->dim, f) != (size_t)p->dim) {
        fprintf(stderr, "ERROR: Failed to read rms_final_weight\n");
        fclose(f);
        return -1;
    }
    
    if (fread(w->q_norm_weights, sizeof(float), (size_t)p->n_layers * p->head_dim, f) != (size_t)p->n_layers * p->head_dim) {
        fprintf(stderr, "ERROR: Failed to read q_norm_weights\n");
        fclose(f);
        return -1;
    }
    
    if (fread(w->k_norm_weights, sizeof(float), (size_t)p->n_layers * p->head_dim, f) != (size_t)p->n_layers * p->head_dim) {
        fprintf(stderr, "ERROR: Failed to read k_norm_weights\n");
        fclose(f);
        return -1;
    }
    
    if (! p->shared_classifier) {
        read_qt(f, &w->wcls);
    } else {
        w->wcls = w->token_embedding_table;
    }
    
    // Allocate memory for quantized tensors
    w->wq = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wk = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wv = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->wo = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->w1 = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->w2 = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    w->w3 = (qtensor *)a_calloc((size_t)p->n_layers * sizeof(qtensor));
    
    if (!w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3) {
        fprintf(stderr, "ERROR: Failed to allocate memory for quantized tensors\n");
        fclose(f);
        return -1;
    }
    
    // Read quantized tensors
    for (int l = 0; l < p->n_layers; l++) {
        read_qt(f, &w->wq[l]);
        read_qt(f, &w->wk[l]);
        read_qt(f, &w->wv[l]);
        read_qt(f, &w->wo[l]);
        read_qt(f, &w->w1[l]);
        read_qt(f, &w->w2[l]);
        read_qt(f, &w->w3[l]);
    }
    
    fclose(f);
    fprintf(stderr, "INFO: Quantized model loaded from %s\n", filepath);

    alloc_state_qwen3(&(model_qwen3->state), &(model_qwen3->config));

    return 0;
}

void rmsnorm(float *o, float *x, float *weight, int size, float eps) {
    float ss = 0.0f;

    #pragma omp simd reduction(+:ss)
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }

    ss = 1.0f / sqrtf((ss / size) + eps);

    #pragma omp simd
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

float *forward_qwen3(Qwen3 *model_qwen3, int token, int pos) {
    config_qwen3 *p = &model_qwen3->config;
    weights_qwen3 *w = &model_qwen3->weights;
    state_qwen3 *s = &model_qwen3->state;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int all_heads_dim = p->n_heads * p->head_dim;
    float eps = p->rms_norm_eps;

    dequantize_row(s->x, &w->token_embedding_table, token);

    for (int l = 0; l < p->n_layers; l++) {
        long long loff = l * p->seq_len * kv_dim;

        rmsnorm(s->xb, s->x, w->rms_att_weight + l * p->dim, p->dim, eps);

        // Quantize input for attention projections
        quantize_vec(&s->xq, s->xb, p->dim);
        
        matmul_qq(s->q, &s->xq, &w->wq[l]);
        matmul_qq(s->k, &s->xq, &w->wk[l]);
        matmul_qq(s->v, &s->xq, &w->wv[l]);

        // Pre-computed RoPE cache
        int rotary_half = p->head_dim / 2;

        if (rotary_half > 0 && s->cos_cache != NULL) {
            float *cos_row = s->cos_cache + pos * rotary_half;
            float *sin_row = s->sin_cache + pos * rotary_half;

            #pragma omp parallel for
            for (int h = 0; h < p->n_heads; h++) {
                float *q_ptr = s->q + h * p->head_dim;

                rmsnorm(q_ptr, q_ptr, w->q_norm_weights + l * p->head_dim, p->head_dim, eps);

                for (int j = 0; j < rotary_half; j++) {
                    float c = cos_row[j], sn = sin_row[j];
                    float x_val = q_ptr[j], y_val = q_ptr[j + rotary_half];
                    q_ptr[j] = x_val * c - y_val * sn;
                    q_ptr[j + rotary_half] = x_val * sn + y_val * c;
                }
            }

            #pragma omp parallel for
            for (int h = 0; h < p->n_kv_heads; h++) {
                float *k_ptr = s->k + h * p->head_dim;

                rmsnorm(k_ptr, k_ptr, w->k_norm_weights + l * p->head_dim, p->head_dim, eps);

                for (int j = 0; j < rotary_half; j++) {
                    float c = cos_row[j], sn = sin_row[j];
                    float x_val = k_ptr[j], y_val = k_ptr[j + rotary_half];
                    k_ptr[j] = x_val * c - y_val * sn;
                    k_ptr[j + rotary_half] = x_val * sn + y_val * c;
                }
            }
        } else {
            // Fallback: no cache available (shouldn't happen, but be safe)
            #pragma omp parallel for
            for (int h = 0; h < p->n_heads; h++) {
                float *q_ptr = s->q + h * p->head_dim;
                rmsnorm(q_ptr, q_ptr, w->q_norm_weights + l * p->head_dim, p->head_dim, eps);
                for (int j = 0; j < rotary_half; j++) {
                    float freq = 1.0f / powf(p->rope_theta, (float)j / rotary_half);
                    float scaled_pos = pos / p->rope_scaling_factor;
                    float cos_freq = cosf(scaled_pos * freq);
                    float sin_freq = sinf(scaled_pos * freq);
                    float x_val = q_ptr[j], y_val = q_ptr[j + rotary_half];
                    q_ptr[j] = x_val * cos_freq - y_val * sin_freq;
                    q_ptr[j + rotary_half] = x_val * sin_freq + y_val * cos_freq;
                }
            }

            #pragma omp parallel for
            for (int h = 0; h < p->n_kv_heads; h++) {
                float *k_ptr = s->k + h * p->head_dim;
                rmsnorm(k_ptr, k_ptr, w->k_norm_weights + l * p->head_dim, p->head_dim, eps);
                for (int j = 0; j < rotary_half; j++) {
                    float freq = 1.0f / powf(p->rope_theta, (float)j / rotary_half);
                    float scaled_pos = pos / p->rope_scaling_factor;
                    float cos_freq = cosf(scaled_pos * freq);
                    float sin_freq = sinf(scaled_pos * freq);
                    float x_val = k_ptr[j], y_val = k_ptr[j + rotary_half];
                    k_ptr[j] = x_val * cos_freq - y_val * sin_freq;
                    k_ptr[j + rotary_half] = x_val * sin_freq + y_val * cos_freq;
                }
            }
        }

        // Store k and v in cache
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        #pragma omp parallel for
        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                float score = 0.0f;

                #pragma omp simd reduction(+:score)
                for (int i = 0; i < p->head_dim; i++) {
                    score += q[i] * k[i];
                }
                att[t] = score / sqrtf(p->head_dim);
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * p->head_dim;
            memset(xb, 0, p->head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                float a = att[t];

                #pragma omp simd
                for (int i = 0; i < p->head_dim; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // Quantize attention output for output projection
        quantize_vec(&s->xq, s->xb, all_heads_dim);

        matmul_qq(s->xb, &s->xq, &w->wo[l]);

        for (int i = 0; i < p->dim; i++) {
            s->x[i] += s->xb[i];
        }

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * p->dim, p->dim, eps);

        // Quantize for MLP input
        quantize_vec(&s->xq, s->xb, p->dim);

        matmul_qq(s->hb, &s->xq, &w->w1[l]);
        matmul_qq(s->hb2, &s->xq, &w->w3[l]);

        #pragma omp parallel for
        for (int i = 0; i < p->hidden_dim; i++) {
            s->hb[i] = s->hb[i] * (1.0f / (1.0f + expf(-s->hb[i]))) * s->hb2[i];
        }

        // Quantize MLP output for down projection
        quantize_vec(&s->hq, s->hb, p->hidden_dim);

        matmul_qq(s->xb, &s->hq, &w->w2[l]);

        for (int i = 0; i < p->dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim, eps);

    matmul_qt(s->logits, s->x, &w->wcls);

    return s->logits;
}

static float *forward_qwen3_wrap(void *model, int token, int pos) {
    return forward_qwen3((Qwen3 *)model, token, pos);
}

static void free_qwen3_wrap(void *model) {
    free_qwen3((Qwen3 *)model);
    free(model);
}

static model_iface *init_qwen3(const char *model_path, int seq_n_max) {
    Qwen3 *model = a_calloc(1 * sizeof(Qwen3));

    if (load_quantized_qwen3(model_path, model, seq_n_max) != 0) {
        free_qwen3(model);
        free(model);
        return NULL;
    }

    model_iface *model_i = a_calloc(sizeof(model_iface));
    *model_i = (model_iface) {
        .model = model,
        .forward = forward_qwen3_wrap,
        .free_model = free_qwen3_wrap,
        .seq_n_max = (seq_n_max != 0) ? seq_n_max : model->config.seq_len,
        .vocab_size = model->config.vocab_size,
        .special_tokens = special_tokens_qwen3,
        .im_end_id = 151645,
    };
    return model_i;
}

int main(int argc, char *argv[]) {
    return common_main(argc, argv, init_qwen3, "dolen3");
}

