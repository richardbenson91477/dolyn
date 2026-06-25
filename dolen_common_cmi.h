#ifndef DOLEN_COMMON_CMI_H
#define DOLEN_COMMON_CMI_H

#include "dolen_common_tokenizer.h"
#include "dolen_common_sampler.h"


#define PROMPT_N_MAX_DEFAULT 32768
#define TEMP_DEFAULT 0.2
#define TOP_P_DEFAULT 0.95
#define TOP_K_DEFAULT 40


typedef struct {
    const char *system;
    const char *main;
    const char *end_turn;
} chat_template;

typedef struct {
    void *model;
    float *(*forward)(void *model, int token, int pos);
    void (*free_model)(void *model);
    int seq_n_max;
    const chat_template *chat_template;
    Tokenizer *tokenizer;
} model_iface;


void generate_common(model_iface *model_i, Sampler *sampler, char *prompt, int steps_n_max);

void chat_common(model_iface *model_i, Sampler *sampler, char *system_prompt, char *init_prompt,
        int prompt_n_max, int steps_n_max, bool debug_);

void error_usage(const char *prog_name);

int common_main(int argc, char *argv[], model_iface *(*init_fn)(const char *model_path, int seq_n_max, bool think_),
        const char *prog_name);


#endif // DOLEN_COMMON_CMI_H

