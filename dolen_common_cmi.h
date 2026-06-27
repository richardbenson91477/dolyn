#ifndef DOLEN_COMMON_CMI_H
#define DOLEN_COMMON_CMI_H

#include "dolen_common_tokenizer.h"
#include "dolen_common_sampler.h"


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

typedef model_iface *(*model_init_fn)(const char *model_path, int seq_n_max, bool think_);


#endif // DOLEN_COMMON_CMI_H

