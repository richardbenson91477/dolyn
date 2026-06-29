#include "dolen_main.h"


static const chat_template CHAT_TEMPLATE_CHATML = {
    ._system_s = "<|im_start|\x3e" "system\n%s<|im_end|\x3e" "\n",
    ._main_s = "<|im_start|\x3e" "user\n%s<|im_end|\x3e" "\n"
            "<|im_start|\x3e" "assistant\n",
    ._end_turn_s = "<|im_end|\x3e" "\n",
};

const model_registry_entry MODEL_REGISTRY[] = {
        {MAGIC_G4, "g4", init_g4},
        {MAGIC_IG4_1, "ig4_1", init_ig4_1},
        {MAGIC_L3, "l3", init_l3},
        {MAGIC_Q3_5, "q3_5", init_q3_5},
        {MAGIC_Q3, "q3", init_q3},
        {MAGIC_Q2, "q2", init_q2},
};

const size_t MODEL_REGISTRY_SIZE = sizeof(MODEL_REGISTRY) / sizeof(MODEL_REGISTRY[0]);


static void generate(model_iface *_model_i, sampler *_sampler, char *_prompt_s, int steps_n_max) {
    if (_prompt_s == NULL) {
        _prompt_s = "";
    }

    int prompt_tokens_n = 0;
    int *_prompt_tokens = (int *)a_calloc((strlen(_prompt_s) * 4 + 3) * sizeof(int));

    encode(_model_i->_tokenizer, _prompt_s, _model_i->_tokenizer->bos_id, 0, _prompt_tokens, &prompt_tokens_n);

    if (prompt_tokens_n < 1) {
        log_msg(stderr, "ERROR: Expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = _prompt_tokens[0];
    int pos = 0;
    while (pos < steps_n_max) {
        float *_logits = _model_i->forward(_model_i->_model, token, pos);

        if (pos < (prompt_tokens_n - 1)) {
            next = _prompt_tokens[pos + 1];
        }
        else {
            next = sample(_sampler, _logits);
        }

        pos++;
        if (next == 1) {
            break;
        }

        char *_piece_s = decode(_model_i->_tokenizer, next);
        log_msg(stdout, "%s", _piece_s);

        token = next;

        if (! start) {
            start = time_in_ms();
        }
    }
    log_msg(stdout, "\n");

    if (pos > 1) {
        long end = time_in_ms();
        log_msg(stdout, "INFO: %f tokens per second.\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(_prompt_tokens);
}

static char *render_chat_turn(const chat_template *_chat_tmpl, bool first_turn_, const char *_system_prompt_s,
        const char *_prompt_s, int *_rendered_len) {
    int len1, len2;

    if (first_turn_ &&
            _system_prompt_s &&
            _system_prompt_s[0]) {
        len1 = snprintf(NULL, 0, _chat_tmpl->_system_s, _system_prompt_s);
        len2 = snprintf(NULL, 0, _chat_tmpl->_main_s, _prompt_s);
    }
    else {
        len1 = snprintf(NULL, 0, _chat_tmpl->_end_turn_s);
        len2 = snprintf(NULL, 0, _chat_tmpl->_main_s, _prompt_s);
    }

    char *_rendered_s = a_calloc(len1 + len2 + 1);
    if (! _rendered_s) {
        log_msg(stderr, "ERROR: Failed to allocate rendered chat prompt\n");
        exit(EXIT_FAILURE);
    }

    if (first_turn_ &&
            _system_prompt_s &&
            _system_prompt_s[0]) {
        snprintf(_rendered_s, len1 + 1, _chat_tmpl->_system_s, _system_prompt_s);
        snprintf(_rendered_s + len1, len2 + 1, _chat_tmpl->_main_s, _prompt_s);
    }
    else {
        snprintf(_rendered_s, len1 + 1, _chat_tmpl->_end_turn_s);
        snprintf(_rendered_s + len1, len2 + 1, _chat_tmpl->_main_s, _prompt_s);
    }

    *_rendered_len = len1 + len2;
    return _rendered_s;
}

static bool is_chat_stop_token(const model_iface *_model_i, int token) {
    if (token == _model_i->_tokenizer->im_end_id) {
        return true;
    }
    if ((_model_i->_tokenizer->eos_id > 0) &&
            (token == _model_i->_tokenizer->eos_id)) {
        return true;
    }

    return token == 2;
}

static void chat(model_iface *_model_i, sampler *_sampler, char *_system_prompt_s, char *_init_prompt_s,
        int prompt_n_max, int steps_n_max) {
    const chat_template *_chat_tmpl = _model_i->_chat_template;
    if (! _chat_tmpl) {
        _chat_tmpl = &CHAT_TEMPLATE_CHATML;
    }
    if ((! _chat_tmpl->_system_s) ||
            (! _chat_tmpl->_main_s) ||
            (! _chat_tmpl->_end_turn_s)) {
        log_msg(stderr, "ERROR: Model supplied an incomplete chat template\n");
        exit(EXIT_FAILURE);
    }

    int _rendered_len = 0;
    char *_rendered_prompt_s = NULL;
    int prompt_tokens_n = 0;
    int *_prompt_tokens = NULL;
    int user_idx;
    bool user_turn_ = true;
    bool first_turn_ = true;
    int next = 0;
    int token;
    int pos = 0;
    long start = 0;
    int generated_tokens = 0;

    char *_prompt_s = (char *)a_calloc((prompt_n_max + 1) * sizeof(char));

    while (pos < steps_n_max) {
        if (user_turn_) {
            if (first_turn_ &&
                    _init_prompt_s) {
                strncpy(_prompt_s, _init_prompt_s, prompt_n_max);
                _prompt_s[prompt_n_max] = '\0';
            }
            else {
                log_msg(stdout, "In: ");
                read_msg(_prompt_s, prompt_n_max);
            }
            if (_prompt_s[0] == '\0') {
                continue;
            }

            _rendered_prompt_s = render_chat_turn(_chat_tmpl, first_turn_, _system_prompt_s, _prompt_s, &_rendered_len);

            if (_prompt_tokens) {
                free(_prompt_tokens);
            }

            _prompt_tokens = (int *)a_calloc(((size_t)_rendered_len * 4 + 3) * sizeof(int));

            int bos_token = first_turn_ ? _model_i->_tokenizer->bos_id : 0;
            encode(_model_i->_tokenizer, _rendered_prompt_s, bos_token, 0, _prompt_tokens, &prompt_tokens_n);

            free(_rendered_prompt_s);
            _rendered_prompt_s = NULL;

            user_idx = 0;
            user_turn_ = false;
            first_turn_ = false;
            generated_tokens = 0;
            start = time_in_ms();

            log_msg(stdout, "Out: ");
        }

        if (user_idx < prompt_tokens_n) {
            token = _prompt_tokens[user_idx++];
        }
        else {
            token = next;
        }

        float *_logits = _model_i->forward(_model_i->_model, token, pos);
        next = sample(_sampler, _logits);
        pos++;

        if (user_idx >= prompt_tokens_n) {
            if (is_chat_stop_token(_model_i, next)) {
                log_msg(stdout, "\n");
                long end = time_in_ms();
                if ((generated_tokens > 0) &&
                        ((end - start) > 0)) {
                    log_msg(stdout, "\ntok/s: %.2f\n", generated_tokens / (double)(end - start) * 1000);
                }
                user_turn_ = 1;
            }
            else {
                char *_piece_s = decode(_model_i->_tokenizer, next);
                log_msg(stdout, "%s", _piece_s);
                generated_tokens++;
            }
        }
    }
    log_msg(stdout, "\n");

    if (_prompt_tokens) {
        free(_prompt_tokens);
    }

    free(_prompt_s);
}

static uint64_t peek_model_magic(const char *_path_s) {
    FILE *_file = fopen(_path_s, "rb");
    if (! _file) {
        log_msg(stderr, "ERROR: can't open \"%s\"\n", _path_s);
        return 0;
    }

    uint64_t magic = 0;
    fread(&magic, sizeof(magic), 1, _file);
    fclose(_file);

    return magic;
}

static void error_usage(const char *_argv0) {
    log_msg(stdout, "Usage: %s [options]\n", _argv0);
    log_msg(stdout, "Options:\n");
    log_msg(stdout, " -m  | --model <str>:         model path, default: none\n");
    log_msg(stdout, " -t  | --temp <float>:        temperature in [0,inf], default: %f\n", TEMP_DEFAULT);
    log_msg(stdout, " -tp | --top_p <float>:       top-p value in [0,1] default: %f\n", TOP_P_DEFAULT);
    log_msg(stdout, " -k  | --top_k <int>:         top-k value, default: %d\n", TOP_K_DEFAULT);
    log_msg(stdout, " -s  | --seed <int>:          random seed, default: current time\n");
    log_msg(stdout, " -n  | --seq_n <int>:         maximum number of steps, default: model max\n");
    log_msg(stdout, " -pn | --prompt_n <int>:      prompt maximum length, default: %d\n", PROMPT_N_MAX_DEFAULT);
    log_msg(stdout, " -p  | --prompt <str>:        prompt, default: none\n");
    log_msg(stdout, " -pf | --prompt_file <str>:   path to a file containing the initial prompt, default: none\n");
    log_msg(stdout, " -M  | --mode <str>:          generate|chat, default: chat\n");
    log_msg(stdout, " -sp | --system_prompt <str>: system prompt, default: none\n");
    log_msg(stdout, " -l  | --log <str>:           path to append all I/O to, default: none\n");
    log_msg(stdout, " -h  | --help:                print this help and exit\n");
    log_msg(stdout, " -th | --think:               enable think-mode chat template, default: disabled\n");

    exit(EXIT_FAILURE);
}

int main(int argc, char *__argv[]) {
    char *_model_path_s = NULL;
    float temp = TEMP_DEFAULT;
    int top_k = TOP_K_DEFAULT;
    float top_p = TOP_P_DEFAULT;
    unsigned long long rng_seed = 0;
    int seq_n_max = 0;
    int prompt_n_max = PROMPT_N_MAX_DEFAULT;
    char *_prompt_s = NULL;
    char *_prompt_file = NULL;
    char *_mode_s = "chat";
    char *_system_prompt_s = NULL;
    bool think_ = false;

    for (int i = 1; i < argc;) {
        if (__argv[i][0] != '-') {
            error_usage(__argv[0]);
        }

        if ((! strcmp(__argv[i], "-h")) ||
                (! strcmp(__argv[i], "--help"))) {
            error_usage(__argv[0]);
        }
        else if ((! strcmp(__argv[i], "-th")) ||
                (! strcmp(__argv[i], "--think"))) {
            think_ = true;
            i += 1;
            continue;
        }

        if ((i + 1) >= argc) {
            error_usage(__argv[0]);
        }

        if ((! strcmp(__argv[i], "-m")) ||
                (! strcmp(__argv[i], "--model"))) {
            _model_path_s = __argv[i + 1];
        }
        else if ((! strcmp(__argv[i], "-t")) ||
                (! strcmp(__argv[i], "--temp"))) {
            temp = atof(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-k")) ||
                (! strcmp(__argv[i], "--top_k"))) {
            top_k = atoi(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-tp")) ||
                (! strcmp(__argv[i], "--top_p"))) {
            top_p = atof(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-s")) ||
                (! strcmp(__argv[i], "--seed"))) {
            rng_seed = atoi(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-n")) ||
                (! strcmp(__argv[i], "--seq_n"))) {
            seq_n_max = atoi(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-pn")) ||
                (! strcmp(__argv[i], "--prompt_n"))) {
            prompt_n_max = atoi(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-p")) ||
                (! strcmp(__argv[i], "--prompt"))) {
            size_t prompt_len = strlen(__argv[i + 1]);
            if (prompt_len > 0) {
                _prompt_s = a_calloc(prompt_len + 1);
                strncpy(_prompt_s, __argv[i + 1], prompt_len + 1);
            }
        }
        else if ((! strcmp(__argv[i], "-pf")) ||
                (! strcmp(__argv[i], "--prompt_file"))) {
            _prompt_file = __argv[i + 1];
        }
        else if ((! strcmp(__argv[i], "-M")) ||
                (! strcmp(__argv[i], "--mode"))) {
            _mode_s = __argv[i + 1];
        }
        else if ((! strcmp(__argv[i], "-sp")) ||
                (! strcmp(__argv[i], "--system_prompt"))) {
            int system_prompt_len = strlen(__argv[i + 1]);
            if (system_prompt_len > 0) {
                _system_prompt_s = a_calloc(system_prompt_len + 1);
                strncpy(_system_prompt_s, __argv[i + 1], system_prompt_len + 1);
            }
        }
        else if ((! strcmp(__argv[i], "-l")) ||
                (! strcmp(__argv[i], "--log"))) {
            _log_path = __argv[i + 1];
        }
        else {
            error_usage(__argv[0]);
        }

        i += 2;
    }

    if (! _model_path_s) {
        log_msg(stderr, "ERROR: Model path required.\n");
        exit(EXIT_FAILURE);
    }

    if (! rng_seed) {
        rng_seed = (unsigned int)time(NULL);
    }
    log_msg(stdout, "INFO: Using seed %lu\n", rng_seed);

    if (_prompt_file) {
        if (_prompt_s) {
            free(_prompt_s);
            _prompt_s = NULL;
        }
        FILE *_file = fopen(_prompt_file, "r");
        if (! _file) {
            log_msg(stderr, "ERROR: Couldn't open prompt file %s\n", _prompt_file);
            exit(EXIT_FAILURE);
        }
        fseek(_file, 0, SEEK_END);
        long f_len = ftell(_file);
        fseek(_file, 0, SEEK_SET);

        if (f_len < 0) {
            log_msg(stderr, "ERROR: Failed to determine size of prompt file %s\n", _prompt_file);
            fclose(_file);
            exit(EXIT_FAILURE);
        }

        _prompt_s = (char *)a_calloc(f_len + 1);
        if (! _prompt_s) {
            log_msg(stderr, "ERROR: Memory allocation failed for prompt file\n");
            fclose(_file);
            exit(EXIT_FAILURE);
        }

        size_t read_bytes = fread(_prompt_s, 1, f_len, _file);
        _prompt_s[read_bytes] = '\0';
        fclose(_file);
    }

    uint64_t magic = peek_model_magic(_model_path_s);
    if (! magic) {
        log_msg(stderr, "ERROR: peek_model_magic(\"%s\") returned 0\n", _model_path_s);
        exit(EXIT_FAILURE);
    }

    const char *_model_type_s = "unknown";
    model_init_fn init_fn = NULL;

    for (size_t i = 0; i < MODEL_REGISTRY_SIZE; i++) {
        if (MODEL_REGISTRY[i].magic == magic) {
            init_fn = MODEL_REGISTRY[i].init_fn;
            _model_type_s = MODEL_REGISTRY[i]._name_s;
            break;
        }
    }

    if (! init_fn) {
        log_msg(stderr, "ERROR: Unknown model magic number 0x%llx in \"%s\"\n", magic, _model_path_s);
        exit(EXIT_FAILURE);
    }
    
    log_msg(stdout, "INFO: Detected model type: \"%s\"\n", _model_type_s);

    model_iface *_model_i = init_fn(_model_path_s, seq_n_max, think_);
    if (! _model_i) {
        exit(EXIT_FAILURE);
    }


    sampler _sampler;
    build_sampler(&_sampler, _model_i->_tokenizer->vocab_size, temp, top_k, top_p, rng_seed);

    if (! memcmp(_mode_s, "generate", strlen("generate") + 1)) {
        generate(_model_i, &_sampler, _prompt_s, _model_i->seq_n_max);
    }
    else if (! memcmp(_mode_s, "chat", strlen("chat") + 1)) {
        chat(_model_i, &_sampler, _system_prompt_s, _prompt_s, prompt_n_max, _model_i->seq_n_max);
    }
    else {
        log_msg(stderr, "ERROR: Unknown mode: %s\n", _mode_s);
        error_usage(__argv[0]);
    }

    free_sampler(&_sampler);

    _model_i->free_model(_model_i->_model);

    free(_model_i);

    return 0;
}

