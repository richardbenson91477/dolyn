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
};

const size_t MODEL_REGISTRY_SIZE = sizeof(MODEL_REGISTRY) / sizeof(MODEL_REGISTRY[0]);


static void generate(model_iface *_model_i, sampler *_sampler, char *_prompt_s, int steps_n_max) {
    if (_prompt_s == NULL) {
        _prompt_s = "";
    }

    int prompt_tokens_n = 0;
    int *prompt_tokens = (int *)a_calloc((strlen(_prompt_s) * 4 + 3) * sizeof(int));

    encode(_model_i->_tokenizer, _prompt_s, _model_i->_tokenizer->bos_id, 0, prompt_tokens, &prompt_tokens_n);

    if (prompt_tokens_n < 1) {
        log_msg(stderr, "ERROR: Expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps_n_max) {
        float *logits = _model_i->forward(_model_i->_model, token, pos);

        if (pos < (prompt_tokens_n - 1)) {
            next = prompt_tokens[pos + 1];
        }
        else {
            next = sample(_sampler, logits);
        }

        pos++;
        if (next == 1) {
            break;
        }

        char *piece = decode(_model_i->_tokenizer, next, false);
        log_msg(stdout, "%s", piece);

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

    free(prompt_tokens);
}

static char *render_chat_turn(const chat_template *chat_tmpl, bool first_turn_, const char *_system_prompt_s,
        const char *_prompt_s, int *rendered_len) {
    const char *format = NULL;
    int len1, len2;

    if (first_turn_ &&
            _system_prompt_s &&
            _system_prompt_s[0]) {
        len1 = snprintf(NULL, 0, chat_tmpl->_system_s, _system_prompt_s);
        len2 = snprintf(NULL, 0, chat_tmpl->_main_s, _prompt_s);
    }
    else {
        len1 = snprintf(NULL, 0, chat_tmpl->_end_turn_s);
        len2 = snprintf(NULL, 0, chat_tmpl->_main_s, _prompt_s);
    }

    char *rendered = a_calloc(len1 + len2 + 1);
    if (! rendered) {
        log_msg(stderr, "ERROR: Failed to allocate rendered chat prompt\n");
        exit(EXIT_FAILURE);
    }

    if (first_turn_ &&
            _system_prompt_s &&
            _system_prompt_s[0]) {
        snprintf(rendered, len1 + 1, chat_tmpl->_system_s, _system_prompt_s);
        snprintf(rendered + len1, len2 + 1, chat_tmpl->_main_s, _prompt_s);
    }
    else {
        snprintf(rendered, len1 + 1, chat_tmpl->_end_turn_s);
        snprintf(rendered + len1, len2 + 1, chat_tmpl->_main_s, _prompt_s);
    }

    *rendered_len = len1 + len2;
    return rendered;
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

static void chat(model_iface *_model_i, sampler *_sampler, char *_system_prompt_s, char *init_prompt,
        int prompt_n_max, int steps_n_max, bool debug_) {
    const chat_template *chat_tmpl = _model_i->_chat_template;
    if (! chat_tmpl) {
        chat_tmpl = &CHAT_TEMPLATE_CHATML;
    }
    if ((! chat_tmpl->_system_s) ||
            (! chat_tmpl->_main_s) ||
            (! chat_tmpl->_end_turn_s)) {
        log_msg(stderr, "ERROR: Model supplied an incomplete chat template\n");
        exit(EXIT_FAILURE);
    }

    int rendered_len = 0;
    char *rendered_prompt = NULL;
    int prompt_tokens_n = 0;
    int *prompt_tokens = NULL;
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
                    init_prompt) {
                strncpy(_prompt_s, init_prompt, prompt_n_max);
                _prompt_s[prompt_n_max] = '\0';
            }
            else {
                log_msg(stdout, "In: ");
                read_msg(_prompt_s, prompt_n_max);
            }
            if (_prompt_s[0] == '\0') {
                continue;
            }

            rendered_prompt = render_chat_turn(chat_tmpl, first_turn_, _system_prompt_s, _prompt_s, &rendered_len);

            if (prompt_tokens) {
                free(prompt_tokens);
            }

            prompt_tokens = (int *)a_calloc(((size_t)rendered_len * 4 + 3) * sizeof(int));

            int bos_token = first_turn_ ? _model_i->_tokenizer->bos_id : 0;
            encode(_model_i->_tokenizer, rendered_prompt, bos_token, 0, prompt_tokens, &prompt_tokens_n);

            free(rendered_prompt);
            rendered_prompt = NULL;

            user_idx = 0;
            user_turn_ = false;
            first_turn_ = false;
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

        float *logits = _model_i->forward(_model_i->_model, token, pos);
        next = sample(_sampler, logits);
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
                char *piece = decode(_model_i->_tokenizer, next, debug_);
                log_msg(stdout, "%s", piece);
                generated_tokens++;
            }
        }
    }
    log_msg(stdout, "\n");

    if (prompt_tokens) {
        free(prompt_tokens);
    }

    free(_prompt_s);
}

static uint64_t peek_model_magic(const char *_path_s) {
    FILE *f = fopen(_path_s, "rb");
    if (! f) {
        return 0;
    }

    uint64_t magic = 0;
    fread(&magic, sizeof(magic), 1, f);
    fclose(f);

    return magic;
}

static void error_usage(const char *argv0) {
    log_msg(stdout, "Usage: %s [options]\n", argv0);
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
    log_msg(stdout, " -d  | --debug:               enable debug output, default: disabled\n");
    log_msg(stdout, " -l  | --log <str>:           path to append all I/O to, default: none\n");
    log_msg(stdout, " -h  | --help:                print this help and exit\n");
    log_msg(stdout, " -th | --think:               enable think-mode chat template, default: disabled\n");

    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *_model_path_s = NULL;
    float temp = TEMP_DEFAULT;
    int top_k = TOP_K_DEFAULT;
    float top_p = TOP_P_DEFAULT;
    unsigned long long rng_seed = 0;
    int seq_n_max = 0;
    int prompt_n_max = PROMPT_N_MAX_DEFAULT;
    char *_prompt_s = NULL;
    char *_prompt_file = NULL;
    char *mode = "chat";
    char *_system_prompt_s = NULL;
    bool debug_ = false;
    bool think_ = false;

    for (int i = 1; i < argc;) {
        if (argv[i][0] != '-') {
            error_usage(argv[0]);
        }

        if ((! strcmp(argv[i], "-h")) ||
                (! strcmp(argv[i], "--help"))) {
            error_usage(argv[0]);
        }
        else if ((! strcmp(argv[i], "-th")) ||
                (! strcmp(argv[i], "--think"))) {
            think_ = true;
            i += 1;
            continue;
        }
        else if ((! strcmp(argv[i], "-d")) ||
                (! strcmp(argv[i], "--debug"))) {
            debug_ = true;
            i += 1;
            continue;
        }

        if ((i + 1) >= argc) {
            error_usage(argv[0]);
        }

        if ((! strcmp(argv[i], "-m")) ||
                (! strcmp(argv[i], "--model"))) {
            _model_path_s = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-t")) ||
                (! strcmp(argv[i], "--temp"))) {
            temp = atof(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-k")) ||
                (! strcmp(argv[i], "--top_k"))) {
            top_k = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-tp")) ||
                (! strcmp(argv[i], "--top_p"))) {
            top_p = atof(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-s")) ||
                (! strcmp(argv[i], "--seed"))) {
            rng_seed = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-n")) ||
                (! strcmp(argv[i], "--seq_n"))) {
            seq_n_max = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-pn")) ||
                (! strcmp(argv[i], "--prompt_n"))) {
            prompt_n_max = atoi(argv[i + 1]);
        }
        else if ((! strcmp(argv[i], "-p")) ||
                (! strcmp(argv[i], "--prompt"))) {
            size_t prompt_len = strlen(argv[i + 1]);
            if (prompt_len > 0) {
                _prompt_s = a_calloc(prompt_len + 1);
                strncpy(_prompt_s, argv[i + 1], prompt_len + 1);
            }
        }
        else if ((! strcmp(argv[i], "-pf")) ||
                (! strcmp(argv[i], "--prompt_file"))) {
            _prompt_file = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-M")) ||
                (! strcmp(argv[i], "--mode"))) {
            mode = argv[i + 1];
        }
        else if ((! strcmp(argv[i], "-sp")) ||
                (! strcmp(argv[i], "--system_prompt"))) {
            int system_prompt_len = strlen(argv[i + 1]);
            if (system_prompt_len > 0) {
                _system_prompt_s = a_calloc(system_prompt_len + 1);
                strncpy(_system_prompt_s, argv[i + 1], system_prompt_len + 1);
            }
        }
        else if ((! strcmp(argv[i], "-l")) ||
                (! strcmp(argv[i], "--log"))) {
            _log_path = argv[i + 1];
        }
        else {
            error_usage(argv[0]);
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
        FILE *pf = fopen(_prompt_file, "r");
        if (! pf) {
            log_msg(stderr, "ERROR: Couldn't open prompt file %s\n", _prompt_file);
            exit(EXIT_FAILURE);
        }
        fseek(pf, 0, SEEK_END);
        long f_len = ftell(pf);
        fseek(pf, 0, SEEK_SET);

        if (f_len < 0) {
            log_msg(stderr, "ERROR: Failed to determine size of prompt file %s\n", _prompt_file);
            fclose(pf);
            exit(EXIT_FAILURE);
        }

        _prompt_s = (char *)a_calloc(f_len + 1);
        if (! _prompt_s) {
            log_msg(stderr, "ERROR: Memory allocation failed for prompt file\n");
            fclose(pf);
            exit(EXIT_FAILURE);
        }

        size_t read_bytes = fread(_prompt_s, 1, f_len, pf);
        _prompt_s[read_bytes] = '\0';
        fclose(pf);
    }

    uint64_t magic = peek_model_magic(_model_path_s);

    const char *model_type_name = "unknown";
    model_init_fn init_fn = NULL;

    for (size_t i = 0; i < MODEL_REGISTRY_SIZE; i++) {
        if (MODEL_REGISTRY[i].magic == magic) {
            init_fn = MODEL_REGISTRY[i].init_fn;
            model_type_name = MODEL_REGISTRY[i]._name_s;
            break;
        }
    }

    if (! init_fn) {
        log_msg(stderr, "ERROR: Unknown model magic number 0x%llx in \"%s\"\n", magic, _model_path_s);
        exit(EXIT_FAILURE);
    }
    
    log_msg(stdout, "INFO: Detected model type: \"%s\"\n", model_type_name);

    model_iface *_model_i = init_fn(_model_path_s, seq_n_max, think_);
    if (! _model_i) {
        exit(EXIT_FAILURE);
    }


    sampler _sampler;
    build_sampler(&_sampler, _model_i->_tokenizer->vocab_size, temp, top_k, top_p, rng_seed);

    if (! memcmp(mode, "generate", strlen("generate") + 1)) {
        generate(_model_i, &_sampler, _prompt_s, _model_i->seq_n_max);
    }
    else if (! memcmp(mode, "chat", strlen("chat") + 1)) {
        chat(_model_i, &_sampler, _system_prompt_s, _prompt_s, prompt_n_max, _model_i->seq_n_max, debug_);
    }
    else {
        log_msg(stderr, "ERROR: Unknown mode: %s\n", mode);
        error_usage(argv[0]);
    }

    free_sampler(&_sampler);

    _model_i->free_model(_model_i->_model);

    free(_model_i);

    return 0;
}

