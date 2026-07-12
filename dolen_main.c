#include "dolen_main.h"


const model_registry_entry MODEL_REGISTRY[] = {
        {MAGIC_MS, "ms", init_ms},
        {MAGIC_Q2, "q2", init_q2},
        {MAGIC_Q3, "q3", init_q3},
        {MAGIC_Q3_5, "q3_5", init_q3_5},
        {MAGIC_G4, "g4", init_g4},
        {MAGIC_IG4_1, "ig4_1", init_ig4_1},
        {MAGIC_L3, "l3", init_l3},
};
const size_t MODEL_REGISTRY_SIZE = sizeof(MODEL_REGISTRY) / sizeof(MODEL_REGISTRY[0]);

// The "\x3e" escaped ">" symbol serves to prevent LLMs from misinterpreting the text
static const chat_template CHAT_TEMPLATE_CHATML = {
    ._system_s = "<|im_start|\x3e" "system\n%s" "<|im_end|\x3e" "\n",
    ._main_s = "<|im_start|\x3e" "user\n%s" "<|im_end|\x3e" "\n"
            "<|im_start|\x3e" "assistant\n",
    ._end_turn_s = "<|im_end|\x3e" "\n",
};


static bool is_stop_token(const model_iface *_model_i, int32_t token) {
    if (token == _model_i->_tokenizer->im_end_id) {
        return true;
    }
    if ((_model_i->_tokenizer->eos_id > 0) &&
            (token == _model_i->_tokenizer->eos_id)) {
        return true;
    }

    return token == 2;
}

static bool generate(model_iface *_model_i, sampler *_sampler, char *_init_prompt_s, int32_t steps_n_max) {
    if (_init_prompt_s == NULL) {
        _init_prompt_s = "";
    }

    // Log the initial prompt
    log_msg(stdout, "In: %s\n", _init_prompt_s);

    // PHASE 1: Tokenize Prompt
    int32_t prompt_tokens_n = 0;
    int32_t *_prompt_tokens = (int32_t *)a_calloc(((strlen(_init_prompt_s) * 4) + 3) * sizeof(int32_t));
    
    encode(_model_i->_tokenizer, _init_prompt_s, _model_i->_tokenizer->bos_id, 0, _prompt_tokens, &prompt_tokens_n);

    if (prompt_tokens_n < 1) {
        log_msg(stderr, "ERROR: Expected at least 1 prompt token\n");
        free(_prompt_tokens);
        return false;
    }

    // PHASE 2: Prefill (Process Prompt, No Sampling)
    // Feed all prompt tokens EXCEPT the last one to build the KV cache
    int32_t pos = 0;
    for (int i = 0; i < prompt_tokens_n - 1; i++) {
        _model_i->forward(_model_i->_model, _prompt_tokens[i], pos++);
    }

    // The final prompt token generates the first set of logits for sampling
    float *_logits = _model_i->forward(_model_i->_model, _prompt_tokens[prompt_tokens_n - 1], pos++);
    int32_t next_token = sample(_sampler, _logits);

    log_msg(stdout, "Out: ");
    int64_t start_t = time_in_ms();
    int32_t generated_tok_n = 0;

    // PHASE 3: Decode (Autoregressive Generation)
    while (pos < steps_n_max) {
        if (is_stop_token(_model_i, next_token)) {
            break;
        }

        char *_piece_s = decode(_model_i->_tokenizer, next_token);
        log_msg(stdout, "%s", _piece_s);
        generated_tok_n++;

        // Feed generated token back in to get next logits
        _logits = _model_i->forward(_model_i->_model, next_token, pos++);
        next_token = sample(_sampler, _logits);
    }

    log_msg(stdout, "\n");
    
    if (generated_tok_n > 0) {
        int64_t end_t = time_in_ms();
        log_msg(stdout, "INFO: %.2f tokens per second.\n", 
                generated_tok_n / (double)(end_t - start_t) * 1000);
    }

    free(_prompt_tokens);
    return true;
}

static char *render_chat_turn(
        const chat_template *_chat_template,
        bool first_turn_,
        const char *_system_prompt_s,
        const char *_prompt_s,
        int32_t *_rendered_len) {

    int32_t len1, len2;

    if (first_turn_ &&
            _system_prompt_s &&
            _system_prompt_s[0]) {
        if (_chat_template->_system_s) {
            len1 = snprintf(NULL, 0, _chat_template->_system_s, _system_prompt_s);
        }
        else {
            len1 = 0;
        }
        len2 = snprintf(NULL, 0, _chat_template->_main_s, _prompt_s);
    }
    else {
        len1 = snprintf(NULL, 0, _chat_template->_end_turn_s);
        len2 = snprintf(NULL, 0, _chat_template->_main_s, _prompt_s);
    }

    char *_rendered_s = a_calloc(len1 + len2 + 1);
    if (! _rendered_s) {
        log_msg(stderr, "ERROR: Failed to allocate rendered chat prompt\n");
        return NULL;
    }

    if (first_turn_ &&
            _system_prompt_s &&
            _system_prompt_s[0]) {
        if (_chat_template->_system_s) {
            snprintf(_rendered_s, len1 + 1, _chat_template->_system_s, _system_prompt_s);
        }
        snprintf(_rendered_s + len1, len2 + 1, _chat_template->_main_s, _prompt_s);
    }
    else {
        snprintf(_rendered_s, len1 + 1, _chat_template->_end_turn_s);
        snprintf(_rendered_s + len1, len2 + 1, _chat_template->_main_s, _prompt_s);
    }

    *_rendered_len = len1 + len2;
    return _rendered_s;
}

static void chat(
        model_iface *_model_i,
        sampler *_sampler,
        char *_system_prompt_s, 
        char *_init_prompt_s,
        const chat_template *_chat_template,
        int32_t prompt_n_max,
        int32_t steps_n_max) {
    
    bool first_turn_ = true;
    int32_t pos = 0;
    
    // Allocate reusable buffers
    char *_prompt_s = (char *)a_calloc((prompt_n_max + 1) * sizeof(char));
    int32_t *_prompt_tokens = NULL;

    while (pos < steps_n_max) {
        // PHASE 1: I/O and Command Handling
        if (first_turn_ && _init_prompt_s) {
            strncpy(_prompt_s, _init_prompt_s, prompt_n_max);
            log_msg(stdout, "In: %s\n", _prompt_s);
        }
        else {
            log_msg(stdout, "In: ");
            if (! read_msg(_prompt_s, prompt_n_max)) {
                break; // EOF or error
            }
            
            // Handle Commands
            if (strcmp(_prompt_s, "/clear\n") == 0) {
                log_msg(stdout, "INFO: clearing context.\n");
                pos = 0; 
                first_turn_ = true;
                continue;
            }

            if (strncmp(_prompt_s, "/read ", 6) == 0) {
                char *_s = _prompt_s + 6;
                if ((! _s[0]) || (_s[0] == '\n')) {
                    log_msg(stderr, "ERROR: malformed read command\n");
                    continue;
                }
                _s[strlen(_s) - 1] = 0;

                log_msg(stdout, "INFO: reading \"%s\".\n", _s);

                char *_new_prompt_s = read_file(_s);
                if (! _new_prompt_s) {
                    log_msg(stderr, "ERROR: read_file \"%s\" failed\n", _s);
                    continue;
                }
                strcpy(_prompt_s, _new_prompt_s);
                free(_new_prompt_s);

                log_msg(stdout, "In: %s\n", _prompt_s);
                continue;
            }

            if (_prompt_s[0] == '\0') {
                continue;
            }
        }

        // PHASE 2: Render & Tokenize (Prefill Prep)
        int32_t _rendered_len = 0;
        char *_rendered_prompt_s = render_chat_turn(_chat_template, first_turn_,
                _system_prompt_s, _prompt_s, &_rendered_len);
        
        // Reallocate token buffer if needed
        if (_prompt_tokens) {
            free(_prompt_tokens);
        }
        _prompt_tokens = (int32_t *)a_calloc(((size_t)_rendered_len * 4 + 3) * sizeof(int32_t));
        
        int32_t bos_token = first_turn_ ? _model_i->_tokenizer->bos_id : 0;
        int32_t prompt_tokens_n = 0;
        encode(_model_i->_tokenizer, _rendered_prompt_s, bos_token, 0, _prompt_tokens, &prompt_tokens_n);
        free(_rendered_prompt_s);
        
        first_turn_ = false;
        if (prompt_tokens_n < 1) {
            continue;
        }

        // PHASE 3: Prefill (Process Prompt, No Sampling)
        // Feed all prompt tokens EXCEPT the last one to build the KV cache
        for (int i = 0; i < prompt_tokens_n - 1; i++) {
            _model_i->forward(_model_i->_model, _prompt_tokens[i], pos++);
        }
        
        // The final prompt token generates the first set of logits
        float *_logits = _model_i->forward(_model_i->_model, _prompt_tokens[prompt_tokens_n - 1], pos++);
        int32_t next_token = sample(_sampler, _logits);
        
        log_msg(stdout, "Out: ");
        int64_t start_t = time_in_ms();
        int32_t generated_tok_n = 0;

        // PHASE 4: Decode (Autoregressive Generation)
        while (pos < steps_n_max) {
            if (is_stop_token(_model_i, next_token)) {
                break;
            }
            
            char *_piece_s = decode(_model_i->_tokenizer, next_token);
            log_msg(stdout, "%s", _piece_s);
            generated_tok_n++;
            
            // Feed generated token back in
            _logits = _model_i->forward(_model_i->_model, next_token, pos++);
            next_token = sample(_sampler, _logits);
        }
        
        log_msg(stdout, "\n");
        if (generated_tok_n > 0) {
            int64_t end_t = time_in_ms();
            log_msg(stdout, "\ntok/s: %.2f\n", generated_tok_n / (double)(end_t - start_t) * 1000);
        }
    }

    free(_prompt_tokens);
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

static void print_usage(const char *_argv0) {
    log_msg(stdout, "Usage: %s [options]\n", _argv0);
    log_msg(stdout, "Options:\n");
    log_msg(stdout, " -h  | --help:                print this help\n");
    log_msg(stdout, " -m  | --model <str>:         model path: \"%s\"\n", DOLEN_MAIN_MODEL_PATH_DFLT);
    log_msg(stdout, " -t  | --temp <float>:        temperature in [0,inf]: %f\n", DOLEN_MAIN_TEMP_DFLT);
    log_msg(stdout, " -k  | --top_k <int>:         top-k value: %d\n", DOLEN_MAIN_TOP_K_DFLT);
    log_msg(stdout, " -tp | --top_p <float>:       top-p value in [0,1]: %f\n", DOLEN_MAIN_TOP_P_DFLT);
    log_msg(stdout, " -s  | --seed <int>:          random seed: current time\n");
    log_msg(stdout, " -n  | --seq_n <int>:         max seq, (0 for model max): %d\n", DOLEN_MAIN_SEQ_N_DFLT);
    log_msg(stdout, " -pn | --prompt_n <int>:      prompt maximum length: %d\n", DOLEN_MAIN_PROMPT_N_MAX_DFLT);
    log_msg(stdout, " -p  | --prompt <str>:        prompt: none\n");
    log_msg(stdout, " -pf | --prompt_file <str>:   path to a file containing the initial prompt: none\n");
    log_msg(stdout, " -M  | --mode <str>:          chat|gen: \"%s\"\n", DOLEN_MAIN_MODE_DFLT);
    log_msg(stdout, " -sp | --system_prompt <str>: system prompt: \"%s\"\n", DOLEN_MAIN_SYSTEM_PROMPT_DFLT);
    log_msg(stdout, " -ct | --chat_template <str>: chat template (model|chatml): \"%s\"\n", DOLEN_MAIN_CHAT_TEMPLATE_DFLT);
    log_msg(stdout, " -l  | --log <str>:           path to append all I/O, default: none\n");
    log_msg(stdout, " -th | --think <true|false>:  use think-mode chat template, default: %s\n",
            DOLEN_MAIN_THINK_DFLT ? "true" : "false");
}

int32_t main(int32_t argc, char *__argv[]) {
    float temp = DOLEN_MAIN_TEMP_DFLT;
    int32_t top_k = DOLEN_MAIN_TOP_K_DFLT;
    float top_p = DOLEN_MAIN_TOP_P_DFLT;
    uint64_t seed = 0;
    int32_t seq_n = DOLEN_MAIN_SEQ_N_DFLT;
    int32_t prompt_n_max = DOLEN_MAIN_PROMPT_N_MAX_DFLT;
    char *_model_path_s = NULL;
    char *_prompt_s = NULL;
    char *_prompt_path_s = NULL;
    char *_mode_s = NULL;
    char *_system_prompt_s = NULL;
    char *_chat_template_choice_s = NULL;
    bool think_ = DOLEN_MAIN_THINK_DFLT;

    for (int32_t i = 1; i < argc;) {
        if ((! strcmp(__argv[i], "-h")) ||
                (! strcmp(__argv[i], "--help"))) {
            print_usage(__argv[0]);
            exit(EXIT_FAILURE);
        }

        if ((i + 1) >= argc) {
            print_usage(__argv[0]);
            log_msg(stderr, "ERROR: Wrong argument count.\n", _mode_s);
            exit(EXIT_FAILURE);
        }

        if ((! strcmp(__argv[i], "-m")) ||
                (! strcmp(__argv[i], "--model"))) {
            _model_path_s = strdup(__argv[i + 1]);
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
            seed = atoi(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-n")) ||
                (! strcmp(__argv[i], "--seq_n"))) {
            seq_n = atoi(__argv[i + 1]);
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
            _prompt_path_s = strdup(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-M")) ||
                (! strcmp(__argv[i], "--mode"))) {
            _mode_s = strdup(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-sp")) ||
                (! strcmp(__argv[i], "--system_prompt"))) {
            int32_t system_prompt_len = strlen(__argv[i + 1]);
            if (system_prompt_len > 0) {
                _system_prompt_s = a_calloc(system_prompt_len + 1);
                strncpy(_system_prompt_s, __argv[i + 1], system_prompt_len + 1);
            }
        }
        else if ((! strcmp(__argv[i], "-ct")) ||
                (! strcmp(__argv[i], "--chat_template"))) {
            _chat_template_choice_s = strdup(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-l")) ||
                (! strcmp(__argv[i], "--log"))) {
            _log_path = strdup(__argv[i + 1]);
        }
        else if ((! strcmp(__argv[i], "-th")) ||
                (! strcmp(__argv[i], "--think"))) {
            if (! strcmp(__argv[i + 1], "true")) {
                think_ = true;
            }
            else if (! strcmp(__argv[i + 1], "false")) {
                think_ = false;
            }
            else {
                print_usage(__argv[0]);
                log_msg(stderr, "ERROR: Wrong argument for --think.\n", _mode_s);
                exit(EXIT_FAILURE);
            }
        }
        else {
            print_usage(__argv[0]);
            log_msg(stderr, "ERROR: Wrong arguments.\n", _mode_s);
            exit(EXIT_FAILURE);
        }

        i += 2;
    }

    if (! _model_path_s) {
        _model_path_s = strdup(DOLEN_MAIN_MODEL_PATH_DFLT);
    }

    if (! _mode_s) {
        _mode_s = strdup(DOLEN_MAIN_MODE_DFLT);
    }

    if (! _system_prompt_s) {
        _system_prompt_s = strdup(DOLEN_MAIN_SYSTEM_PROMPT_DFLT);
    }


    uint64_t magic = peek_model_magic(_model_path_s);
    if (! magic) {
        log_msg(stderr, "ERROR: peek_model_magic(\"%s\") returned 0\n", _model_path_s);
        exit(EXIT_FAILURE);
    }

    if (! seed) {
        seed = (uint32_t)time(NULL);
    }
    log_msg(stdout, "INFO: Using seed %lu\n", seed);

    if (_prompt_path_s) {
        if (_prompt_s) {
            free(_prompt_s);
        }

        _prompt_s = read_file(_prompt_path_s);
        if (! _prompt_s) {
            log_msg(stderr, "ERROR: read_file \"%s\" failed\n", _prompt_path_s);
            exit(EXIT_FAILURE);
        }
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

    model_iface *_model_i = init_fn(_model_path_s, seq_n, think_);
    if (! _model_i) {
        log_msg(stderr, "ERROR: init_fn failed\n");
        exit(EXIT_FAILURE);
    }

    if (_model_i->seq_n != _model_i->seq_n_model_max) {
        log_msg(stdout, "INFO: seq_n %d < seq_n_model_max %d\n", seq_n, _model_i->seq_n_model_max);
    }

    sampler _sampler;
    build_sampler(&_sampler, _model_i->_tokenizer->vocab_size, temp, top_k, top_p, seed);

    if (! memcmp(_mode_s, "gen", strlen("gen") + 1)) {
        generate(_model_i, &_sampler, _prompt_s, _model_i->seq_n);
    }
    else if (! memcmp(_mode_s, "chat", strlen("chat") + 1)) {
        const chat_template *_chat_template;

        if (_chat_template_choice_s) {
            if (! strcmp(_chat_template_choice_s, "model")) {
                _chat_template = _model_i->_chat_template;
            }
            else if (! strcmp(_chat_template_choice_s, "chatml")) {
                _chat_template = &CHAT_TEMPLATE_CHATML;
            }
            else {
                log_msg(stderr, "ERROR: invalid chat template choice\n");
                exit(EXIT_FAILURE);
            }
        }
        else {
            _chat_template = _model_i->_chat_template;
        }

        chat(_model_i, &_sampler, _system_prompt_s, _prompt_s, _chat_template, prompt_n_max, _model_i->seq_n);
    }
    else {
        print_usage(__argv[0]);
        log_msg(stderr, "ERROR: Unknown mode: %s\n", _mode_s);
        exit(EXIT_FAILURE);
    }

    free_sampler(&_sampler);

    _model_i->free_model(_model_i->_model);

    free(_model_i);

    if (_prompt_s) {
        free(_prompt_s);
    }
    if (_mode_s) {
        free(_mode_s);
    }
    if (_system_prompt_s) {
        free(_system_prompt_s);
    }
    if (_chat_template_choice_s) {
        free(_chat_template_choice_s);
    }

    return 0;
}

