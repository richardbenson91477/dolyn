#include "dolen_common_sampler.h"
#include "dolen_common_cmi.h"
#include "dolen_common_io.h"
#include "dolen_common_mem.h"

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

        if (! start) {
            start = time_in_ms();
        }
    }
    log_msg(stdout, "\n");

    if (pos > 1) {
        long end = time_in_ms();
        log_msg(stderr, "INFO: %f tokens per second.\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

static const chat_template CHAT_TEMPLATE_CHATML = {
    .system = "<|im_start|>system\n%s<|im_end|>\n",
    .main = "<|im_start|>user\n%s<|im_end|>\n"
            "<|im_start|>assistant\n",
    .end_turn = "<|im_end|>\n",
};

static char *render_chat_turn(const chat_template *chat_tmpl, bool first_turn, const char *system_prompt,
        const char *prompt, int *rendered_len) {
    const char *format = NULL;
    int len1, len2;

    if (first_turn &&
            system_prompt &&
            system_prompt[0]) {
        len1 = snprintf(NULL, 0, chat_tmpl->system, system_prompt);
        len2 = snprintf(NULL, 0, chat_tmpl->main, prompt);
    } else {
        len1 = snprintf(NULL, 0, chat_tmpl->end_turn);
        len2 = snprintf(NULL, 0, chat_tmpl->main, prompt);
    }

    char *rendered = a_calloc(len1 + len2 + 1);
    if (! rendered) {
        log_msg(stderr, "ERROR: Failed to allocate rendered chat prompt\n");
        exit(EXIT_FAILURE);
    }

    if (first_turn &&
            system_prompt &&
            system_prompt[0]) {
        snprintf(rendered, len1 + 1, chat_tmpl->system, system_prompt);
        snprintf(rendered + len1, len2 + 1, chat_tmpl->main, prompt);
    } else {
        snprintf(rendered, len1 + 1, chat_tmpl->end_turn);
        snprintf(rendered + len1, len2 + 1, chat_tmpl->main, prompt);
    }

    *rendered_len = len1 + len2;
    return rendered;
}

static bool is_chat_stop_token(const model_iface *model_i, int token) {
    if (token == model_i->im_end_id) {
        return true;
    }
    if ((model_i->eos_token_id > 0) &&
            (token == model_i->eos_token_id)) {
        return true;
    }

    return token == 2;
}

void chat_common(model_iface *model_i, Tokenizer *tokenizer, Sampler *sampler, char *system_prompt, char *init_prompt,
        int prompt_n_max, int steps_n_max, bool _debug) {
    const chat_template *chat_tmpl = model_i->chat_template;
    if (! chat_tmpl) {
        chat_tmpl = &CHAT_TEMPLATE_CHATML;
    }
    if ((! chat_tmpl->system) ||
            (! chat_tmpl->main) ||
            (! chat_tmpl->end_turn)) {
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
            if (first_turn &&
                    init_prompt) {
                strncpy(prompt, init_prompt, prompt_n_max);
                prompt[prompt_n_max] = '\0';
            } else {
                log_msg(stdout, "In: ");
                read_msg(prompt, prompt_n_max);
            }
            if (prompt[0] == '\0') {
                continue;
            }

            rendered_prompt = render_chat_turn(chat_tmpl, first_turn, system_prompt, prompt, &rendered_len);

            if (prompt_tokens) {
                free(prompt_tokens);
            }

            prompt_tokens = (int *)a_calloc(((size_t)rendered_len * 4 + 3) * sizeof(int));

            int bos_token = first_turn ? model_i->bos_token_id : 0;
            encode(tokenizer, rendered_prompt, bos_token, 0, prompt_tokens, &prompt_tokens_n);

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
        } else {
            token = next;
        }

        float *logits = model_i->forward(model_i->model, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= prompt_tokens_n) {
            if (is_chat_stop_token(model_i, next)) {
                log_msg(stdout, "\n");
                long end = time_in_ms();
                if ((generated_tokens > 0) &&
                        ((end - start) > 0)) {
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
    log_msg(stderr, " -d  | --debug:               enable debug output, default: disabled\n");
    log_msg(stderr, " -l  | --log <str>:           path to append all I/O to, default: none\n");
    log_msg(stderr, " -h  | --help:                print this help and exit\n");
    log_msg(stderr, " -th | --think:               enable think-mode chat template, default: disabled\n");

    exit(EXIT_FAILURE);
}

int common_main(int argc, char *argv[], model_iface *(*init_fn)(const char *, int, bool), const char *prog_name) {
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
    bool _think = false;

    for (int i = 1; i < argc;) {
        if (argv[i][0] != '-') {
            error_usage(prog_name);
        }

        if ((! strcmp(argv[i], "-h")) ||
                (! strcmp(argv[i], "--help"))) {
            error_usage(prog_name);
        } else if ((! strcmp(argv[i], "-th")) ||
                (! strcmp(argv[i], "--think"))) {
            _think = true;
            i += 1;
            continue;
        } else if ((! strcmp(argv[i], "-d")) ||
                (! strcmp(argv[i], "--debug"))) {
            _debug = true;
            i += 1;
            continue;
        }

        if ((i + 1) >= argc) {
            error_usage(prog_name);
        }

        if ((! strcmp(argv[i], "-m")) ||
                (! strcmp(argv[i], "--model"))) {
            model_path = argv[i + 1];
        } else if ((! strcmp(argv[i], "-t")) ||
                (! strcmp(argv[i], "--temp"))) {
            temperature = atof(argv[i + 1]);
        } else if ((! strcmp(argv[i], "-k")) ||
                (! strcmp(argv[i], "--top_k"))) {
            topk = atoi(argv[i + 1]);
        } else if ((! strcmp(argv[i], "-tp")) ||
                (! strcmp(argv[i], "--top_p"))) {
            topp = atof(argv[i + 1]);
        } else if ((! strcmp(argv[i], "-s")) ||
                (! strcmp(argv[i], "--seed"))) {
            rng_seed = atoi(argv[i + 1]);
        } else if ((! strcmp(argv[i], "-n")) ||
                (! strcmp(argv[i], "--seq_n"))) {
            seq_n_max = atoi(argv[i + 1]);
        } else if ((! strcmp(argv[i], "-pn")) ||
                (! strcmp(argv[i], "--prompt_n"))) {
            prompt_n_max = atoi(argv[i + 1]);
        } else if ((! strcmp(argv[i], "-p")) ||
                (! strcmp(argv[i], "--prompt"))) {
            prompt = a_calloc(strlen(argv[i + 1]) + 1);
            strcpy(prompt, argv[i + 1]);
        } else if ((! strcmp(argv[i], "-pf")) ||
                (! strcmp(argv[i], "--prompt_file"))) {
            prompt_file = argv[i + 1];
        } else if ((! strcmp(argv[i], "-tk")) ||
                (! strcmp(argv[i], "--tokenizer"))) {
            tokenizer_path = argv[i + 1];
        } else if ((! strcmp(argv[i], "-M")) ||
                (! strcmp(argv[i], "--mode"))) {
            mode = argv[i + 1];
        } else if ((! strcmp(argv[i], "-sp")) ||
                (! strcmp(argv[i], "--system_prompt"))) {
            system_prompt = a_calloc(strlen(argv[i + 1]) + 1);
            strcpy(system_prompt, argv[i + 1]);
        } else if ((! strcmp(argv[i], "-l")) ||
                (! strcmp(argv[i], "--log"))) {
            log_path = argv[i + 1];
        } else {
            error_usage(prog_name);
        }

        i += 2;
    }

    if (! model_path) {
        log_msg(stderr, "ERROR: Model path required.\n");
        exit(EXIT_FAILURE);
    }

    if (! rng_seed) {
        rng_seed = (unsigned int)time(NULL);
    }
    log_msg(stderr, "INFO: Using seed %lu\n", rng_seed);

    if (prompt_file) {
        if (prompt) {
            free(prompt);
            prompt = NULL;
        }
        FILE *pf = fopen(prompt_file, "r");
        if (! pf) {
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
        if (! prompt) {
            log_msg(stderr, "ERROR: Memory allocation failed for prompt file\n");
            fclose(pf);
            exit(EXIT_FAILURE);
        }

        size_t read_bytes = fread(prompt, 1, f_len, pf);
        prompt[read_bytes] = '\0';
        fclose(pf);
    }

    model_iface *model_i = init_fn(model_path, seq_n_max, _think);
    if (! model_i) {
        exit(EXIT_FAILURE);
    }

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model_i->vocab_size, model_i->special_tokens);

    Sampler sampler;
    build_sampler(&sampler, model_i->vocab_size, temperature, topk, topp, rng_seed);

    if (! memcmp(mode, "generate", strlen("generate") + 1)) {
        generate_common(model_i, &tokenizer, &sampler, prompt, model_i->seq_n_max);
    } else if (! memcmp(mode, "chat", strlen("chat") + 1)) {
        chat_common(model_i, &tokenizer, &sampler, system_prompt, prompt, prompt_n_max, model_i->seq_n_max, _debug);
    } else {
        log_msg(stderr, "ERROR: Unknown mode: %s\n", mode);
        error_usage(prog_name);
    }

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);

    model_i->free_model(model_i->model);

    free(model_i);

    return 0;
}

