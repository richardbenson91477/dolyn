export OMP_NUM_THREADS := 11
export OMP_PLACES := cores
export OMP_PROC_BIND := close
CC := ccache gcc
CFLAGS = -fPIC 
CFLAGS_OPT = $(CFLAGS) -fopenmp -O3 -pipe -march=x86-64-v4 -fomit-frame-pointer -funroll-loops -fpermissive
CFLAGS_DBG = $(CFLAGS) -fopenmp -Og -ggdb -fkeep-inline-functions
PROMPT := "Please count from 1 to 10."
MODEL_PATH := /home/models/dolen_models
MODEL3 := "$(MODEL_PATH)/qwen3_1b"
MODEL3_5 := "$(MODEL_PATH)/qwen3_5_1b"
TOK3 := "$(MODEL_PATH)/qwen3_tokenizer.bin"
TOK3_5 := "$(MODEL_PATH)/qwen3_5_tokenizer.bin"
LIB_SRCS = ext/csafetensors.c ext/json.c
SRCS = $(SRC3_5) $(SRC3_5Q) $(SRC3) $(SRC3Q)
BINS = $(BIN3_5) $(BIN3_5Q) $(BIN3) $(BIN3Q)
BINS_DBG = $(BIN3_5_DBG) $(BIN3_5Q_DBG) $(BIN3_DBG) $(BIN3Q_DBG)
BIN3 = dolen3
BIN3Q = dolen3_quantize
BIN3_DBG = dolen3_dbg
BIN3Q_DBG = dolen3_quantize_dbg
BIN3_5 = dolen3_5
BIN3_5Q = dolen3_5_quantize
BIN3_5_DBG = dolen3_5_dbg
BIN3_5Q_DBG = dolen3_5_quantize_dbg
SEQN3 := 1024
SEQN3_5 := 1024
SRC3 = dolen3.c dolen3_common.c dolen_common.c
SRC3Q = dolen3_quantize.c dolen3_common.c dolen_q_common.c dolen_common.c
SRC3_5 = dolen3_5.c dolen3_5_common.c dolen_common.c
SRC3_5Q = dolen3_5_quantize.c dolen3_5_common.c dolen_q_common.c dolen_common.c
SYSP3 := "You are a helpful assistant."
SYSP3_5 := "You are a helpful assistant."

all: $(BINS) $(BINS_DBG)

test: test3_5

debug: debug3_5

clean:
	rm -vf $(BINS) $(BINS_DBG)


$(BIN3_5): $(SRC3_5)
	$(CC) $(CFLAGS_OPT) -o $@ $^ -lm
	strip $@

$(BIN3_5_DBG): $(SRC3_5)
	$(CC) $(CFLAGS_DBG) -o $@ $^ -lm

$(BIN3_5Q): $(SRC3_5Q) $(LIB_SRCS)
	$(CC) $(CFLAGS_OPT) -o $@ $^ -lm
	strip $@

$(BIN3_5Q_DBG): $(SRC3_5Q) $(LIB_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $^ -lm

test3_5: $(BIN3_5)
	./$(BIN3_5) $(MODEL3_5) -tk $(TOK3_5) -p $(PROMPT) -sp $(SYSP3_5) -n $(SEQN3_5)

test3_5_dbg: $(BIN3_5_DBG)
	./$(BIN3_5_DBG) $(MODEL3_5) -tk $(TOK3_5) -p $(PROMPT) -sp $(SYSP3_5) -n $(SEQN3_5)

debug3_5: $(BIN3_5_DBG)
	./run_gdb_3_5 $(BIN3_5_DBG) $(MODEL3_5) $(TOK3_5) $(PROMPT) $(SYSP3_5) $(SEQN3_5)


$(BIN3): $(SRC3)
	$(CC) $(CFLAGS_OPT) -o $@ $^ -lm
	strip $@

$(BIN3_DBG): $(SRC3)
	$(CC) $(CFLAGS_DBG) -o $@ $^ -lm

$(BIN3Q): $(SRC3Q) $(LIB_SRCS)
	$(CC) $(CFLAGS_OPT) -o $@ $^ -lm
	strip $@

$(BIN3Q_DBG): $(SRC3Q) $(LIB_SRCS)
	$(CC) $(CFLAGS_DBG) -o $@ $^ -lm

test3: $(BIN3)
	./$(BIN3) $(MODEL3) -tk $(TOK3) -p $(PROMPT) -sp $(SYSP3) -n $(SEQN3)

test3_dbg: $(BIN3_DBG)
	./$(BIN3_DBG) $(MODEL3) -tk $(TOK3) -p $(PROMPT) -sp $(SYSP3) -n $(SEQN3)

debug3: $(BIN3_DBG)
	./run_gdb_3 $(BIN3_DBG) $(MODEL3) $(TOK3) $(PROMPT) $(SYSP3) $(SEQN3)

