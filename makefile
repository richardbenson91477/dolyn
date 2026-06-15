export OMP_NUM_THREADS := 11
export OMP_PLACES := cores
export OMP_PROC_BIND := close
CC := gcc
CFLAGS = -fPIC 
CFLAGS_OPT = $(CFLAGS) -fopenmp -O3 -pipe -march=x86-64-v4 -fomit-frame-pointer -funroll-loops -fpermissive
CFLAGS_DBG = $(CFLAGS) -fopenmp -Og -ggdb -fkeep-inline-functions
PROMPT := "Please count from 1 to 10 using a comma separated list. Spell out the numbers in English."
SYS_PROMPT := "You are a helpful language model."
MODEL_PATH := /home/models/dolen_models
MODEL_Q3_5 := "$(MODEL_PATH)/qwen3_5_1b"
MODEL_Q3 := "$(MODEL_PATH)/qwen3_1b"
MODEL_G4U := "$(MODEL_PATH)/gemma4_12b_a"
TOKENIZER_Q3_5 := "$(MODEL_PATH)/qwen3_5_tokenizer.bin"
TOKENIZER_Q3 := "$(MODEL_PATH)/qwen3_tokenizer.bin"
TOKENIZER_G4U := "$(MODEL_PATH)/gemma4_12b_tokenizer.bin"
LIB_SRCS = ext/csafetensors.c ext/json.c
LIB_INCS = ext/csafetensors.h ext/json.h
BIN_Q3_5 = dolen_q3_5
BIN_Q3_5_Q = dolen_q3_5_quantize
BIN_Q3_5_DBG = dolen_q3_5_dbg
BIN_Q3_5_Q_DBG = dolen_q3_5_quantize_dbg
BIN_Q3 = dolen_q3
BIN_Q3_Q = dolen_q3_quantize
BIN_Q3_DBG = dolen_q3_dbg
BIN_Q3_Q_DBG = dolen_q3_quantize_dbg
BIN_G4U = dolen_g4u
BIN_G4U_Q = dolen_g4u_quantize
BIN_G4U_DBG = dolen_g4u_dbg
BIN_G4U_Q_DBG = dolen_g4u_quantize_dbg
SEQN_Q3_5 := 1024
SEQN_Q3 := 1024
SEQN_G4U := 1024
SRC_Q3_5 = dolen_q3_5.c dolen_q3_5_common.c dolen_common.c
INC_Q3_5 = dolen_q3_5_common.h dolen_common.h
SRC_Q3_5_Q = dolen_q3_5_quantize.c dolen_q3_5_common.c dolen_quantize_common.c dolen_common.c
INC_Q3_5_Q = dolen_q3_5_common.h dolen_quantize_common.h dolen_common.h
SRC_Q3 = dolen_q3.c dolen_q3_common.c dolen_common.c
INC_Q3 = dolen_q3_common.h dolen_common.h
SRC_Q3_Q = dolen_q3_quantize.c dolen_q3_common.c dolen_quantize_common.c dolen_common.c
INC_Q3_Q = dolen_q3_common.h dolen_quantize_common.h dolen_common.h
SRC_G4U = dolen_g4u.c dolen_g4u_common.c dolen_common.c
INC_G4U = dolen_g4u_common.h dolen_common.h
SRC_G4U_Q = dolen_g4u_quantize.c dolen_g4u_common.c dolen_quantize_common.c dolen_common.c
INC_G4U_Q = dolen_g4u_common.h dolen_quantize_common.h dolen_common.h
SRCS = $(SRC_Q3_5) $(SRC_Q3_5_Q) $(SRC_Q3) $(SRC_Q3_Q) $(SRC_G4U) $(SRC_G4U_Q)
BINS = $(BIN_Q3_5) $(BIN_Q3_5_Q) $(BIN_Q3) $(BIN_Q3_Q) $(BIN_G4U) $(BIN_G4U_Q)
BINS_DBG = $(BIN_Q3_5_DBG) $(BIN_Q3_5_Q_DBG) $(BIN_Q3_DBG) $(BIN_Q3_Q_DBG) $(BIN_G4U_DBG) $(BIN_G4U_Q_DBG)


all: strip $(BINS_DBG)

test: test_q3_5
test_dbg: test_q3_5_dbg
debug: debug_q3_5
debug_q: debug_q35_q


$(BIN_Q3_5): $(SRC_Q3_5) $(INC_Q3_5)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_5) -lm

$(BIN_Q3_5_DBG): $(SRC_Q3_5) $(INC_Q3_5)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_5) -lm

$(BIN_Q3_5_Q): $(SRC_Q3_5_Q) $(LIB_SRCS) $(INC_Q3_5_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_5_Q) $(LIB_SRCS) -lm

$(BIN_Q3_5_Q_DBG): $(SRC_Q3_5_Q) $(LIB_SRCS) $(INC_Q3_5_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_5_Q) $(LIB_SRCS) -lm

test_q3_5: $(BIN_Q3_5)
	./$(BIN_Q3_5) -m $(MODEL_Q3_5) -tk $(TOKENIZER_Q3_5) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN_Q3_5)

test_q3_5_dbg: $(BIN_Q3_5_DBG)
	./$(BIN_Q3_5_DBG) -m $(MODEL_Q3_5) -tk $(TOKENIZER_Q3_5) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN_Q3_5)

debug_q3_5: $(BIN_Q3_5_DBG)
	./dolen_run_gdb $(BIN_Q3_5_DBG) $(MODEL_Q3_5) $(TOKENIZER_Q3_5) $(PROMPT) $(SYS_PROMPT) $(SEQN_Q3_5)

debug_q3_5_q: $(BIN_Q3_5_Q_DBG)
	./dolen_q_run_gdb $(BIN_Q3_5_Q_DBG)


$(BIN_Q3): $(SRC_Q3) $(INC_Q3)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3) -lm

$(BIN_Q3_DBG): $(SRC_Q3) $(INC_Q3)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3) -lm

$(BIN_Q3_Q): $(SRC_Q3_Q) $(LIB_SRCS) $(INC_Q3_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_Q) $(LIB_SRCS) -lm

$(BIN_Q3_Q_DBG): $(SRC_Q3_Q) $(LIB_SRCS) $(INC_Q3_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_Q) $(LIB_SRCS) -lm

test_q3: $(BIN_Q3)
	./$(BIN_Q3) -m $(MODEL_Q3) -tk $(TOKENIZER_Q3) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN_Q3)

test_q3_dbg: $(BIN_Q3_DBG)
	./$(BIN_Q3_DBG) -m $(MODEL_Q3) -tk $(TOKENIZER_Q3) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN_Q3)

debug_q3: $(BIN_Q3_DBG)
	./dolen_run_gdb $(BIN_Q3_DBG) $(MODEL_Q3) $(TOKENIZER_Q3) $(PROMPT) $(SYS_PROMPT) $(SEQN_Q3)

debug_q3_q: $(BIN_Q3_Q_DBG)
	./dolen_q_run_gdb $(BIN_Q3_Q_DBG)


$(BIN_G4U): $(SRC_G4U) $(INC_G4U)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_G4U) -lm

$(BIN_G4U_DBG): $(SRC_G4U) $(INC_G4U)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_G4U) -lm

$(BIN_G4U_Q): $(SRC_G4U_Q) $(LIB_SRCS) $(INC_G4U_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_G4U_Q) $(LIB_SRCS) -lm

$(BIN_G4U_Q_DBG): $(SRC_G4U_Q) $(LIB_SRCS) $(INC_G4U_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_G4U_Q) $(LIB_SRCS) -lm

test_g4u: $(BIN_G4U)
	./$(BIN_G4U) -m $(MODEL_G4U) -tk $(TOKENIZER_G4U) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN_G4U)

test_g4u_dbg: $(BIN_G4U_DBG)
	./$(BIN_G4U_DBG) -m $(MODEL_G4U) -tk $(TOKENIZER_G4U) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN_G4U)

debug_g4u: $(BIN_G4U_DBG)
	./dolen_run_gdb $(BIN_G4U_DBG) $(MODEL_G4U) $(TOKENIZER_G4U) $(PROMPT) $(SYS_PROMPT) $(SEQN_G4U)

debug_g4u_q: $(BIN_G4U_Q_DBG)
	./dolen_q_run_gdb $(BIN_G4U_Q_DBG)


strip: $(BINS)
	strip $(BINS)

clean:
	rm -vf $(BINS) $(BINS_DBG)

