export OMP_NUM_THREADS := 11
export OMP_PLACES := cores
export OMP_PROC_BIND := close

CC := gcc

CFLAGS = -fPIC 
CFLAGS_OPT = $(CFLAGS) -fopenmp -O3 -pipe -march=x86-64-v4 -fomit-frame-pointer -funroll-loops -fpermissive
CFLAGS_DBG = $(CFLAGS) -fopenmp -Og -ggdb -fkeep-inline-functions

PROMPT := "Please count from 1 to 10 using a comma separated list. Spell out the numbers in English."
SYS_PROMPT := "You are a helpful language model."
SEQN := 1024

MODEL_PATH := ${lena_rolocal_path}/models/dolen_models
MODEL_Q3_5 := "$(MODEL_PATH)/qwen3_5_1b_q8.dolq"
MODEL_Q3 := "$(MODEL_PATH)/qwen3_1b_q8.dolq"
MODEL_G4 := "$(MODEL_PATH)/gemma4_12b_a_q8.dolq"
MODEL_IG4_1 := "$(MODEL_PATH)/granite4_1_3b_a_q8.dolq"
MODEL_L3 := "/scratch/model"

BIN_Q3_5 = dolen_q3_5
BIN_Q3_5_Q = dolen_q3_5_quantize
BIN_Q3_5_DBG = dolen_q3_5_dbg
BIN_Q3_5_Q_DBG = dolen_q3_5_quantize_dbg

BIN_Q3 = dolen_q3
BIN_Q3_Q = dolen_q3_quantize
BIN_Q3_DBG = dolen_q3_dbg
BIN_Q3_Q_DBG = dolen_q3_quantize_dbg

BIN_G4 = dolen_g4
BIN_G4_Q = dolen_g4_quantize
BIN_G4_DBG = dolen_g4_dbg
BIN_G4_Q_DBG = dolen_g4_quantize_dbg

BIN_IG4_1 = dolen_ig4_1
BIN_IG4_1_Q = dolen_ig4_1_quantize
BIN_IG4_1_DBG = dolen_ig4_1_dbg
BIN_IG4_1_Q_DBG = dolen_ig4_1_quantize_dbg

BIN_L3 = dolen_l3
BIN_L3_Q = dolen_l3_quantize
BIN_L3_DBG = dolen_l3_dbg
BIN_L3_Q_DBG = dolen_l3_quantize_dbg

LIB_SRCS = ext/csafetensors.c ext/json.c
LIB_INCS = ext/csafetensors.h ext/json.h

DOLEN_COMMON_INC=dolen_common_cmi.h dolen_common_io.h dolen_common_math.h dolen_common_mem.h dolen_common_qtensor.h dolen_common_sampler.h dolen_common_tokenizer.h
DOLEN_COMMON_SRC=dolen_common_cmi.c dolen_common_io.c dolen_common_math.c dolen_common_mem.c dolen_common_qtensor.c dolen_common_sampler.c dolen_common_tokenizer.c

SRC_Q3_5 = dolen_q3_5.c dolen_q3_5_common.c $(DOLEN_COMMON_SRC)
INC_Q3_5 = dolen_q3_5_common.h $(DOLEN_COMMON_INC)
SRC_Q3_5_Q = dolen_q3_5_quantize.c dolen_q3_5_common.c dolen_quantize_common.c $(DOLEN_COMMON_SRC)
INC_Q3_5_Q = dolen_q3_5_common.h dolen_quantize_common.h $(DOLEN_COMMON_INC)

SRC_Q3 = dolen_q3.c dolen_q3_common.c $(DOLEN_COMMON_SRC)
INC_Q3 = dolen_q3_common.h $(DOLEN_COMMON_INC)
SRC_Q3_Q = dolen_q3_quantize.c dolen_q3_common.c dolen_quantize_common.c $(DOLEN_COMMON_SRC)
INC_Q3_Q = dolen_q3_common.h dolen_quantize_common.h $(DOLEN_COMMON_INC)

SRC_G4 = dolen_g4.c dolen_g4_common.c $(DOLEN_COMMON_SRC)
INC_G4 = dolen_g4_common.h $(DOLEN_COMMON_INC)
SRC_G4_Q = dolen_g4_quantize.c dolen_g4_common.c dolen_quantize_common.c $(DOLEN_COMMON_SRC)
INC_G4_Q = dolen_g4_common.h dolen_quantize_common.h $(DOLEN_COMMON_INC)
 
SRC_IG4_1 = dolen_ig4_1.c dolen_ig4_1_common.c $(DOLEN_COMMON_SRC)
INC_IG4_1 = dolen_ig4_1_common.h $(DOLEN_COMMON_INC)
SRC_IG4_1_Q = dolen_ig4_1_quantize.c dolen_ig4_1_common.c dolen_quantize_common.c $(DOLEN_COMMON_SRC)
INC_IG4_1_Q = dolen_ig4_1_common.h dolen_quantize_common.h $(DOLEN_COMMON_INC)

SRC_L3 = dolen_l3.c dolen_l3_common.c $(DOLEN_COMMON_SRC)
INC_L3 = dolen_l3_common.h $(DOLEN_COMMON_INC)
SRC_L3_Q = dolen_l3_quantize.c dolen_l3_common.c dolen_quantize_common.c $(DOLEN_COMMON_SRC)
INC_L3_Q = dolen_l3_common.h dolen_quantize_common.h $(DOLEN_COMMON_INC)


SRCS = $(SRC_Q3_5) $(SRC_Q3_5_Q) $(SRC_Q3) $(SRC_Q3_Q) $(SRC_G4) $(SRC_G4_Q) $(SRC_IG4_1) $(SRC_IG4_1_Q) $(SRC_L3) $(SRC_L3_Q) 
BINS = $(BIN_Q3_5) $(BIN_Q3_5_Q) $(BIN_Q3) $(BIN_Q3_Q) $(BIN_G4) $(BIN_G4_Q) $(BIN_IG4_1) $(BIN_IG4_1_Q) $(BIN_L3) $(BIN_L3_Q)
BINS_DBG = $(BIN_Q3_5_DBG) $(BIN_Q3_5_Q_DBG) $(BIN_Q3_DBG) $(BIN_Q3_Q_DBG) $(BIN_G4_DBG) $(BIN_G4_Q_DBG) $(BIN_IG4_1_DBG) $(BIN_IG4_1_Q_DBG) $(BIN_L3_DBG) $(BIN_L3_Q_DBG) 


all: strip $(BINS_DBG)

test: test_q3_5
test_dbg: test_q3_5_dbg
debug: debug_q3_5
debug_q: debug_q3_5_q


$(BIN_Q3_5): $(SRC_Q3_5) $(INC_Q3_5)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_5) -lm

$(BIN_Q3_5_DBG): $(SRC_Q3_5) $(INC_Q3_5)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_5) -lm

$(BIN_Q3_5_Q): $(SRC_Q3_5_Q) $(LIB_SRCS) $(INC_Q3_5_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_5_Q) $(LIB_SRCS) -lm

$(BIN_Q3_5_Q_DBG): $(SRC_Q3_5_Q) $(LIB_SRCS) $(INC_Q3_5_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_5_Q) $(LIB_SRCS) -lm

test_q3_5: $(BIN_Q3_5)
	./$(BIN_Q3_5) -m $(MODEL_Q3_5) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

test_q3_5_dbg: $(BIN_Q3_5_DBG)
	./$(BIN_Q3_5_DBG) -m $(MODEL_Q3_5) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

debug_q3_5: $(BIN_Q3_5_DBG)
	./dolen_gdb $(BIN_Q3_5_DBG) $(MODEL_Q3_5) $(PROMPT) $(SYS_PROMPT) $(SEQN)

debug_q3_5_q: $(BIN_Q3_5_Q_DBG)
	./dolen_gdb_q $(BIN_Q3_5_Q_DBG)


$(BIN_Q3): $(SRC_Q3) $(INC_Q3)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3) -lm

$(BIN_Q3_DBG): $(SRC_Q3) $(INC_Q3)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3) -lm

$(BIN_Q3_Q): $(SRC_Q3_Q) $(LIB_SRCS) $(INC_Q3_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_Q3_Q) $(LIB_SRCS) -lm

$(BIN_Q3_Q_DBG): $(SRC_Q3_Q) $(LIB_SRCS) $(INC_Q3_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_Q3_Q) $(LIB_SRCS) -lm

test_q3: $(BIN_Q3)
	./$(BIN_Q3) -m $(MODEL_Q3) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

test_q3_dbg: $(BIN_Q3_DBG)
	./$(BIN_Q3_DBG) -m $(MODEL_Q3) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

debug_q3: $(BIN_Q3_DBG)
	./dolen_gdb $(BIN_Q3_DBG) $(MODEL_Q3) $(PROMPT) $(SYS_PROMPT) $(SEQN)

debug_q3_q: $(BIN_Q3_Q_DBG)
	./dolen_gdb_q $(BIN_Q3_Q_DBG)


$(BIN_G4): $(SRC_G4) $(INC_G4)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_G4) -lm

$(BIN_G4_DBG): $(SRC_G4) $(INC_G4)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_G4) -lm

$(BIN_G4_Q): $(SRC_G4_Q) $(LIB_SRCS) $(INC_G4_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_G4_Q) $(LIB_SRCS) -lm

$(BIN_G4_Q_DBG): $(SRC_G4_Q) $(LIB_SRCS) $(INC_G4_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_G4_Q) $(LIB_SRCS) -lm

test_g4: $(BIN_G4)
	./$(BIN_G4) -m $(MODEL_G4) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

test_g4_dbg: $(BIN_G4_DBG)
	./$(BIN_G4_DBG) -m $(MODEL_G4) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

debug_g4: $(BIN_G4_DBG)
	./dolen_gdb $(BIN_G4_DBG) $(MODEL_G4) $(PROMPT) $(SYS_PROMPT) $(SEQN)

debug_g4_q: $(BIN_G4_Q_DBG)
	./dolen_gdb_q $(BIN_G4_Q_DBG)


$(BIN_IG4_1): $(SRC_IG4_1) $(INC_IG4_1)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_IG4_1) -lm

$(BIN_IG4_1_DBG): $(SRC_IG4_1) $(INC_IG4_1)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_IG4_1) -lm

$(BIN_IG4_1_Q): $(SRC_IG4_1_Q) $(LIB_SRCS) $(INC_IG4_1_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_IG4_1_Q) $(LIB_SRCS) -lm

$(BIN_IG4_1_Q_DBG): $(SRC_IG4_1_Q) $(LIB_SRCS) $(INC_IG4_1_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_IG4_1_Q) $(LIB_SRCS) -lm

test_ig4_1: $(BIN_IG4_1)
	./$(BIN_IG4_1) -m $(MODEL_IG4_1) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

test_ig4_1_dbg: $(BIN_IG4_1_DBG)
	./$(BIN_IG4_1_DBG) -m $(MODEL_IG4_1) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

debug_ig4_1: $(BIN_IG4_1_DBG)
	./dolen_gdb $(BIN_IG4_1_DBG) $(MODEL_IG4_1) $(PROMPT) $(SYS_PROMPT) $(SEQN)

debug_ig4_1_q: $(BIN_IG4_1_Q_DBG)
	./dolen_gdb_q $(BIN_IG4_1_Q_DBG)


$(BIN_L3): $(SRC_L3) $(INC_L3)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_L3) -lm

$(BIN_L3_DBG): $(SRC_L3) $(INC_L3)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_L3) -lm

$(BIN_L3_Q): $(SRC_L3_Q) $(LIB_SRCS) $(INC_L3_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_OPT) -o $@ $(SRC_L3_Q) $(LIB_SRCS) -lm

$(BIN_L3_Q_DBG): $(SRC_L3_Q) $(LIB_SRCS) $(INC_L3_Q) $(LIB_INCS)
	$(CC) $(CFLAGS_DBG) -o $@ $(SRC_L3_Q) $(LIB_SRCS) -lm

test_l3: $(BIN_L3)
	./$(BIN_L3) -m $(MODEL_L3) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

test_l3_dbg: $(BIN_L3_DBG)
	./$(BIN_L3_DBG) -m $(MODEL_L3) -p $(PROMPT) -sp $(SYS_PROMPT) -n $(SEQN)

debug_l3: $(BIN_L3_DBG)
	./dolen_gdb $(BIN_L3_DBG) $(MODEL_L3) $(PROMPT) $(SYS_PROMPT) $(SEQN)

debug_l3_q: $(BIN_L3_Q_DBG)
	./dolen_gdb_q $(BIN_L3_Q_DBG)


strip: $(BINS)
	strip $(BINS)

format:
	clang-format -i $(SRCS)

tags_rebuild:
	./_tags_rebuild

clean:
	rm -vf $(BINS) $(BINS_DBG)

