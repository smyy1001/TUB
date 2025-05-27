TARGET_SHELL = shell

BIN = \
	$(TARGET_SHELL)

SRC_SHELL = \
	shell.c \
	shell.h \
	util/linenoise.c \
	util/linenoise.h

CC     = clang
CFLAGS = -g -Wall -Wextra -O2

.PHONY: default all
default: all
all: $(BIN)

$(BIN): $(SRC_SHELL)
	$(CC) $(CFLAGS) shell.c util/linenoise.c -o $(TARGET_SHELL)

.PHONY: run
run: $(BIN)
	./$(TARGET_SHELL)

.PHONY: debug
debug: $(BIN)
	gdb ./$(TARGET_SHELL)

.PHONY: clean
clean:
	rm -rf $(TARGET_SHELL)
