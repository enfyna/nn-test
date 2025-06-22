CFLAGS=-Wall -Wextra -pedantic -I./src -lm

default: build/nn_xor

build/nn_xor: src/nn_xor.c src/mat.h
	clang $< -o $@ $(CFLAGS)

build/xor: src/xor.c src/mat.h
	clang $< -o $@ $(CFLAGS)

build/main: src/main.c
	clang $^ -o $@ -lm

build/simple: src/simple.c
	clang $^ -o $@ -lm

build/gate: src/gate.c
	clang $^ -o $@ -lm
