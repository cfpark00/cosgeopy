#Make
CC=g++
CFLAGS=-Wall -lfftw3 -lm
CFLAGS+=-lfftw3_omp -fopenmp


all: gaussianfield

gaussianfield: gaussianfield.cpp tools.cpp
	$(CC) gaussianfield.cpp tools.cpp $(CFLAGS) -o gaussianfield
clean:
	$(RM) gaussianfield
