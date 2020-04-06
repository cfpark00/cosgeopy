#Make
#CC=mpic++
CC=g++
CFLAGS=-Wall -lfftw3_omp -lfftw3 -lm -fopenmp


all: gaussianfield

gaussianfield: gaussianfield.cpp tools.cpp
	$(CC) gaussianfield.cpp tools.cpp $(CFLAGS) -o gaussianfield
clean:
	$(RM) gaussianfield
