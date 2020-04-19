#Make
#CC=mpic++
CC=g++
CFLAGS=-Wall -lfftw3_omp -lfftw3 -lm -fopenmp -O3


all: fullrun gaussianfield powerspectrum bispectrum bispectrum_naive fft3d

fullrun: fullrun.cpp tools.cpp
	$(CC) fullrun.cpp tools.cpp $(CFLAGS) -o fullrun
gaussianfield: gaussianfield.cpp tools.cpp
	$(CC) gaussianfield.cpp tools.cpp $(CFLAGS) -o gaussianfield
powerspectrum: powerspectrum.cpp tools.cpp
	$(CC) powerspectrum.cpp tools.cpp $(CFLAGS) -o powerspectrum
bispectrum: bispectrum.cpp tools.cpp
	$(CC) bispectrum.cpp tools.cpp $(CFLAGS) -o bispectrum
bispectrum_naive: bispectrum_naive.cpp tools.cpp
	$(CC) bispectrum_naive.cpp tools.cpp $(CFLAGS) -o bispectrum_naive
fft3d: fft3d.cpp
	$(CC) fft3d.cpp $(CFLAGS) -o fft3d
clean:
	$(RM) fullrun gaussianfield powerspectrum bispectrum bispectrum_naive
