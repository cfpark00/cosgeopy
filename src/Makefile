#Make
#CC=mpic++
CC=g++
CFLAGS=-Wall -lfftw3_omp -lfftw3 -lm -fopenmp -O3 -std=c++11

#options
OPTIONS=
OPTIONS += -DDOUBLE


COMPILE_FLAGS = $(CFLAGS) $(OPTIONS)

all: gaussianfield powerspectrum bispectrum fft3d bispectrum_custom_k isotropic_wavelet given_filter_scatter

gaussianfield: gaussianfield.cpp tools.cpp
	$(CC) gaussianfield.cpp tools.cpp $(COMPILE_FLAGS) -o gaussianfield 
powerspectrum: powerspectrum.cpp tools.cpp
	$(CC) powerspectrum.cpp tools.cpp $(COMPILE_FLAGS) -o powerspectrum 
bispectrum: bispectrum.cpp tools.cpp
	$(CC) bispectrum.cpp tools.cpp $(COMPILE_FLAGS) -o bispectrum
bispectrum_custom_k: bispectrum_custom_k.cpp tools.cpp
	$(CC) bispectrum_custom_k.cpp tools.cpp $(COMPILE_FLAGS) -o bispectrum_custom_k
#bispectrum_naive: bispectrum_naive.cpp tools.cpp
#	$(CC) bispectrum_naive.cpp tools.cpp $(CFLAGS) -o bispectrum_naive
fft3d: fft3d.cpp
	$(CC) fft3d.cpp $(COMPILE_FLAGS) -o fft3d
isotropic_wavelet: isotropic_wavelet.cpp WC_tools.cpp
	$(CC) isotropic_wavelet.cpp WC_tools.cpp $(COMPILE_FLAGS) -o isotropic_wavelet
given_filter_scatter: given_filter_scatter.cpp WC_tools.cpp
	$(CC) given_filter_scatter.cpp WC_tools.cpp $(COMPILE_FLAGS) -o given_filter_scatter
clean:
	$(RM) fullrun gaussianfield powerspectrum bispectrum fft3d bispectrum_custom_k isotropic_wavelet
