#Make
#CC=mpic++
CC=g++
CFLAGS=-Wall -lfftw3_omp -lfftw3 -lm -fopenmp -O3


all: gaussianfield powerspectrum bispectrum fft3d bispectrum_custom_k isotropic_wavelet

gaussianfield: gaussianfield.cpp tools.cpp
	$(CC) gaussianfield.cpp tools.cpp $(CFLAGS) -o gaussianfield
powerspectrum: powerspectrum.cpp tools.cpp
	$(CC) powerspectrum.cpp tools.cpp $(CFLAGS) -o powerspectrum
bispectrum: bispectrum.cpp tools.cpp
	$(CC) bispectrum.cpp tools.cpp $(CFLAGS) -o bispectrum
bispectrum_custom_k: bispectrum_custom_k.cpp tools.cpp
	$(CC) bispectrum_custom_k.cpp tools.cpp $(CFLAGS) -o bispectrum_custom_k
#bispectrum_naive: bispectrum_naive.cpp tools.cpp
#	$(CC) bispectrum_naive.cpp tools.cpp $(CFLAGS) -o bispectrum_naive
fft3d: fft3d.cpp
	$(CC) fft3d.cpp $(CFLAGS) -o fft3d
isotropic_wavelet: isotropic_wavelet.cpp WC_tools.cpp
	$(CC) isotropic_wavelet.cpp WC_tools.cpp $(CFLAGS) -o isotropic_wavelet
clean:
	$(RM) fullrun gaussianfield powerspectrum bispectrum fft3d bispectrum_custom_k isotropic_wavelet
