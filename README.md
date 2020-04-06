# Gaussian_Field
Simple Gaussian Random Field generator and naive Power/Bi spectrum estimator.

## Prerequisite
-g++ compiler
-fftw3
-OpenMP for parallelization

-jupyter, numpy, matplotlib for Python interface

## Installation
Run the following in the unzipped folder(where the Makefile is)

    make


## Run Example
If doPk is 1, the powerspectrum is recomputed from the field, for sanity check.

n_thread is the number of OpenMP thread to use

The Bispectrum is a naive estimator, which is extremely slow.


    ./gaussianfield -nside BOXSIDE -filenamek NAMETOSAVE_delta_k -filenamer NAMETOSAVE_delta_k -doPk 1 -filenamePk NAMETOSAVE_Pk -doBk 1 -filenameBk NAMETOSAVE_Bk -realspace -quiet -n_thread

## Contact

cfpark00@gmail.com
