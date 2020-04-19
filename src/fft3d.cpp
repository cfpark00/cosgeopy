#ifdef _OPENMP
#include <omp.h>
#endif

#include <string.h>
#include <assert.h>
#include <iostream>
//#include <sys/stat.h>
#include <fstream>
//#include <algorithm>
#include<complex.h>
#include <fftw3.h>

int main(int argc, char *argv[]){
    int nside=256;
	int i=1;
	std::string filename("");
    std::string filenamek("");
    bool inverse=0;
    while (i<argc){
    	if (!strcmp(argv[i],"-inverse")) inverse=1;
    	else if (!strcmp(argv[i],"-nside")) nside = atoi(argv[++i]);
    	else if (!strcmp(argv[i],"-filename")) filename = argv[++i];
        else if (!strcmp(argv[i],"-n_thread")) omp_set_num_threads(atoi(argv[++i]));
        else if (!strcmp(argv[i],"-filenamek")) filenamek = argv[++i];
        else {
            std::cout<<std::endl<<"Error in arguments"<<std::endl;
            assert(0);
        }
		i++;
    }
    assert(nside>0);
    assert(((nside+1)%2)&&"Only even sides accepted.");

    int size=nside*nside*nside;
    int csize=nside*nside*(nside/2+1);

    fftw_complex *delta_k;
    delta_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*csize);
    double *delta;
    delta = (double*) fftw_malloc(sizeof(double)*size);

    std::ifstream filein;
    if (inverse){
        filein.open(filenamek);
        filein.read((char*)delta_k, sizeof(fftw_complex)*csize);
    }
    else{
        filein.open(filename);
        filein.read((char*)delta, sizeof(double)*size);
    }
    filein.close();

    #ifdef _OPENMP
    fftw_init_threads();
    #endif
    fftw_plan p;
    #ifdef _OPENMP
    fftw_plan_with_nthreads(omp_get_max_threads());
    #endif

    if (inverse) p = fftw_plan_dft_c2r_3d(nside,nside,nside, delta_k, delta,FFTW_ESTIMATE);
    else p = fftw_plan_dft_r2c_3d(nside,nside,nside, delta, delta_k,FFTW_ESTIMATE);

    fftw_execute(p);
    fftw_destroy_plan(p);

    std::ofstream fileout;
    if (inverse){
        fileout.open(filename);
        fileout.write((char*)delta, sizeof(double)*size);
    }
    else{
        fileout.open(filenamek);
        fileout.write((char*)delta_k, sizeof(fftw_complex)*csize);
    }
    fileout.close();


    free(delta);
    fftw_free(delta_k);

	return 0;
}