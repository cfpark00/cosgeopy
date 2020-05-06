#ifdef _OPENMP
#include <omp.h>
#endif

#include <string.h>
#include <assert.h>
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include<complex.h>
#include <fftw3.h>
#include <vector>


#include <sys/time.h>

#include "WC_tools.h"
struct timeval now;
double tm1;
double t;
double dt;
double t0;

void timer(const std::string& Message,bool quiet){
    gettimeofday(&now,NULL);
    t=now.tv_sec+0.000001*now.tv_usec-t0;
    dt=(t-tm1);
    if (!quiet) std::cout<<" Time: "<<dt<<std::endl<<Message<<t;
    tm1=t;
}

int* meta_size(int order, int numJs){
	int* meta;
	meta=(int*) malloc(sizeof(int)*3);
	int indsize=0;//for the mean
	int count=0;//for the mean
	int maxbat=0;
	for(int m=0;m<order+1;m++){
		int quo=1;
		int div=1;
		for(int i=0;i<m;i++){
			quo*=(numJs-i);
			div*=(i+1);
		}
		if ((quo/div)>maxbat) maxbat=(quo/div);
		indsize+=(quo/div)*m;//m indice for order m
		count+=(quo/div);
	}
	meta[2]=maxbat;
	meta[1]=count;
	meta[0]=indsize;
	return meta;
}

#include <typeinfo>
int main(int argc, char *argv[]){
    
    gettimeofday(&now,NULL);
    t0=now.tv_sec+0.000001*now.tv_usec;
    
    int nside=256;
	int i=1;
	std::string filename("");
    std::string filenameWCind("./data/wcind.dat");
    std::string filenameWC("./data/wc.dat");
    double sigma=1;
	bool quiet=0;
    bool fromkspace=0;
    bool save_memory=1;
    int order=2;
    while (i<argc){
    	if (!strcmp(argv[i],"-quiet")) quiet=1;
    	else if (!strcmp(argv[i],"-nside")) nside = atoi(argv[++i]);
    	else if (!strcmp(argv[i],"-sigma")) sigma = atof(argv[++i]);
    	else if (!strcmp(argv[i],"-order")) order = atoi(argv[++i]);
    	else if (!strcmp(argv[i],"-filename")) filename = argv[++i];
        else if (!strcmp(argv[i],"-fromkspace")) fromkspace=1;
        else if (!strcmp(argv[i],"-not_save_memory")) save_memory=0;
        else if (!strcmp(argv[i],"-n_thread")) omp_set_num_threads(atoi(argv[++i]));
        else if (!strcmp(argv[i],"-filenameWC")) filenameWC = argv[++i];
        else if (!strcmp(argv[i],"-filenameWCind")) filenameWCind = argv[++i];
        else {
            std::cout<<std::endl<<"Error in arguments"<<std::endl;
            assert(0);
        }
		i++;
    }
    assert(nside>0);
    assert(((nside+1)%2)&&"Only even sides accepted.");

    if (!quiet){
        std::cout<<"Starting Program"<<std::endl;
        std::cout<<"--------------"<<std::endl;
        std::cout<<std::endl<<"nside="<<nside<<std::endl;
        if (fromkspace) std::cout<<"Will compute from k-space: "<<filename<<std::endl;
        else std::cout<<"From real filename="<<filename<<std::endl;
    }

    /*------Start----*/
    timer("Start program at: ",quiet);

    int size=nside*nside*nside;
    int csize=nside*nside*(nside/2+1);

    double *delta;
    delta = (double*) fftw_malloc(sizeof(double)*size);
    if (fromkspace){
        timer("Start Fourier Transform: ",quiet);

    	fftw_complex *delta_k;
    	delta_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*csize);
        std::ifstream filek;
        filek.open(filename);
        filek.read((char*)delta_k, sizeof(fftw_complex)*csize);
        filek.close();

        #ifdef _OPENMP
        fftw_init_threads();
        #endif
        fftw_plan p;
        #ifdef _OPENMP
        fftw_plan_with_nthreads(omp_get_max_threads());
        #endif
        p = fftw_plan_dft_c2r_3d(nside,nside,nside, delta_k, delta,FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
        free(delta_k);

        //rescale
        for(int i=0;i<size;i++) delta[i]/=size;

    }
    else{
        std::ifstream delta_in;
        delta_in.open(filename);
        delta_in.read((char*)delta,sizeof(double)*size);
        delta_in.close();
    }

    timer("Start WC: ",quiet);
    
    int numJs=0;
    int twototheJ=1;
    while (twototheJ* 2 <= nside) {
    	twototheJ*= 2;
    	numJs+=1;
	}
	std::cout<<std::endl<<"J from 0 to "<<numJs-1<<std::endl;
	assert((numJs>=order)&&("Order too high"));
    int* fsize=meta_size(order,numJs);
    //std::cout<<fsize[0]<<" "<<fsize[1]<<std::endl;

    int* WCind=new int[fsize[0]];
    double* WC=new double[fsize[1]];

    getisoWC(WCind,WC,nside,fsize,delta,order,numJs,sigma,save_memory,quiet);
    timer("Make Stat at: ",quiet);
    if (!quiet) std::cout<<std::endl<<"  Computed "<<fsize[1]<<" coefficients in "<<dt<<". "<<(dt/fsize[1])<<" per coefficient"<<std::endl;//<<start<<step<<fsize<<""<<Bk[0]<<std::endl;
    timer("Start WC write at: ",quiet);
    std::ofstream fileWC(filenameWC);
    fileWC.write((char*)WC, sizeof(double)*fsize[1]);
    fileWC.close();
    std::ofstream fileWCI(filenameWCind);
    fileWCI.write((char*)WCind, sizeof(int)*fsize[0]);
    fileWCI.close();
    delete[] WC;
    delete[] WCind;

    if (!quiet){
        timer("\nRun successful. ",quiet);
        gettimeofday(&now,NULL);
        std::cout<<std::endl<<"--------------"<<std::endl;
        std::cout<<std::endl<<"Total Time: "<<now.tv_sec+0.000001*now.tv_usec-t0<<std::endl;
    }




	return 0;
}