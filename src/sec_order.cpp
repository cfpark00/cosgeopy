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
#include <numeric>

#include <sys/time.h>

#include "WC_tools.h"

/*
#ifdef DOUBLE   #do this later
typedef double var_t;
#else
typedef float var_t;
#endif
*/

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

#include <typeinfo>
int main(int argc, char *argv[]){
    
    gettimeofday(&now,NULL);
    t0=now.tv_sec+0.000001*now.tv_usec;
    
    int nside=256;
	int i=1;
	std::string filename("");
	std::string filename_filt("");
    std::string filenameScatind("./data/wcind.dat");
    std::string filenameScat("./data/wc.dat");
    std::string filenameScatcount("./data/wccount.dat");
	bool quiet=0;
    bool full=0;
    int numFilts=0;
    while (i<argc){
    	if (!strcmp(argv[i],"-quiet")) quiet=1;
    	else if (!strcmp(argv[i],"-nside")) nside = atoi(argv[++i]);
    	else if (!strcmp(argv[i],"-filename")) filename = argv[++i];
        else if (!strcmp(argv[i],"-full")) full=1;
    	else if (!strcmp(argv[i],"-filename_filt")) filename_filt = argv[++i];
    	else if (!strcmp(argv[i],"-numFilts")) numFilts=atoi(argv[++i]);
        else if (!strcmp(argv[i],"-n_thread")) omp_set_num_threads(atoi(argv[++i]));
        else if (!strcmp(argv[i],"-filenameScat")) filenameScat = argv[++i];
        else if (!strcmp(argv[i],"-filenameScatind")) filenameScatind = argv[++i];
        else if (!strcmp(argv[i],"-filenameScatcount")) filenameScatcount = argv[++i];
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
        std::cout<<"From real filename="<<filename<<std::endl;
    }


    /*------Start----*/
    timer("Start program at: ",quiet);

    int size=nside*nside*nside;
    int csize=nside*nside*(nside/2+1);

    timer("Start Read: ",quiet);
    double *delta;
    delta = (double*) fftw_malloc(sizeof(double)*size);
    std::ifstream delta_in;
    delta_in.open(filename);
    delta_in.read((char*)delta,sizeof(double)*size);
    delta_in.close();

    double* filters;
    filters=(double*) fftw_malloc(sizeof(double)*numFilts*csize);
    std::ifstream filt_in;
    filt_in.open(filename_filt);
    filt_in.read((char*)filters,sizeof(double)*numFilts*csize);
    filt_in.close();

    
    
	std::cout<<std::endl<<numFilts<<" filters assumed ordered"<<std::endl;
    int fsize=1+numFilts+numFilts*numFilts;//(numFilts*(numFilts-1))/2;

    int* Scatind=new int[fsize*2];
    double* Scat=new double[fsize];
    double* Scatcount=new double[fsize];

    timer("Start WC: ",quiet);
    get_2nd_order(Scatind,Scat,Scatcount,nside,fsize,delta,filters,numFilts,full,quiet);

    delete[] filters;

    timer("Make Stat at: ",quiet);
    if (!quiet) std::cout<<std::endl<<"  Computed "<<fsize<<" coefficients in "<<dt<<". "<<(dt/fsize)<<" per coefficient"<<std::endl;//<<start<<step<<fsize<<""<<Bk[0]<<std::endl;
    timer("Start WC write at: ",quiet);
    std::ofstream fileScat(filenameScat);
    fileScat.write((char*)Scat, sizeof(double)*fsize);
    fileScat.close();
    std::ofstream fileScatI(filenameScatind);
    fileScatI.write((char*)Scatind, sizeof(int)*fsize*2);
    fileScatI.close();
    std::ofstream fileScatC(filenameScatcount);
    fileScatC.write((char*)Scatcount, sizeof(int)*fsize);
    fileScatC.close();
    delete[] Scat;
    delete[] Scatind;
    delete[] Scatcount;

    if (!quiet){
        timer("\nRun successful. ",quiet);
        gettimeofday(&now,NULL);
        std::cout<<std::endl<<"--------------"<<std::endl;
        std::cout<<std::endl<<"Total Time: "<<now.tv_sec+0.000001*now.tv_usec-t0<<std::endl;
    }




	return 0;
}