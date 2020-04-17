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


#include <sys/time.h>

#include "tools.h"
struct timeval now;
double tm1;
double t;
double t0;

void timer(const std::string& Message,bool quiet){
    gettimeofday(&now,NULL);
    t=now.tv_sec+0.000001*now.tv_usec-t0;
    if (!quiet) std::cout<<" Time: "<<(t-tm1)<<std::endl<<Message<<t;
    tm1=t;
}


inline int flatsize(int ny){
    return (ny+1)*(ny+2)*(ny+3)/6;
}

#include <typeinfo>
int main(int argc, char *argv[]){
    
    gettimeofday(&now,NULL);
    t0=now.tv_sec+0.000001*now.tv_usec;
    

    int nside=256;
	int i=1;
	std::string filenamek("./data/delta_k.dat");
	std::string filenamer("./data/delta.dat");
	std::string Pkfilename("");
    std::string filenamePk("./data/pk.dat");
    std::string filenameBk("./data/bk_naive.dat");
    std::string filenameBkind("./data/bkinds.dat");
	bool quiet=0;
	bool realspace=0;
    bool doPk=1;
    bool doBk=0;
    while (i<argc){
    	if (!strcmp(argv[i],"-quiet")) quiet=1;
    	else if (!strcmp(argv[i],"-Pkfilename")) Pkfilename=argv[++i];
    	else if (!strcmp(argv[i],"-nside")) nside = atoi(argv[++i]);
    	else if (!strcmp(argv[i],"-filenamek")) filenamek = argv[++i];
        else if (!strcmp(argv[i],"-filenameBk")) filenameBk = argv[++i];
        else if (!strcmp(argv[i],"-filenamePk")) filenamePk = argv[++i];
    	else if (!strcmp(argv[i],"-filenamer")) filenamer = argv[++i];
    	else if (!strcmp(argv[i],"-realspace")) realspace = 1;
        else if (!strcmp(argv[i],"-n_thread")) omp_set_num_threads(atoi(argv[++i]));
        else if (!strcmp(argv[i],"-doPk")) doPk=(strcmp(argv[++i],"0"));
        else if (!strcmp(argv[i],"-doBk")) doBk=(strcmp(argv[++i],"0"));
        else if (!strcmp(argv[i],"-filenameBkind")) filenameBkind = argv[++i];
        else {
            std::cout<<std::endl<<"Error in arguments"<<std::endl;
            assert(0);
        }
		i++;
    }
    assert(nside>0);
    assert(((nside+1)%2)&&"Only even sides accepted.");
    if (Pkfilename!=""){

    }

    if (!quiet){
        std::cout<<"Starting Program"<<std::endl;
        std::cout<<"--------------"<<std::endl;
		std::cout<<std::endl<<"nside="<<nside<<std::endl<<"filenamek="<<filenamek<<std::endl;
    	if (realspace) std::cout<<"will also save realspace as filenamer="<<filenamer<<std::endl;
    }

    /*------Start----*/    
    //MPI_Bcast(&nside,1, MPI_INT,0,MPI_COMM_WORLD);
    //para_range(0,n-1,nproc,iproc,ista,iend);
    //loc_dim = iend - ista + 1;
    timer("Start program at: ",quiet);

    int size=nside*nside*nside;
    int csize=nside*nside*(nside/2+1);

    

    fftw_complex *delta_k;
    delta_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*csize);
    
    timer("Start filling at: ",quiet);
    fill_delta_k(delta_k,nside,10);

    timer("Start delta_k write at: ",quiet);
	std::ofstream filek;
	filek.open(filenamek);
	filek.write((char*)delta_k, sizeof(fftw_complex)*csize);
	filek.close();

    if (doPk){
        timer("Start Pk: ",quiet);
        double* Pk=new double[nside/2+1];
        memset(Pk, 0,(nside/2+1)*sizeof(double));
        getPk(Pk,delta_k,nside);
        std::ofstream filep;
        filep.open(filenamePk);
        filep.write((char*)Pk, sizeof(double)*(nside/2+1));
        filep.close();
        free(Pk);
    }

    if (doBk){
        timer("Start Bk: ",quiet);
        double* Bk=new double[flatsize(nside/2)];
        //std::cout<<" "<<flatsize(nside/2)<<std::endl;
        int* Bkind=getBk_ind(nside);
        memset(Bk, 0,flatsize(nside/2)*sizeof(double));
        getBk_naive(Bk,delta_k,nside);
        //getBkangav_naive(Bk,delta_k,nside);
        timer("Start Bk write at: ",quiet);
        std::ofstream fileb;
        fileb.open(filenameBk);
        fileb.write((char*)Bk, sizeof(double)*flatsize(nside/2));
        fileb.close();

        std::ofstream filebi;
        filebi.open(filenameBkind);
        filebi.write((char*)Bkind, sizeof(int)*3*flatsize(nside/2));
        filebi.close();
        free(Bk);
    }


	if (realspace){
        timer("Start FFT at: ",quiet);
        double *delta;
        delta = (double*) fftw_malloc(sizeof(double)*size);
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

        timer("Start delta write at: ",quiet);
		std::ofstream filer;
		filer.open(filenamer);
		filer.write((char*)delta, sizeof(double)*size);
		filer.close();
        fftw_free(delta);
    }
    //MPI_Finalize();

    timer("Start Closing: ",quiet);
    fftw_free(delta_k);

    if (!quiet){
        timer("\nRun successful. ",quiet);
        gettimeofday(&now,NULL);
        std::cout<<std::endl<<"--------------"<<std::endl;
        std::cout<<std::endl<<"Total Time: "<<now.tv_sec+0.000001*now.tv_usec-t0<<std::endl;
    }




	return 0;
}