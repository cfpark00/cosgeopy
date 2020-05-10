#define INT_MAX 32768
#define INT_MAXdiv2pi 5215.189175235227//INT_MAX/(2pi)
#define sqrtEightPI3 15.749609945722419
#include<complex.h>
#include <fftw3.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <numeric>
#include <map>

#include <iostream>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif


double sum_all(double* input,int num){
    double res=0.0;
    #pragma omp parallel for reduction(+: res)
    for(int sumind=0;sumind<num;sumind++){
        res+=input[sumind];
    }

    return res;
}

void scatter(fftw_plan p,fftw_plan ip,double* tempr,fftw_complex* tempk,double* input,double* filter,double* res,int nside){
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int size=nside*nside*nside;
    //int csize=nside*nside*middleplus1;

    #pragma omp parallel for shared(tempr)
    for (int i=0;i<size;i++) tempr[i]=input[i];

    fftw_execute(p);

    
    #pragma omp parallel for shared(tempk)
    for (int i=0;i<nside;i++){
        int ind;
        for (int j=0;j<nside;j++){
            for(int k=0;k<middleplus1;k++){
                ind=yzsize*i+middleplus1*j+k;
                tempk[ind][0]=filter[ind]*tempk[ind][0]/size;
                tempk[ind][1]=filter[ind]*tempk[ind][1]/size;
            }
        }
    }


/*
    #pragma omp parallel for shared(tempk)
    for (int i=0;i<csize;i++){
        tempk[i][0]=filter[i]*tempk[i][0]/size;
        tempk[i][1]=filter[i]*tempk[i][1]/size;
    }
*/
    fftw_execute(ip);

    #pragma omp parallel for shared(res)
    for (int i=0;i<size;i++) res[i]=abs(tempr[i]);
}

void scatter_norm(fftw_plan p,fftw_plan ip,double* tempr,fftw_complex* tempk,double* input_norm,double* filter,double* res_norm,int nside){
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int size=nside*nside*nside;
    //int csize=nside*nside*middleplus1;

    #pragma omp parallel for shared(tempr)
    for (int i=0;i<size;i++) tempr[i]=input_norm[i];

    fftw_execute(p);

    
    #pragma omp parallel for shared(tempk)
    for (int i=0;i<nside;i++){
        int ind;
        for (int j=0;j<nside;j++){
            for(int k=0;k<middleplus1;k++){
                ind=yzsize*i+middleplus1*j+k;
                tempk[ind][0]=filter[ind]*tempk[ind][0]/size;
                tempk[ind][1]=filter[ind]*tempk[ind][1]/size;
            }
        }
    }


/*
    #pragma omp parallel for shared(tempk)
    for (int i=0;i<csize;i++){
        tempk[i][0]=filter[i]*tempk[i][0]/size;
        tempk[i][1]=filter[i]*tempk[i][1]/size;
    }
*/
    fftw_execute(ip);

    #pragma omp parallel for shared(res_norm)
    for (int i=0;i<size;i++) res_norm[i]=abs(tempr[i]);
}

void downscatter(fftw_plan p,fftw_plan ip,double* tempr,fftw_complex* tempk,double* WC,double* WCcount, double* got, double* got_norm,double* filters,int order,int numJs,int nside,int m,int parent_j,int& count){
    int csize=(nside/2+1)*(nside)*(nside);
    int size=nside*nside*nside;
    if ((m==order)||(parent_j==numJs-1)){
        WC[count]=sum_all(got,size);
        WCcount[count]=sum_all(got_norm,size);
        //std::cout<<"endp "<<count<<" "<<WC[count]<<std::endl;
        count++;
    }
    else{
        WC[count]=sum_all(got,size);
        WCcount[count]=sum_all(got_norm,size);
        //std::cout<<got[0]<<" "<<got[1]<<" "<<got[2]<<" "<<got[3]<<" "<<std::endl;
        //std::cout<<"midp "<<count<<" "<<WC[count]<<std::endl;
        count++;

        double* res;
        res=(double*) fftw_malloc(sizeof(double)*size);
        double* res_norm;
        res_norm=(double*) fftw_malloc(sizeof(double)*size);

        for(int j=parent_j+1;j<numJs;j++){
            scatter(p,ip,tempr,tempk,got,filters+csize*j,res,nside);
            scatter_norm(p,ip,tempr,tempk,got_norm,filters+csize*j,res_norm,nside);
            downscatter(p,ip,tempr,tempk,WC,WCcount,res,res_norm,filters,order,numJs,nside,m+1,j,count);  
        }

        free(res);
        free(res_norm);
    }
}

void getisoWC(int* WCind,double* WC,double* WCcount,int nside,int fsize,double* delta,int order,int numJs,double sigma,bool save_memory,bool quiet){
    if (!quiet) std::cout<<std::endl<<"  Generate k-space filters"<<std::endl;
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int csize=(nside/2+1)*(nside)*(nside);
    int size=nside*nside*nside;
    double nside2=nside*nside;
    double sigma2=sigma*sigma;

    //Make the -space filters
    double* filters=new double[numJs*csize];

    /*----------------------------MOVE J LOOP IN------------------------*/
    #pragma omp parallel for shared(filters)
    for(int J=0;J<numJs;J++){ //avoid confusion with j!!!!
        for (int i=0;i<nside;i++){
            int ii,jj,ind;
            double kabs,kabs2,lambda,lambda2,fac;
            if (i<middleplus1) ii=i*i;
            else ii=(i-nside)*(i-nside);
            for (int j=0;j<nside;j++){
                if (j<middleplus1) jj=j*j;
                else jj=(j-nside)*(j-nside);
                for(int k=0;k<middleplus1;k++){
                    kabs2=(ii/nside2)+(jj/nside2)+(k*k/nside2);
                    kabs=sqrt(kabs2);
                    ind=yzsize*i+middleplus1*j+k;
                    lambda=pow(2,-J);
                    lambda2=lambda*lambda;

                    fac=kabs/lambda;
                    if (fac==0) fac=1;
                    else fac=sinh(fac*sigma2)/(fac*sigma2);
                    filters[J*csize+ind]=(sqrtEightPI3/lambda)*exp(-(lambda2+kabs2)*sigma2/(2*lambda2))*2*fac;

                }   
            }
        }
    }
    /*
    std::ofstream file_filt;
    file_filt.open("./data/filters.dat");
    file_filt.write((char*)filters,sizeof(double)*numJs*csize);
    file_filt.close();
    */

    if (!quiet) std::cout<<std::endl<<"  Scatter"<<std::endl;
    if(save_memory){
        #ifdef _OPENMP
        fftw_init_threads();
        #endif
        fftw_plan ip;
        fftw_plan p;
        #ifdef _OPENMP
        fftw_plan_with_nthreads(omp_get_max_threads());
        #endif

        double* tempr;
        tempr=(double*) fftw_malloc(sizeof(double)*size);
        fftw_complex* tempk;
        tempk=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*csize);

        p = fftw_plan_dft_r2c_3d(nside,nside,nside, tempr, tempk,FFTW_ESTIMATE);
        ip = fftw_plan_dft_c2r_3d(nside,nside,nside, tempk, tempr,FFTW_ESTIMATE);


        //DEPTH FIRST IMPLEMENTATION
        int count=0;

        double* normfield;
        normfield=(double*) fftw_malloc(sizeof(double)*size);
        for(int i=0;i<size;i++){
            if (i==0) normfield[i]=1.0;
            else normfield[i]=0.0;
        }

        downscatter(p,ip,tempr,tempk,WC,WCcount,delta,normfield,filters,order,numJs,nside,0,-1,count);
        assert((count==fsize)&&("this should match"));

        for(int i=0;i<count;i++){
            if (WCcount[i]!=0) WC[i]/=WCcount[i];
        }

        free(normfield);
    }
    else{
        assert(0&&("Not Implemented"));
    }

}

void getisoWC_loadfilt(int* WCind,double* WC,double* WCcount,int nside,int fsize,double* delta,double* filters,int order,int numJs,bool quiet){
    int csize=(nside/2+1)*(nside)*(nside);
    int size=nside*nside*nside;


    if (!quiet) std::cout<<std::endl<<"  Scatter"<<std::endl;

    #ifdef _OPENMP
    fftw_init_threads();
    #endif
    fftw_plan ip;
    fftw_plan p;
    #ifdef _OPENMP
    fftw_plan_with_nthreads(omp_get_max_threads());
    #endif

    double* tempr;
    tempr=(double*) fftw_malloc(sizeof(double)*size);
    fftw_complex* tempk;
    tempk=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*csize);

    p = fftw_plan_dft_r2c_3d(nside,nside,nside, tempr, tempk,FFTW_ESTIMATE);
    ip = fftw_plan_dft_c2r_3d(nside,nside,nside, tempk, tempr,FFTW_ESTIMATE);


    //DEPTH FIRST IMPLEMENTATION
    int count=0;

    double* normfield;
    normfield=(double*) fftw_malloc(sizeof(double)*size);
    for(int i=0;i<size;i++){
        if (i==0) normfield[i]=1.0;
        else normfield[i]=0.0;
    }

    downscatter(p,ip,tempr,tempk,WC,WCcount,delta,normfield,filters,order,numJs,nside,0,-1,count);
    assert((count==fsize)&&("this should match"));


    for(int i=0;i<count;i++){
        if (WCcount[i]!=0) WC[i]/=WCcount[i];
    }

    free(normfield);

}
