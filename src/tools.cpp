#define INT_MAX 32768
#define INT_MAXdiv2pi 5215.189175235227//INT_MAX/(2pi)
#include<complex.h>
#include <fftw3.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <numeric>

#include <iostream>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif


static unsigned int g_seed;
inline void fast_srand(int seed) {
    g_seed = seed;
}

inline int fastrand() { 
  g_seed = (214013*g_seed+2531011); 
  return (g_seed>>16)&0x7FFF; 
} 

inline double getmag(double k) {
    return 4*(pow(sin(k/4),2)/pow((k+1),2));
}

void fill_delta_k(fftw_complex* delta_k,int nside,int seed){
    fast_srand(seed);
    int csize=nside*nside*(nside/2+1);
    double *phases=new double[csize];
    for (int i=0;i<csize;i++) phases[i]=((double)fastrand())/INT_MAXdiv2pi+phases[i];

    int middle=nside/2;
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ind=0;
        double mag;
        double k_abs;
        int ii,jj;

        if (i<middleplus1){
            ii=i*i;
        }else{
            ii=(nside-i)*(nside-i);
        }

        for (int j=0;j<nside;j++){

            if (j<middleplus1){
                jj=j*j;
            }else{
                jj=(nside-j)*(nside-j);
            }

            for(int k=0;k<middleplus1;k++){
                ind=yzsize*i+middleplus1*j+k;
                k_abs=sqrt(ii+jj+k*k);
                mag=getmag(k_abs);
                delta_k[ind][0]=mag*cos(phases[ind]);
                delta_k[ind][1]=mag*sin(phases[ind]);
            }
        }
    }

    int ii,jj;
    for (int i=1;i<middleplus1;i++){

        ii=yzsize*(nside-i);
        jj=middleplus1*(nside-i);
        //zeroline
        delta_k[ii+0+0][1]=-delta_k[yzsize*i+0+0][1];
        delta_k[ii+0+0][0]=delta_k[yzsize*i+0+0][0];
        delta_k[ii+0+middle][1]=-delta_k[yzsize*i+0+middle][1];
        delta_k[ii+0+middle][0]=delta_k[yzsize*i+0+middle][0];

        delta_k[0+jj+0][1]=-delta_k[0+i*middleplus1+0][1];
        delta_k[0+jj+0][0]=delta_k[0+i*middleplus1+0][0];
        delta_k[0+jj+middle][1]=-delta_k[0+i*middleplus1+middle][1];
        delta_k[0+jj+middle][0]=delta_k[0+i*middleplus1+middle][0];

        //plane
        for (int j=1;j<middleplus1;j++){
            delta_k[ii+middleplus1*(nside-j)+0][1]=-delta_k[yzsize*i+middleplus1*j+0][1];
            delta_k[ii+middleplus1*(nside-j)+0][0]=delta_k[yzsize*i+middleplus1*j+0][0];
            delta_k[ii+middleplus1*(nside-j)+middle][1]=-delta_k[yzsize*i+middleplus1*j+middle][1];
            delta_k[ii+middleplus1*(nside-j)+middle][0]=delta_k[yzsize*i+middleplus1*j+middle][0];
        }
    }

    for (int i=middleplus1;i<nside;i++){
        ii=yzsize*(nside-i);
        //plane
        for (int j=1;j<middleplus1;j++){
            delta_k[ii+middleplus1*(nside-j)+0][1]=-delta_k[yzsize*i+middleplus1*j+0][1];
            delta_k[ii+middleplus1*(nside-j)+0][0]=delta_k[yzsize*i+middleplus1*j+0][0];
            delta_k[ii+middleplus1*(nside-j)+middle][1]=-delta_k[yzsize*i+middleplus1*j+middle][1];
            delta_k[ii+middleplus1*(nside-j)+middle][0]=delta_k[yzsize*i+middleplus1*j+middle][0];
        }
    }


    delta_k[yzsize*0+middleplus1*0+0][0]=0;
    delta_k[yzsize*0+middleplus1*0+0][1]=0;
    delta_k[yzsize*0+middleplus1*0+middle][0]=getmag(middle);
    delta_k[yzsize*0+middleplus1*0+middle][1]=0;
    delta_k[yzsize*0+middleplus1*middle+0][0]=getmag(middle);
    delta_k[yzsize*0+middleplus1*middle+0][1]=0;
    delta_k[yzsize*0+middleplus1*middle+middle][0]=getmag(sqrt(2)*middle);
    delta_k[yzsize*0+middleplus1*middle+middle][1]=0;
    delta_k[yzsize*middle+middleplus1*0+0][0]=getmag(middle);
    delta_k[yzsize*middle+middleplus1*0+0][1]=0;
    delta_k[yzsize*middle+middleplus1*0+middle][0]=getmag(sqrt(2)*middle);
    delta_k[yzsize*middle+middleplus1*0+middle][1]=0;
    delta_k[yzsize*middle+middleplus1*middle+0][0]=getmag(sqrt(2)*middle);
    delta_k[yzsize*middle+middleplus1*middle+0][1]=0;
    delta_k[yzsize*middle+middleplus1*middle+middle][0]=getmag(sqrt(3)*middle);
    delta_k[yzsize*middle+middleplus1*middle+middle][1]=0;
    
}

void getPk(double* Pk,fftw_complex* delta_k,int nside){
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int* counts=new int[nside/2+1];
    memset(counts, 0,(nside/2+1)*sizeof(int));

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ind=0;
        int k_abs_ind;
        int ii,jj;
        if (i<middleplus1){
            ii=i*i;
        }else{
            ii=(nside-i)*(nside-i);
        }
        for (int j=0;j<nside;j++){
            if (j<middleplus1){
                jj=j*j;
            }else{
                jj=(nside-j)*(nside-j);
            }
            for(int k=0;k<middleplus1;k++){
                ind=yzsize*i+middleplus1*j+k;
                k_abs_ind=(int)(sqrt(ii+jj+k*k)+0.5);
                if (k_abs_ind<middleplus1){
                    Pk[k_abs_ind]+=delta_k[ind][0]*delta_k[ind][0]+delta_k[ind][1]*delta_k[ind][1];
                    counts[k_abs_ind]+=1;
                }
            }
        }
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<middleplus1;i++){
        if (counts[i]!=0) Pk[i]/=counts[i];
    }

    free(counts);
}


inline int to1dind(int a1,int a2, int a3, int d){
    return a1*(a1+1)*(a1+2)/6+(a2)*(a2+1)/2+a3;
}

void getBk(double* Bk,fftw_complex* delta_k,int nside,int* ks,int numks){
    std::cout<<"T0"<<std::endl;
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int csize=(nside/2+1)*(nside)*(nside);
    int size=nside*nside*nside;
    int ny=nside/2;

    //Initialize the kbins
    if (ks==NULL){
        ks=(int*) malloc(sizeof(int)*(nside/2+1));
        numks=nside/2+1;
        for(int i=0;i<numks;i++) ks[i]=i;
    }
    else assert(0&&("Not Implemented"));

    //Make the k-space rings
    fftw_complex *partialdelta_ks;
    partialdelta_ks = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(nside/2+1)*csize);
    memset(partialdelta_ks, 0,sizeof(fftw_complex)*numks*csize);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ii,jj,kabs,ind;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabs=(int)(sqrt(ii+jj+k*k)+0.5);
                if (kabs<numks){
                    ind=yzsize*i+middleplus1*j+k;
                    partialdelta_ks[kabs*csize+ind][0]+=delta_k[ind][0];
                    partialdelta_ks[kabs*csize+ind][1]+=delta_k[ind][1];
                    //memcpy(partialdelta_ks+kabs*csize+yzsize*i+middleplus1*j+k,delta_k+yzsize*i+middleplus1*j+k,sizeof(fftw_complex));
                }     
            }
        }
    }
    /*
    std::cout<<"HI3"<<std::endl;
    std::ofstream file;
    file.open("TEST");
    file.write((char*)partialdelta_ks, sizeof(fftw_complex)*numks*csize);
    file.close();
    free(partialdelta_ks);
    */

    std::cout<<"T1"<<std::endl;
    double* partialdeltas=new double[(nside/2+1)*size];
    double* partialcounts=new double[(nside/2+1)*size];

    int n[]={nside,nside,nside};
    
    #ifdef _OPENMP
    fftw_init_threads();
    #endif
    fftw_plan p;
    #ifdef _OPENMP
    fftw_plan_with_nthreads(omp_get_max_threads());
    #endif
    p=fftw_plan_many_dft_c2r(3,n,numks
        ,partialdelta_ks,NULL,1,csize
        ,partialdeltas,NULL,1,size
        ,FFTW_ESTIMATE);
    fftw_execute(p);
    std::cout<<"T2"<<std::endl;
    //Now the counts reuse same array for memory
    memset(partialdelta_ks, 0,sizeof(fftw_complex)*numks*csize);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ii,jj,kabs;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabs=(int)(sqrt(ii+jj+k*k)+0.5);
                if (kabs<numks){
                    partialdelta_ks[kabs*csize+yzsize*i+middleplus1*j+k][0]=1;
                }     
            }
        }
    }

    p=fftw_plan_many_dft_c2r(3,n,numks
        ,partialdelta_ks,NULL,1,csize
        ,partialcounts,NULL,1,size
        ,FFTW_ESTIMATE);
    fftw_execute(p);

    //free
    fftw_destroy_plan(p);
    free(partialdelta_ks);
    std::cout<<"T3"<<std::endl;

    

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int i=0;i<numks;i++){
        //int multi=1;
        double deltasum,countsum;
        int ind;
        double* partialproddelta=new double[size];
        double* partialprodcount=new double[size];
        for(int j=0;j<i+1;j++){

            for(int sumind=0;sumind<size;sumind++){
                partialproddelta[sumind]=partialdeltas[i*size+sumind]*partialdeltas[j*size+sumind];
                partialprodcount[sumind]=partialcounts[i*size+sumind]*partialcounts[j*size+sumind];
            }

            for(int k=i-j;k<j+1;k++){
                /*
                if (i==j){
                    if (j==k) multi=6;
                    else multi=2;
                }
                else if ((j==k)or(i==k)) multi=2;
                */
                deltasum=0;
                countsum=0;
                for(int sumind=0;sumind<size;sumind++){
                    deltasum+=partialproddelta[sumind]*partialdeltas[k*size+sumind];
                    countsum+=partialprodcount[sumind]*partialcounts[k*size+sumind];
                }
                ind=to1dind(ks[i],ks[j],ks[k],ny);
                if (countsum!=0) Bk[ind]+=deltasum/countsum;

            }
        }
        free(partialproddelta);
        free(partialprodcount);
    }
    std::cout<<"T4"<<std::endl;



    /*
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
    */
/*
    int* counts=new int[(nside/2+1)*(nside/2+1)];
    int* kabsarr=new int[nside*nside*(nside/2+1)];
    memset(counts, 0,(nside/2+1)*(nside/2+1)*sizeof(int));

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ii,jj;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabsarr[yzsize*i+middleplus1*j+k]=(int)(sqrt(ii+jj+k*k)+0.5);
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i1=0;i1<nside;i1++){
        int ind1,ind2,ind3;
        int k_abs1,k_abs2;
        int ii1,jj1,ii2,jj2,ii3,jj3;
        if (i1<middleplus1){
            ii1=i1;
        }else{
            ii1=(i1-nside);
        }

        for (int j1=0;j1<nside;j1++){
            if (j1<middleplus1){
                jj1=j1;
            }else{
                jj1=(j1-nside);
            }
            if (kabsarr[yzsize*i1+middleplus1*j1+0]>middleplus1) continue;

            for(int k1=0;k1<middleplus1;k1++){
                ind1=yzsize*i1+middleplus1*j1+k1;
                k_abs1=kabsarr[yzsize*i1+middleplus1*j1+k1];

                if (k_abs1>middleplus1) continue;
                for (int i2=0;i2<nside;i2++){
                    if (i2<middleplus1){
                        ii2=i2;
                    }else{
                        ii2=(i2-nside);
                    }
                    for (int j2=0;j2<nside;j2++){
                        if (j2<middleplus1){
                            jj2=j2;
                        }else{
                            jj2=(j2-nside);
                        }


                        for(int k2=0;k2<middleplus1;k2++){
                            ind2=yzsize*i2+middleplus1*j2+k2;
                            ii3=ii1+ii2;
                            if ((ii3<-middleplus1)||(ii3>middleplus1)) continue;
                            if (ii3<0){
                                ii3=nside+ii3;
                            }
                            jj3=jj1+jj2;
                            if ((jj3<-middleplus1)||(jj3>middleplus1))  continue;
                            if (jj3<0){
                                jj3=nside+jj3;
                            }
                            if ((k1+k2)>middleplus1) continue;


                            ind3=yzsize*(ii3)+middleplus1*(jj3)+(k1+k2);
                            
                            k_abs2=kabsarr[yzsize*i2+middleplus1*j2+k2];
                            if ((k_abs1+k_abs2)<middleplus1){
                                Bk[k_abs1*middleplus1+k_abs2]+=delta_k[ind3][0]*(delta_k[ind1][0]*delta_k[ind2][0]-delta_k[ind1][1]*delta_k[ind2][1])+delta_k[ind3][1]*(delta_k[ind1][0]*delta_k[ind2][1]+delta_k[ind1][1]*delta_k[ind2][0]);
                                counts[k_abs1*middleplus1+k_abs2]+=1;
                            }

                        }
                    }
                }
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<middleplus1;i++){
        for (int j=0;j<middleplus1;j++){
            if (counts[i*middleplus1+j]!=0){
                Bk[i*middleplus1+j]/=counts[i*middleplus1+j];
            }
        }
    }

    free(counts);
    free(kabsarr);
    */
}

inline int flatsize(int ny){
    return (ny+1)*(ny+2)*(ny+3)/6;
}

int* getBk_ind(int nside){
    int ny=nside/2;
    int* Bkind=new int[3*flatsize(ny)];
    int count=0;
    for(int i=0;i<ny+1;i++){
        for(int j=0;j<i+1;j++){
            for(int k=0;k<j+1;k++){
                Bkind[3*count]=i;
                Bkind[3*count+1]=j;
                Bkind[3*count+2]=k;
                count++;
            }
        }
    }
    return Bkind;
}




void getBk_naive(double* Bk,fftw_complex* delta_k,int nside){
    int middleplus1=nside/2+1;
    int ny=nside/2;
    int yzsize=middleplus1*nside;
    int* counts=new int[flatsize(ny)];
    memset(counts, 0,flatsize(ny)*sizeof(int));


    int* kabsarr=new int[nside*nside*(nside/2+1)];
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ii,jj;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabsarr[yzsize*i+middleplus1*j+k]=(int)(sqrt(ii+jj+k*k)+0.5);
            }
        }
    }
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i1=0;i1<nside;i1++){
        int ind1,ind2,ind3;
        int k_abs1,k_abs2,k_abs3;
        int ii1,jj1,ii2,jj2,ii3,jj3;
        int ind;
        if (i1<middleplus1){
            ii1=i1;
        }else{
            ii1=(i1-nside);
        }

        for (int j1=0;j1<nside;j1++){
            if (j1<middleplus1){
                jj1=j1;
            }else{
                jj1=(j1-nside);
            }
            if (kabsarr[yzsize*i1+middleplus1*j1+0]>middleplus1) continue;

            for(int k1=0;k1<middleplus1;k1++){
                ind1=yzsize*i1+middleplus1*j1+k1;
                k_abs1=kabsarr[ind1];

                if (k_abs1>middleplus1) continue;
                for (int i2=0;i2<nside;i2++){
                    if (i2<middleplus1){
                        ii2=i2;
                    }else{
                        ii2=(i2-nside);
                    }
                    for (int j2=0;j2<nside;j2++){
                        if (j2<middleplus1){
                            jj2=j2;
                        }else{
                            jj2=(j2-nside);
                        }


                        for(int k2=0;k2<middleplus1;k2++){
                            ind2=yzsize*i2+middleplus1*j2+k2;
                            ii3=ii1+ii2;
                            if ((ii3<-middleplus1)||(ii3>middleplus1)) continue;
                            if (ii3<0) ii3=nside+ii3;

                            jj3=jj1+jj2;
                            if ((jj3<-middleplus1)||(jj3>middleplus1))  continue;
                            if (jj3<0) jj3=nside+jj3;

                            if ((k1+k2)>middleplus1) continue;

                            ind3=yzsize*(ii3)+middleplus1*(jj3)+(k1+k2);
                            
                            k_abs2=kabsarr[ind2];
                            k_abs3=kabsarr[ind3];
                            if ((k_abs1+k_abs2)<middleplus1){
                                ind=to1dind(k_abs1,k_abs2,k_abs3,ny);
                                //std::cout<<"-"<<Bk[ind]<<"-"<<std::endl;
                                counts[ind]+=1;
                                Bk[ind]+=delta_k[ind3][0]*(delta_k[ind1][0]*delta_k[ind2][0]-delta_k[ind1][1]*delta_k[ind2][1])+delta_k[ind3][1]*(delta_k[ind1][0]*delta_k[ind2][1]+delta_k[ind1][1]*delta_k[ind2][0]);
                            }

                        }
                    }
                }
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<flatsize(ny);i++){
        if (counts[i]!=0){
                Bk[i]/=counts[i];
            }
    }

    free(counts);
    free(kabsarr);
}

void getBkangav_naive(double* Bk,fftw_complex* delta_k,int nside){
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int* counts=new int[(nside/2+1)*(nside/2+1)];
    int* kabsarr=new int[nside*nside*(nside/2+1)];
    memset(counts, 0,(nside/2+1)*(nside/2+1)*sizeof(int));

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<nside;i++){
        int ii,jj;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabsarr[yzsize*i+middleplus1*j+k]=(int)(sqrt(ii+jj+k*k)+0.5);
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i1=0;i1<nside;i1++){
        int ind1,ind2,ind3;
        int k_abs1,k_abs2;
        int ii1,jj1,ii2,jj2,ii3,jj3;
        if (i1<middleplus1){
            ii1=i1;
        }else{
            ii1=(i1-nside);
        }

        for (int j1=0;j1<nside;j1++){
            if (j1<middleplus1){
                jj1=j1;
            }else{
                jj1=(j1-nside);
            }
            if (kabsarr[yzsize*i1+middleplus1*j1+0]>middleplus1) continue;

            for(int k1=0;k1<middleplus1;k1++){
                ind1=yzsize*i1+middleplus1*j1+k1;
                k_abs1=kabsarr[yzsize*i1+middleplus1*j1+k1];

                if (k_abs1>middleplus1) continue;
                for (int i2=0;i2<nside;i2++){
                    if (i2<middleplus1){
                        ii2=i2;
                    }else{
                        ii2=(i2-nside);
                    }
                    for (int j2=0;j2<nside;j2++){
                        if (j2<middleplus1){
                            jj2=j2;
                        }else{
                            jj2=(j2-nside);
                        }


                        for(int k2=0;k2<middleplus1;k2++){
                            ind2=yzsize*i2+middleplus1*j2+k2;
                            ii3=ii1+ii2;
                            if ((ii3<-middleplus1)||(ii3>middleplus1)) continue;
                            if (ii3<0){
                                ii3=nside+ii3;
                            }
                            jj3=jj1+jj2;
                            if ((jj3<-middleplus1)||(jj3>middleplus1))  continue;
                            if (jj3<0){
                                jj3=nside+jj3;
                            }
                            if ((k1+k2)>middleplus1) continue;


                            ind3=yzsize*(ii3)+middleplus1*(jj3)+(k1+k2);
                            
                            k_abs2=kabsarr[yzsize*i2+middleplus1*j2+k2];
                            if ((k_abs1+k_abs2)<middleplus1){
                                Bk[k_abs1*middleplus1+k_abs2]+=delta_k[ind3][0]*(delta_k[ind1][0]*delta_k[ind2][0]-delta_k[ind1][1]*delta_k[ind2][1])+delta_k[ind3][1]*(delta_k[ind1][0]*delta_k[ind2][1]+delta_k[ind1][1]*delta_k[ind2][0]);
                                counts[k_abs1*middleplus1+k_abs2]+=1;
                            }

                        }
                    }
                }
            }
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i=0;i<middleplus1;i++){
        for (int j=0;j<middleplus1;j++){
            if (counts[i*middleplus1+j]!=0){
                Bk[i*middleplus1+j]/=counts[i*middleplus1+j];
            }
        }
    }

    free(counts);
    free(kabsarr);
}