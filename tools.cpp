#define INT_MAX 32768
#define INT_MAXdiv2pi 5215.189175235227//INT_MAX/(2pi)
#include <fftw3.h>
#include <math.h>
#include <string.h>

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
    for (int i=0;i<csize;i++) phases[i]=((double)fastrand())/INT_MAXdiv2pi;

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
    for (int i=0;i<nside/2+2;i++){
        if (counts[i]!=0) Pk[i]/=counts[i];
    }

    free(counts);
}

void getBk_naive(fftw_complex* Bk,fftw_complex* delta_k,int nside){
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int* counts=new int[(nside/2+1)*(nside/2+1)];
    memset(counts, 0,(nside/2+1)*(nside/2+1)*sizeof(int));

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

            for(int k1=0;k1<middleplus1;k1++){
                ind1=yzsize*i1+middleplus1*j1+k1;
                k_abs1=(int)(sqrt(ii1*ii1+jj1*jj1+k1*k1)+0.5);


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
                            jj3=jj1+jj2;
                            if (ii3<0){
                                ii3=nside+ii3;
                            }
                            jj3=jj1+jj2;
                            if (jj3<0){
                                jj3=nside+jj3;
                            }


                            ind3=yzsize*(ii3)+middleplus1*(jj3)+(k1+k2);

                            k_abs2=(int)(sqrt(ii2*ii2+jj2*jj2+k2*k2)+0.5);
                            if ((k_abs1+k_abs2)<middleplus1){
                                Bk[k_abs1*middleplus1+k_abs2][0]+=delta_k[ind3][0]*(delta_k[ind1][0]*delta_k[ind2][0]-delta_k[ind1][1]*delta_k[ind2][1])
                                +delta_k[ind3][1]*(delta_k[ind1][0]*delta_k[ind2][1]+delta_k[ind1][1]*delta_k[ind2][0]);
                                Bk[k_abs1*middleplus1+k_abs2][1]+=-delta_k[ind3][1]*(delta_k[ind1][0]*delta_k[ind2][0]-delta_k[ind1][1]*delta_k[ind2][1])
                                +delta_k[ind3][0]*(delta_k[ind1][0]*delta_k[ind2][1]+delta_k[ind1][1]*delta_k[ind2][0]);
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
    for (int i=0;i<nside/2+2;i++){
        for (int j=0;j<nside/2+2;j++){
            if (counts[i*middleplus1+j]!=0){
                Bk[i][0]/=counts[i*middleplus1+j];
                Bk[i][1]/=counts[i*middleplus1+j];
            }
        }
    }

    free(counts);   
}