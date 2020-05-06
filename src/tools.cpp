#define INT_MAX 32768
#define INT_MAXdiv2pi 5215.189175235227//INT_MAX/(2pi)
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


static unsigned int g_seed;
inline void fast_srand(int seed) {
    g_seed = seed;
}

inline int fastrand() { 
  g_seed = (214013*g_seed+2531011); 
  return (g_seed>>16)&0x7FFF; 
} 

inline double getmag(double k) {
    return 4*(pow(sin(k/3.8),2)/pow((k+1),2));
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

inline double invsinc(double x){
    if (x==0) return 1.0;
    else return x/sin(x);
}

void getPk(double* ks,double* Pk,fftw_complex* delta_k,int nside,int MASexp){
    int len=(int)(sqrt(3*(nside/2)*(nside/2)))+1;
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int* counts=new int[len];
    memset(counts, 0,(len)*sizeof(int));


    //#pragma omp parallel for
    #pragma omp parallel for reduction(+:ks[0:len],Pk[0:len],counts[0:len])
    for (int i=0;i<nside;i++){
        int ind=0;
        double k_abs;
        int k_abs_ind;
        int ii,jj;
        double MASfactorx,MASfactory,factor;
        if (i<middleplus1){
            ii=i;
        }else{
            ii=(nside-i);
        }
        MASfactorx=pow(invsinc(M_PI*ii/nside),MASexp);
        for (int j=0;j<nside;j++){
            if (j<middleplus1){
                jj=j;
            }else{
                jj=(nside-j);
            }
            MASfactory=pow(invsinc(M_PI*jj/nside),MASexp);
            for(int k=0;k<middleplus1;k++){
                ind=yzsize*i+middleplus1*j+k;
                factor=MASfactorx*MASfactory*pow(invsinc(M_PI*k/nside),MASexp);
                k_abs=sqrt(ii*ii+jj*jj+k*k);
                k_abs_ind=(int)(k_abs);

                //#pragma omp critical
                Pk[k_abs_ind]+=(delta_k[ind][0]*delta_k[ind][0]+delta_k[ind][1]*delta_k[ind][1])*factor*factor;
                //#pragma omp critical
                counts[k_abs_ind]+=1;
                //#pragma omp critical
                ks[k_abs_ind]+=k_abs;

            }
        }
    }

    for (int i=0;i<len;i++){
        if (counts[i]!=0){
            Pk[i]/=counts[i];
            ks[i]/=counts[i];
        }
    }

    delete[] counts;
}


inline int to1dind(int a1,int a2, int a3, int d){
    return a1*(a1+1)*(a1+2)/6+(a2)*(a2+1)/2+a3;
}

void getBk(int* Bkind,double* Bk,double* Bkcount,int fsize,fftw_complex* delta_k,int nside,int k_min, int k_max,int step,bool quiet){
    if (!quiet) std::cout<<std::endl<<"  Generate k-rings"<<std::endl;
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int csize=(nside/2+1)*(nside)*(nside);
    int size=nside*nside*nside;
	int numks=k_max-k_min+1;

    //Initialize the kbins
    //No adaptive binning yet
    //assert((start>=0)&&("Start is negative"));
    //assert((step>0)&&("Step should be 1 or bigger"));

    //Make the k-space rings
    fftw_complex *partialdelta_ks;
    partialdelta_ks = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*numks*csize);
    memset(partialdelta_ks, 0,sizeof(fftw_complex)*numks*csize);

    #pragma omp parallel for shared(partialdelta_ks)
    for (int i=0;i<nside;i++){
        int ii,jj,kabs,ind;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabs=(int)(sqrt(ii+jj+k*k)+0.5);
                if ((kabs-k_min*step)%step==0){
                	kabs=(kabs-k_min*step)/step;
                	if ((kabs>=0)and(kabs<numks)){
                		ind=yzsize*i+middleplus1*j+k;
                    	partialdelta_ks[kabs*csize+ind][0]+=delta_k[ind][0];
                    	partialdelta_ks[kabs*csize+ind][1]+=delta_k[ind][1];
                	}
                }
            }     
        }
    }
    
    //std::cout<<"got"<<std::endl;

    if (!quiet) std::cout<<"  FFT k-rings"<<std::endl;
    double* partialdeltas=new double[numks*size];
    double* partialcounts=new double[numks*size];
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
    //Now the counts reuse same array for memory
    memset(partialdelta_ks, 0,sizeof(fftw_complex)*numks*csize);

    #pragma omp parallel for
    for (int i=0;i<nside;i++){
        int ii,jj,kabs;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){  
                kabs=(int)(sqrt(ii+jj+k*k)+0.5);
                if ((kabs-k_min*step)%step==0){
                    kabs=(kabs-k_min*step)/step;
                    if ((kabs>=0)and(kabs<numks)) partialdelta_ks[kabs*csize+yzsize*i+middleplus1*j+k][0]=1;
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
    delete[] partialdelta_ks;

    std::ofstream tt;
    tt.open("tesets");
    tt.write((char*)partialdeltas,sizeof(double)*numks*size);
    tt.close();


    if (!quiet) std::cout<<"  Sum over realspace"<<std::endl;
    /*
    //Change to val array

	std::valarray<double> partialdeltasv[numks];
	//partialdeltasv=new valarray<double>[numks];
	
	for(int i=0;i<numks;i++){
		partialdeltasv[i]=std::valarray<double>(partialdeltas+i*size,size);
	}
	//free(partialdeltas);

	std::valarray<double> partialcountsv[numks];
	for(int i=0;i<numks;i++){
		partialcountsv[i]=std::valarray<double>(partialcounts+i*size,size);
	}
	//free(partialcounts);
	*/
    //std::cout<<fsize<<std::endl;

    int count=0;
    for(int i=k_min;i<k_max+1;i++){
        //int multi=1;
        int ii=i-k_min;
        int jj,kk;
        double deltasum,countsum;
        //int ind;
        double* partialproddelta=new double[size];
        double* partialprodcount=new double[size];
        
        for(int j=k_min;j<i+1;j++){
            jj=j-k_min;

            for(int sumind=0;sumind<size;sumind++){
                partialproddelta[sumind]=partialdeltas[ii*size+sumind]*partialdeltas[jj*size+sumind];
                partialprodcount[sumind]=partialcounts[ii*size+sumind]*partialcounts[jj*size+sumind];
            }
            

            for(int k=k_min;k<j+1;k++){
                if (k<(i-j)) continue;
                /*
                if (i==j){
                    if (j==k) multi=6;
                    else multi=2;
                }
                else if ((j==k)or(i==k)) multi=2;
                */
                kk=k-k_min;
                
                deltasum=0;
                countsum=0;
                
    			#pragma omp parallel for reduction(+: deltasum,countsum)
                for(int sumind2=0;sumind2<size;sumind2++){
                    deltasum+=partialproddelta[sumind2]*partialdeltas[kk*size+sumind2];
                    countsum+=partialprodcount[sumind2]*partialcounts[kk*size+sumind2];
                }
                
                //deltasum = (partialdeltasv[i] * partialdeltasv[j]*partialdeltasv[k]).sum();
                //countsum = (partialcountsv[i] * partialcountsv[j]*partialcountsv[k]).sum();

                if (countsum>0.0){
                    Bk[count]=deltasum/countsum;
                    Bkcount[count]=countsum;
                }
                else{
                    Bk[count]=0.0;
                    Bkcount[count]=0.0;
                }
                Bkind[3*count+0]=step*i;
                Bkind[3*count+1]=step*j;
                Bkind[3*count+2]=step*k;
                count++;
            }
        }
        
        delete[] partialproddelta;
        delete[] partialprodcount;
        
    }
    //std::cout<<numks<<" "<<fsize<<" "<<count<<std::endl;
    assert(count==fsize);
    if (!quiet) std::cout<<"  Done"<<std::endl;
}

void getBk_custom_k(int* Bkindice,std::map<int,int> kmap,double* Bk,double* Bkcount,int fsize,fftw_complex* delta_k,int nside,int numks,bool quiet){
    if (!quiet) std::cout<<std::endl<<"  Generate k-rings"<<std::endl;
    int middleplus1=nside/2+1;
    int yzsize=middleplus1*nside;
    int csize=(nside/2+1)*(nside)*(nside);
    int size=nside*nside*nside;

    //Initialize the kbins
    //Make the k-space rings
    fftw_complex *partialdelta_ks;
    partialdelta_ks = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*numks*csize);
    memset(partialdelta_ks, 0,sizeof(fftw_complex)*numks*csize);

    std::map<int,int>::iterator no;
    no=kmap.end();

    #pragma omp parallel for shared(partialdelta_ks,kmap,no)
    for (int i=0;i<nside;i++){
        int ii,jj,kabs,ind;
        std::map<int,int>::iterator it;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){
                kabs=(int)(sqrt(ii+jj+k*k)+0.5);
                
                it=kmap.find(kabs);
                /*
                if  ((i==0)&&(j==0)){
                std::cout<<"k"<<kabs<<"pass"<<(it!=no)<<std::endl;
                std::cout<<it->second<<std::endl;
                }
                */
                if (it!=no){
                    kabs=it->second;
                    ind=yzsize*i+middleplus1*j+k;
                    partialdelta_ks[kabs*csize+ind][0]+=delta_k[ind][0];
                    partialdelta_ks[kabs*csize+ind][1]+=delta_k[ind][1]; 
                }
            }     
        }
    }
    /*
    std::ofstream ttt("tttt");
    ttt.write((char*)partialdelta_ks, sizeof(fftw_complex)*numks*csize);
    ttt.close();
    */

    if (!quiet) std::cout<<"  FFT k-rings"<<std::endl;
    double* partialdeltas=new double[numks*size];
    double* partialcounts=new double[numks*size];
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
    //Now the counts reuse same array for memory
    memset(partialdelta_ks, 0,sizeof(fftw_complex)*numks*csize);

    #pragma omp parallel for shared(partialdelta_ks,kmap,no)
    for (int i=0;i<nside;i++){
        int ii,jj,kabs;
        std::map<int,int>::iterator it;
        if (i<middleplus1) ii=i*i;
        else ii=(i-nside)*(i-nside);
        for (int j=0;j<nside;j++){
            if (j<middleplus1) jj=j*j;
            else jj=(j-nside)*(j-nside);
            for(int k=0;k<middleplus1;k++){  
                kabs=(int)(sqrt(ii+jj+k*k)+0.5);
                it=kmap.find(kabs);
                if (it!=no){
                    kabs=it->second;
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
    delete[] partialdelta_ks;
    /*
    std::ofstream tt;
    tt.open("tesets");
    tt.write((char*)partialdeltas,sizeof(double)*numks*size);
    tt.close();
    */

    if (!quiet) std::cout<<"  Sum over realspace"<<std::endl;

    int count=0;
    double deltasum,countsum;
    for(int i=0;i<fsize;i++){
        int k1ind=kmap[Bkindice[3*i+0]];
        int k2ind=kmap[Bkindice[3*i+1]];
        int k3ind=kmap[Bkindice[3*i+2]];
        //std::cout<<k1ind<<" "<<k2ind<<" "<<k3ind<<" "<<std::endl;

        deltasum=0;
        countsum=0;
        #pragma omp parallel for reduction(+: deltasum,countsum)
        for(int sumind=0;sumind<size;sumind++){
            deltasum+=partialdeltas[k1ind*size+sumind]*partialdeltas[k2ind*size+sumind]*partialdeltas[k3ind*size+sumind];
            countsum+=partialcounts[k1ind*size+sumind]*partialcounts[k2ind*size+sumind]*partialcounts[k3ind*size+sumind];
        }
        if (countsum>0.0){
            std::cout<<deltasum<<std::endl;
            Bk[i]=deltasum/countsum;
            Bkcount[i]=countsum;
        }
        else{
            Bk[i]=0.0;
            Bkcount[i]=0.0;
        }
        count++;
    }

    assert(count==fsize);
    if (!quiet) std::cout<<"  Done"<<std::endl;
}





























/*
inline int flatsize(int ny){
    return (ny+1)*(ny+2)*(ny+3)/6;
}

int* getBk_ind(int nside,int start, int step){
    int ny=(nside/2-start)/step;
    int* Bkind=new int[3*flatsize(ny)];
    int count=0;
    for(int i=0;i<ny+1;i++){
        for(int j=0;j<i+1;j++){
            for(int k=0;k<j+1;k++){
                Bkind[3*count]=i*step+start;
                Bkind[3*count+1]=j*step+start;
                Bkind[3*count+2]=k*step+start;
                count++;
            }
        }
    }
    return Bkind;
}
*/


/*
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

    delete[] counts;
    delete[] kabsarr;
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

    delete[] counts;
    delete[] kabsarr;
}
*/