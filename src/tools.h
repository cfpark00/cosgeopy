#include <map>
void fill_delta_k(fftw_complex*,int,int);
void getPk(double*,double*,fftw_complex*,int,int);
void getBkangav_naive(double*,fftw_complex*,int);
int* getBk_ind(int,int start=0, int step=1);
void getBk_naive(double*,fftw_complex*,int);
void getBk(int*,double*,double*,int,fftw_complex*,int,int, int,int,bool);
void getBk_custom_k(int*,std::map<int,int>,double*,double*,int,fftw_complex*,int,int,bool);