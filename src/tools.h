void fill_delta_k(fftw_complex*,int,int);
void getPk(double*,fftw_complex*,int);
void getBkangav_naive(double*,fftw_complex*,int);
int* getBk_ind(int,int start=0, int step=1);
void getBk_naive(double*,fftw_complex*,int);
void getBk(double*,fftw_complex*,int,int* ks=NULL,int numks=0,int start=0, int step=1,bool quiet=0);