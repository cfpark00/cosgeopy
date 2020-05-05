#include<iostream>
using namespace std;

void disp(double* arr){
	cout<<arr[0]<<" "<<arr[1]<<endl;
}

void add1toint(int& a){
	a++;
	a++;
}

int main(){
	double a[5];
	for (int i=0;i<5;i++){
		a[i]=(double)i;
	}
	disp(a+2);

	cout<<endl;
	int o=1;
	add1toint(o);
	cout<<o<<endl;
	return 0;
}