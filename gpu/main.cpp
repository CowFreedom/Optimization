import optimization.transformation;
import optimization.gpubindings;
#include <iostream>
#include<stdio.h>
#include <stdlib.h>
#include <time.h>



extern "C"{
void dgemm_nn(size_t m, size_t n, size_t k, double alpha, float* A, size_t stride_row_a, size_t stride_col_a,
	float* B, size_t stride_row_b, size_t stride_col_b, double beta, float* C, size_t stride_row_c, size_t stride_col_c);

}

void fill_matrix(float* M, size_t m, size_t n){

	int a=20;
	
	for (int i=0;i<m*n;i++){
		M[i]=((float)rand()/(float)(RAND_MAX)) * a;
	}

}

void run_test_float(){
	const size_t n=16*1000; //multiple of block size for now
	float* A=(float*) malloc(static_cast<size_t>(n)*static_cast<size_t>(n)*sizeof(float));
	float* B=(float*) malloc(n*n*sizeof(float));
	float* C=(float*) malloc(n*n*sizeof(float));
	srand((unsigned int)time(NULL));
	fill_matrix(A,n,n);
	fill_matrix(B,n,n);

    clock_t start1, start2, start3,end;
    double time_used1, time_used2, time_used3;
	start1=clock();
	opt::math::cpu::dgemm_nn(n,n,n,1.0,A,1,n,B,1,n,0.0,C,1,n);
	end=clock();

	time_used1=((double) (end - start1)) / CLOCKS_PER_SEC;
	start3=clock();
	printf("Start blockmatrix:\n");
	opt::math::gpu::dgemm_nn(n,n,n,1.0,A,1,n,B,1,n,0.0,C,1,n);
	end=clock();
	time_used3=((double) (end - start3)) / CLOCKS_PER_SEC;
	printf("Matrix mult CPU: %f seconds\n",time_used1);
	printf("Matrix mult GPU: %f seconds\n",time_used3);
	/*
	printf("\nA=\n");
	for (int i=0;i<n;i++){
		for (int j=0;j<n;j++){
			printf("%f\t",A[i*n+j]);
		}
		printf("\n");
	}
	
	printf("\nB=\n");
	for (int i=0;i<n;i++){
		for (int j=0;j<n;j++){
			printf("%f\t",B[i*n+j]);
		}
		printf("\n");
	}
	*/

	
	free(A);
	free(B);
	free(C);
}

int main(){
	run_test_float();


}