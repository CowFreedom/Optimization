#include<stdio.h>
#include <stdlib.h>
#include <time.h>
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

__global__ 
void k_matrix_mul_block(float* d_A, float* d_B, float* d_C, size_t width){
	const int TILE_WIDTH=16;
	__shared__ float d_A_buf[TILE_WIDTH*TILE_WIDTH];
	__shared__ float d_B_buf[TILE_WIDTH*TILE_WIDTH];

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;

	
	int row=by*TILE_WIDTH+ty;
	int col=bx*TILE_WIDTH+tx;
	float res=0.0;
	
	for (int i=0;i<width/TILE_WIDTH;i++){
		
		d_A_buf[ty*TILE_WIDTH+tx]=d_A[(row*width)+i*TILE_WIDTH+tx];
		d_B_buf[tx*TILE_WIDTH+ty]=d_B[(i*TILE_WIDTH+ty)*width+col];
		__syncthreads();
		
		
		//multiply numbers

		for (int j=0;j<TILE_WIDTH;j++){
			res+=d_A_buf[TILE_WIDTH*ty+j]*d_B_buf[TILE_WIDTH*tx+j];
		}
		__syncthreads();
	}
	d_C[row*width+col]=res;
	
}

__host__ 
void mult_block(const float* A, const float* B, const float* C, size_t n){

	float* d_A;
	float* d_B;
	float* d_C;
	
	int size=sizeof(float)*n*n;
	cudaMalloc((void**) &d_A,size);
	cudaMalloc((void**) &d_B,size);
	cudaMalloc((void**) &d_C,size);
	
	cudaError_t copy1=cudaMemcpy((void*) d_A,(void*) A,size,cudaMemcpyHostToDevice);
	cudaError_t copy2=cudaMemcpy((void*) d_B,(void*) B,size,cudaMemcpyHostToDevice);
	cudaMemset(d_C, 0.0, n*n*sizeof(float));
	if (copy1==cudaSuccess && copy2==cudaSuccess){
		float blockSize=16.0;
		int gridxyz=ceil(n/blockSize);
		dim3 blockLayout(blockSize,blockSize,1);
		dim3 grid(gridxyz,gridxyz,1);
		k_matrix_mul_block<<<grid,blockLayout>>>(d_A,d_B,d_C,n);
		cudaError_t copyResult=cudaMemcpy((void*) C,(void*) d_C,size,cudaMemcpyDeviceToHost);		
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		
	}
	else if (copy1==cudaErrorInvalidValue){
		printf("Error copying values to device. cudaErrorInvalidValue\n");
	}
	else if (copy1==cudaErrorInvalidMemcpyDirection){
		printf("Error copying values to device. cudaErrorInvalidMemcpyDirection\n");
	}
	else{
		printf("Error copying values to device.\n");
	}
}


__global__ 
void k_matrix_mul_block_double(double* d_A, double* d_B, double* d_C, size_t width){
	const int TILE_WIDTH=16;
	__shared__ double d_A_buf[TILE_WIDTH*TILE_WIDTH];
	__shared__ double d_B_buf[TILE_WIDTH*TILE_WIDTH];

	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;

	
	int row=by*TILE_WIDTH+ty;
	int col=bx*TILE_WIDTH+tx;
	
	float res=0.0;
	for (int i=0;i<width/TILE_WIDTH;i++){
		d_A_buf[ty*TILE_WIDTH+tx]=d_A[(row*width)+i*TILE_WIDTH+tx];
		d_B_buf[tx*TILE_WIDTH+ty]=d_B[(i*TILE_WIDTH+ty)*width+col];
		__syncthreads();

		
		//multiply numbers

		for (int j=0;j<TILE_WIDTH;j++){
			res+=d_A_buf[TILE_WIDTH*ty+j]*d_B_buf[TILE_WIDTH*tx+j];
		}
		__syncthreads();
	}
	d_C[row*width+col]=res;
}

__host__ 
void mult_block_double(const double* A, const double* B, const double* C, size_t n){

	double* d_A;
	double* d_B;
	double* d_C;
	
	int size=sizeof(double)*n*n;
	cudaMalloc((void**) &d_A,size);
	cudaMalloc((void**) &d_B,size);
	cudaMalloc((void**) &d_C,size);
	
	cudaError_t copy1=cudaMemcpy((void*) d_A,(void*) A,size,cudaMemcpyHostToDevice);
	cudaError_t copy2=cudaMemcpy((void*) d_B,(void*) B,size,cudaMemcpyHostToDevice);
	cudaMemset(d_C, 0.0, n*n*sizeof(double));
	if (copy1==cudaSuccess){
		float blockSize=16.0;
		int gridxyz=ceil(n/blockSize);
		dim3 blockLayout(blockSize,blockSize,1);
		dim3 grid(gridxyz,gridxyz,1);
		k_matrix_mul_block_double<<<grid,blockLayout>>>(d_A,d_B,d_C,n);
		cudaMemcpy((void*) C,(void*) d_C,size,cudaMemcpyDeviceToHost);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		
	}
	else if (copy1==cudaErrorInvalidValue){
		printf("Error copying values to device. cudaErrorInvalidValue\n");
	}
	else if (copy1==cudaErrorInvalidMemcpyDirection){
		printf("Error copying values to device. cudaErrorInvalidMemcpyDirection\n");
	}
	else{
		printf("Error copying values to device.\n");
	}
}




extern "C"{
	__host__
	void dgemm_nn_gpu_float(size_t m, size_t n, size_t k, float alpha, float* A, size_t stride_row_a, size_t stride_col_a,
	float* B, size_t stride_row_b, size_t stride_col_b, float beta, float* C, size_t stride_row_c, size_t stride_col_c) {
		
		mult_block(A,B,C,m);

	}
	
	__host__
	void dgemm_nn_gpu_double(size_t m, size_t n, size_t k, double alpha, double* A, size_t stride_row_a, size_t stride_col_a,
	double* B, size_t stride_row_b, size_t stride_col_b, double beta, double* C, size_t stride_row_c, size_t stride_col_c) {
		
		mult_block_double(A,B,C,m);

	}
}