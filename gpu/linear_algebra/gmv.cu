#include "gmv.cuh"
#include "stdio.h"

//Calculates c=alpha*A*x+beta*c, for matrix A (dimensions nxm), vectors c,x and scalars alpha, beta
__global__
void k_gmv_f32(int n, int m, float alpha, float* A, int stride_row_a, int stride_col_a,float* x, int stride_x, float beta, float* c, int stride_c){

	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int col=bx*TILE_SIZE+tx;
	if (col<n){
		int q=m/TILE_SIZE;
		int rem=m%TILE_SIZE;
		
		float* ptrA=A+col*stride_col_a;
		//float* a_end=ptrA+TILE_SIZE*stride_col_a;
		float* ptrX=x;

		__shared__ float buf[TILE_SIZE];
		float sum=0.0;
		for (int i=0;i<q;i++){
			for (int j=0;j<TILE_SIZE;j++){
				buf[j]=ptrX[j*stride_x];
			}
			
			#pragma unroll
			for (int j=0;j<TILE_SIZE;j++){		
				sum+=buf[j]*ptrA[j*stride_row_a];
			}
			
			ptrA+=TILE_SIZE*stride_row_a;
			ptrX+=TILE_SIZE*stride_x;
			__syncthreads();
		}
		
		if (rem>0){
			
			for (int j=0;j<rem;j++){
				buf[j]=ptrX[j*stride_x];
			}
			
			for (int j=0;j<rem;j++){				
				sum+=buf[j]*ptrA[j*stride_row_a];
			}
		}
		
		c[col*stride_c]=beta*c[col*stride_c]+alpha*sum;
	}
}


__host__
void gmv_f32_device(int n, int m, float alpha, float* A_d, int stride_row_a, int stride_col_a, float* x, int stride_x, float beta, float* C_d, int stride_c){
	float bsmx=64; //blocksize x
	dim3 threadLayout=dim3(bsmx,1,1);
	dim3 grid=dim3(ceil(n/bsmx),1,1);
	k_gmv_f32<<<grid,threadLayout>>>(n,m,alpha,A_d,stride_row_a,stride_col_a,x,stride_x,beta,C_d,stride_c);
}

__host__
void gmv_f32(int n, int m, float alpha, float* A_h, int stride_row_a, int stride_col_a, float* x_h, int stride_x, float beta, float* C_h, int stride_c){
	if ((n==0) || (m==0)){
		return;
	}
	
	float* A_d;
	float* x_d;
	float* C_d;
	
	int sizeA=sizeof(float)*n*m;
	int sizeX=sizeof(float)*m;
	
	cudaMalloc((void**)&A_d, sizeA);
	cudaMalloc((void**)&x_d,sizeX);
	cudaMalloc((void**)&C_d,sizeA);

	cudaError_t copy1=cudaMemcpy((void*) A_d, (void*) A_h, sizeA,cudaMemcpyHostToDevice);
	cudaError_t copy2=cudaMemcpy((void*) x_d, (void*) x_h, sizeX,cudaMemcpyHostToDevice);
	cudaError_t copy3=cudaMemcpy((void*) C_d, (void*) C_h, sizeA,cudaMemcpyHostToDevice);
	
	if ((copy1==cudaSuccess) && (copy2==cudaSuccess) && (copy3==cudaSuccess)){
		float bsmx=64; //blocksize x
		dim3 threadLayout=dim3(bsmx,1,1);
		dim3 grid=dim3(ceil(n/bsmx),1,1);
		k_gmv_f32<<<grid,threadLayout>>>(n,m,alpha,A_d,stride_row_a,stride_col_a,x_d,stride_x,beta,C_d,stride_c);	
		
		cudaMemcpy((void*)C_h,(void*)C_d,sizeA,cudaMemcpyDeviceToHost);
		
	}
	else{
		printf("Error copying value to device in gmv_f32\n");
	}
	cudaFree(A_d);
	cudaFree(x_d);
	cudaFree(C_d);
}