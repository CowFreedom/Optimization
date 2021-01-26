#include<stdio.h>

#include "linear_algebra\gemm.cuh"
#include "linear_algebra\gmv.cuh"
#include "linear_algebra\ldl.cuh"
#include "linear_algebra\lu.cuh"
#include "hostgpu_bindings.h"

__global__
void check_diagonal_has_zero(int n,const float* A, int stride_row, int stride_col, bool* res){
	
	for (int i=0;i<n;i++){
		float val=A[i*stride_row+i*stride_col];
		//printf("val:%f\n",val);
		if (val==0){
			*res=true;
			return;
		}
	}
	*res=false;
}

__host__
void calc_stepdirection_f32(int rdim, int xdim, const float* xi_h,const float* residual_h, const float* J_h, float* output_h,bool* lu_used){
	float* xi_d;
	float* output_d;
	float* J_d;
	float* J_t_J_d;
	float* b_d;
	float* residual_d;
	bool* lu_used_d;
	
	int sizeX=sizeof(float)*xdim;
	
	//printf("in calc_stepdirection_f32\n");
	
	int sizeR=sizeof(float)*rdim;
	int sizeJ=sizeof(float)*xdim*rdim;
	
	cudaMalloc((void**) &xi_d,sizeX);
	cudaMalloc((void**)&output_d,sizeX);
	cudaMalloc((void**)&J_d,sizeJ);
	cudaMalloc((void**)&J_t_J_d,sizeof(float)*xdim*xdim);
	cudaMalloc((void**)&b_d,sizeX);
	cudaMalloc((void**)&residual_d,sizeR);
	cudaMalloc((void**)&lu_used_d,sizeof(bool));
	
	cudaMemcpy((void*) xi_d, (void*) xi_h, sizeX,cudaMemcpyHostToDevice);
	cudaMemcpy((void*) residual_d, (void*) residual_h, sizeR,cudaMemcpyHostToDevice);
	cudaMemcpy((void*) J_d, (void*) J_h, sizeJ,cudaMemcpyHostToDevice);
	
	gemm_f32_device(xdim, xdim, rdim, 1.0,J_d,xdim, 1, J_d, 1,xdim, 0.0, J_t_J_d,1,xdim);
	gmv_f32_device(xdim,rdim,1.0,J_d,xdim,1,residual_d,1,0.0,b_d,1);
	
	/*
	k_lu_f32<<<1,1>>>(xdim,J_t_J_d,1,xdim);
	k_lu_solve_lower_f32<<<1,1>>>();
	*/
	
//	k_printmatrix<<<1,1>>>("J_d",rdim,xdim,J_d,1,xdim);
//	k_printmatrix<<<1,1>>>("J_t_J_d",xdim,xdim,J_t_J_d,1,xdim);
	//printf("xdim %d\n",xdim);
	choi_f32_device(xdim,J_t_J_d,1,xdim);
//	k_printmatrix<<<1,1>>>("J_t_J_d",xdim,xdim,J_t_J_d,1,xdim);
	check_diagonal_has_zero<<<1,1>>>(xdim, J_t_J_d,1,xdim,lu_used_d);	
	cudaMemcpy((void*) lu_used, (void*) lu_used_d, sizeof(bool),cudaMemcpyDeviceToHost);

//
	if (!(*lu_used)){
		//Using cholesky LDL^{T} decomposition
		//k_printmatrix<<<1,1>>>("Output",xdim,xdim,J_t_J_d,1,xdim);
		//k_printmatrix<<<1,1>>>("b_d",xdim,1,b_d,1,1);
		//k_printmatrix<<<1,1>>>("residual",rdim,1,residual_d,1,1);
		//k_printmatrix<<<1,1>>>("J_t_J_d",xdim,xdim,J_t_J_d,1,xdim);
		choi_solve_f32_device(xdim,1,J_t_J_d,1,xdim,b_d,1,1,output_d,1,1);
					
	}
	else{
		//Using LU decomposition
	//	k_printmatrix<<<1,1>>>("b_d",xdim,1,b_d,1,1);
//		printf("In LU calculation\n");
		//Recreate the initial state of J_t_J_d;
		gemm_f32_device(xdim, xdim, rdim, 1.0,J_d,xdim, 1, J_d, 1,xdim, 0.0, J_t_J_d,1,xdim); 
	//	k_printmatrix<<<1,1>>>("J_t_J_d",xdim,xdim,J_t_J_d,1,xdim);
		k_lu_f32<<<1,1>>>(xdim,J_t_J_d,1,xdim);
		//k_printmatrix<<<1,1>>>("J_t_J_d",xdim,xdim,J_t_J_d,1,xdim);
		
		lu_solve_f32_device(xdim,1,J_t_J_d,1,xdim,b_d,1,1,output_d,1,1);

	}
		
	cudaMemcpy((void*) output_h, (void*) output_d, sizeX,cudaMemcpyDeviceToHost);		
	//k_printmatrix<<<1,1>>>("Output",xdim,1,output_d,1,1);
	cudaFree(xi_d);
	cudaFree(output_d);
	cudaFree(J_d);
	cudaFree(J_t_J_d);
	cudaFree(lu_used_d);	

}