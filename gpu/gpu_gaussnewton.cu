#include "gpu_gaussnewton.cuh"
#include "gpu_blas.cuh"



__host__
float f0_f(const float* params, int xdim, float* residuals,EvalGNSGPU* r){
	
	float result[1]={-1};
	
	EvalGNSGPU_eval(r,params,residuals);
	dgemm_nn_gpu_float(1,1,xdim,float(1.0),residuals,1,xdim,residuals,1,1,float(0.0),result,1,1);
	return *result;
}

__host__
bool GNSGPU_j_f(float* result, int rdim, float* x0, int xdim, EvalGNSGPU* r, EvalGNSGPU j_t, EvalGNSGPU j_t_j_inv, float tol, int max_iter, int max_step_iter){
	bool run_finished=false;
	
	float fmin;
	float* J_t_h=(float*) malloc(sizeof(float)*xdim*rdim);
	float* grad_h=(float*) malloc(sizeof(float)*xdim);
	float* J_t_J_inv_h=(float*) (sizeof(float)*xdim*xdim);
	
	int n_threads=1;
	
	float* J_t_d;
	float* grad_d;
	float* J_t_J_inv_d;
	float* C_d;
	
	float* C_h=(float*) malloc(sizeof(float)*xdim*rdim);
	float* xi=(float*) malloc(sizeof(float)*xdim);
	float* xs=(float*) malloc(sizeof(float)*xdim*n_threads);
	
	float alpha=0.7;
	int iter=0;
	
	memcpy(xi,&x0,sizeof(float)*xdim);
	
	cudaMalloc((void**) &J_t_d, sizeof(float)*xdim*rdim);
	cudaMalloc((void**) &grad_d, sizeof(float)*xdim);
	cudaMalloc((void**) &J_t_J_inv_d, sizeof(float)*xdim*xdim);
	cudaMalloc((void**) &C_d,sizeof(float)*xdim*xdim);

	while(run_finished==false && (iter<max_iter)){
	//	EvalGNSGPU_eval(j_t,xi,J_t_d);
	//	EvalGNSGPU_eval(j_t_j_inv,xi,J_t_J_inv_d);
		
	//	dgemm_nn_gpu_float(xdim,rdim,xdim,gfloat(1.0),J_t_J_inv.begin(),1,xdim,J_t.begin(),1,rdim,gfloat(0.0),C.begin(),1,rdim);
	}
	
	return true;
	

}


