#include "gpu_gaussnewton.cuh"
#include "gpu_blas.cuh"



__host__
float f0_f(int rdim, int xim, const float* params, EvalGNSGPU* r){
	
	float result[1]={-1};
	
	EvalGNSGPU_eval(r,params,residuals);
	
	return *result;
}

__host__
bool GNSGPU(float* result, int rdim, int xdim, float* x0, EvalGNSGPU* r, EvalGNSGPU j,  float tol, int max_iter, int max_step_iter){
	bool run_finished=false;
	
	float fmin;
	float* J_h=malloc(sizeof(float)*xdim*rdim);
	float* grad_h=malloc(sizeof(float)*xdim);
	
	float* J_d;
	float* grad_d;
	cudaMalloc((void**) &J_d, sizeof(float)*xdim*rim);
	cudaMalloc((void**) &grad_d, sizeof(float)*xdim);
	
	if ((xdim == 0) || (rdim==0)){
		return false;
	}
	
	float fmin=f0(rdim,xdim,x0,r);
	
	float alpha=0.7;
	int iter=0;
	
	memcpy(xi,&x0,sizeof(float)*xdim);
	
	while(run_finished==false && (iter<max_iter)){
	//	EvalGNSGPU_eval(j_t,xi,J_t_d);
	//	EvalGNSGPU_eval(j_t_j_inv,xi,J_t_J_inv_d);
		
	//	dgemm_nn_gpu_float(xdim,rdim,xdim,gfloat(1.0),J_t_J_inv.begin(),1,xdim,J_t.begin(),1,rdim,gfloat(0.0),C.begin(),1,rdim);
	}
	
	return true;
	

}


