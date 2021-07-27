module;
export module optimization.solvers;

export import optimization.solvers.gaussnewton;


export bool abs_tol2(int dim, const float* x, int stride_x, float tol){
	for (int i=0;i<dim;i++){
		float val=(x[i*stride_x]>=0.0)?x[i*stride_x]:-x[i*stride_x];
		if (val>tol){
			return false;
		}
	}
	return true;
}