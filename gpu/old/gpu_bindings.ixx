module;
#include <vector>
#include "gpu_blas.cuh"
#include "gpu_gaussnewton.cuh"
#include <functional>
export module optimization.gpubindings;

//forward declaration
/*
extern "C"{

	struct EvalGNSGPU{
		void* res; //pointer to function object, see PIMPL pattern http://www.olivierlanglois.net/idioms_for_using_cpp_in_c_programs.html
	};
	
	void EvalGNSGPU_eval(EvalGNSGPU* r, double* x, double* storage){
	
		delete static_cast<std::function<void(double*, double*)>>(r->res)(x,storage);
		delete r;
	}	

	void EvalGNSGPU_destroy(EvalGNSGPU* r){
	
		delete static_cast<std::function<void(double*, double*)>>(r->res);
		delete r;
	}


		
}
*/

namespace opt{


	namespace math{

		namespace gpu{

			export void dgemm_nn(size_t m, size_t n, size_t k, float alpha, float* A, size_t stride_row_a, size_t stride_col_a,
			float* B, size_t stride_row_b, size_t stride_col_b, float beta, float* C, size_t stride_row_c, size_t stride_col_c) {
		
				dgemm_nn_gpu_float(m,n,k,alpha,A,stride_row_a, stride_col_a,B,stride_row_b,stride_col_b,beta,C,stride_row_c,stride_col_c);

			}
			
			export void dgemm_nn(size_t m, size_t n, size_t k, double alpha, double* A, size_t stride_row_a, size_t stride_col_a,
			double* B, size_t stride_row_b, size_t stride_col_b, double beta, double* C, size_t stride_row_c, size_t stride_col_c) {
		
				dgemm_nn_gpu_double(m,n,k,alpha,A,stride_row_a, stride_col_a,B,stride_row_b,stride_col_b,beta,C,stride_row_c,stride_col_c);

			}

		}
	}
}