module;

export module gpubindings;

//forward declaration
extern "C"{
	export void dgemm_nn_gpu_float(size_t m, size_t n, size_t k, float alpha, float* A, size_t stride_row_a, size_t stride_col_a,
	float* B, size_t stride_row_b, size_t stride_col_b, float beta, float* C, size_t stride_row_c, size_t stride_col_c);
	
	export void dgemm_nn_gpu_double(size_t m, size_t n, size_t k, double alpha, double* A, size_t stride_row_a, size_t stride_col_a,
	double* B, size_t stride_row_b, size_t stride_col_b, double beta, double* C, size_t stride_row_c, size_t stride_col_c);


}
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