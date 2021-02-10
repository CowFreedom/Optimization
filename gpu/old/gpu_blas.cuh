

extern "C"{
	//__host__
	void dgemm_nn_gpu_float(size_t m, size_t n, size_t k, float alpha, float* A, size_t stride_row_a, size_t stride_col_a,
	float* B, size_t stride_row_b, size_t stride_col_b, float beta, float* C, size_t stride_row_c, size_t stride_col_c);

	//__host__
	void dgemm_nn_gpu_double(size_t m, size_t n, size_t k, double alpha, double* A, size_t stride_row_a, size_t stride_col_a,
	double* B, size_t stride_row_b, size_t stride_col_b, double beta, double* C, size_t stride_row_c, size_t stride_col_c);

}