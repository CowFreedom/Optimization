#ifdef __CUDACC__
__global__
#endif
void k_diag_lower_gemm_ldl_f32(int n, float alpha, const float* D, int stride_row_d, int stride_col_d,const  float* L, int stride_row_l, int stride_col_l, float beta, float* C, int stride_row_c, int stride_col_c);

#ifdef __CUDACC__
__global__
#endif
void diag_lower_gemm_ldl_f32(int n, float alpha, const float* D_h,const  float* L_h, float beta, float* C_h);

#ifdef __CUDACC__
__global__
#endif
void k_upper_inverse_ldl_f32(int n, const float* A, int stride_row_a,int stride_col_a, float* C, int stride_row_c, int stride_col_c);

#ifdef __CUDACC__
__global__
#endif
void k_printmatrix(const char* name, int n, int m,const float* A, int stride_row_a, int stride_col_a);

#ifdef __CUDACC__
__global__
#endif
void k_choi_solve_lower_f32(int n, int m, const float* L, int stride_row_l,int stride_col_l, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x);

#ifdef __CUDACC__
__global__
#endif
void k_choi_single_f32(int n, float* A, int stride_row, int stride_col);

#ifdef __CUDACC__
__host__
#endif
void upper_inverse_ldl_f32_device(int n, const float* A_d, int stride_row_a,int stride_col_a, float* C_d, int stride_row_c, int stride_col_c);

#ifdef __CUDACC__
__host__
#endif
void solve_lower_f32_device(int n, int m, float* L_d, int stride_row_l,int stride_col_l, float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x);

#ifdef __CUDACC__
__host__
#endif
void solve_upper_f32_v1_device(int n, const float* U_d, int stride_row_u,int stride_col_u, const float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x);

#ifdef __CUDACC__
__host__
#endif
void solve_upper_f32_v1_device(int n, const float* U_d, int stride_row_u,int stride_col_u, const float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x);

#ifdef __CUDACC__
__host__
#endif
void solve_lower_f32_v1_device(int n, int m, const float* L_d, int stride_row_l,int stride_col_l, const float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x);

#ifdef __CUDACC__
__host__
#endif
void choi_f32_device(int n, float* A_d, int stride_row, int stride_col);

#ifdef __CUDACC__
__host__
#endif
void choi_solve_f32_device(int n, int m, const float* A_d, int stride_row_a,int stride_col_a, const float* B_d, int stride_row_b, int stride_col_b, float* X_d, int stride_row_x, int stride_col_x);