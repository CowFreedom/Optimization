#pragma once

#ifdef __CUDACC__
__global__
#endif
void k_lu_f32(int n, float* A, int stride_row, int stride_col);

#ifdef __CUDACC__
__host__
#endif
void lu_solve_f32_device(int n, int m, float* A_d, int stride_row_a, int stride_col_a, float* B_d, int stride_row_b, int stride_col_b, float* X_d, int stride_row_x, int stride_col_x);