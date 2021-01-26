#pragma once

__global__
void k_gmv_f32(int n, int m, float alpha, float* A, int stride_row_a, int stride_col_a, float* x, int stride_x, float beta, float* C, int stride_c);

__host__
void gmv_f32_device(int n, int m, float alpha, float* A_d, int stride_row_a, int stride_col_a, float* x, int stride_x, float beta, float* C_d, int stride_c);

__host__
void gmv_f32(int n, int m, float alpha, float* A_h, int stride_row_a, int stride_col_a, float* x_h, int stride_x, float beta, float* C_h, int stride_c);