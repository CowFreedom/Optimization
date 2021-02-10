#pragma once

#ifdef __CUDACC__
__global__
#endif
void k_gemm_f32(float alpha, const float* A, int stride_row_a, int stride_col_a,const  float* B, int stride_row_b, int stride_col_b,float* C, int stride_row_c, int stride_col_c);

#ifdef __CUDACC__
__global__
#endif
void k_scal_f32(int m, int n, float beta, float* C, int stride_row_c, int stride_col_c);

#ifdef __CUDACC__
__host__
#endif
void gemm_f32_blockmultiple(int m, int n, int k, float alpha, const float* A_h, const float* B_h, float beta, float* C_h);

#ifdef __CUDACC__
__host__
#endif
void gemm_f32_nonblockmultiple(int m, int n, int k, float alpha, const float* A_h, const float* B_h, float beta, float* C_h);

//General matrix-to-matrix multiplication for 32 bit floats. Input matrices are padded if they are not a multiple of block size bsmx and bsmy
#ifdef __CUDACC__
__host__
#endif
void gemm_f32(int m, int n, int k, float alpha, const float* A_h, const float* B_h, float beta, float* C_h);

//General matrix-to-matrix multiplication for 32 bit floats. This assumes that the input parameters are already allocated in device memory
#ifdef __CUDACC__
__host__
#endif
void gemm_f32_device(int m, int n, int k, float alpha, const float* A_d, int stride_row_a, int stride_col_a, const float* B_d, int stride_row_b, int stride_col_b, float beta, float* C_d,int stride_row_c, int stride_col_c);

