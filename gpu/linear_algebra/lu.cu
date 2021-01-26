#include <stdio.h>
__global__
void k_lu_f32(int n, float* A, int stride_row, int stride_col){
	for (int i=0;i<n;i++){
		for (int j=i+1;j<n;j++){
			float factor=-A[j*stride_col+i*stride_row]/A[i*stride_col+i*stride_row];
			for (int k=i+1;k<n;k++){
				A[j*stride_col+k*stride_row]=A[j*stride_col+k*stride_row]+A[i*stride_col+k*stride_row]*factor;
			}
			A[j*stride_col+i*stride_row]=-factor;
		}
	}
}


//Solves LX=A, whereas L is a lower triangular matrix and X an unknown Matrix. It is assumed that L is stored inside A=LU with ones on the diagonal
__global__
void k_lu_solve_lower_f32(int n, int m, const float* L, int stride_row_l,int stride_col_l, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int bpx=(bx*TILE_SIZE+tx)*stride_row_x;
	int bpa=(bx*TILE_SIZE+tx)*stride_row_a;//weitermachen
	if ((bx*TILE_SIZE+tx)<m){
		for (int i=0;i<n;i++){
			float sum=A[bpa+i*stride_col_a];
			for (int j=0;j<i;j++){
				sum-=L[i*stride_col_l+j*stride_row_l]*X[bpx+j*stride_col_x];
			}
			X[bpx+i*stride_col_x]=sum;
		}
	}
}

//Solves UX=A, whereas U is a upprt triangular matrix coming from lu factorization and X an unknown Matrix
__global__
void k_lu_solve_upper_f32(int n, int m,const float* U, int stride_row_u,int stride_col_u, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int bpx=(bx*TILE_SIZE+tx)*stride_row_x;
	int bpa=(bx*TILE_SIZE+tx)*stride_row_a;
	if ((bx*TILE_SIZE+tx)<m){
		for (int i=n-1;i>=0;i--){
			float sum=A[bpa+i*stride_col_a];
			
			for (int j=n-1;j>i;j--){
				sum-=U[i*stride_col_u+j*stride_row_u]*X[bpx+j*stride_col_x];
				
			}
			sum/=U[i*stride_col_u+i*stride_row_u];
			//printf("sum:%.7f and U:%.g and i%d\n",sum,U[i*stride_col_u+i*stride_row_u], i);
			X[bpx+i*stride_col_x]=sum;
		}
	}
}

//Solves AX=B
__host__
void lu_solve_f32_device(int n, int m, float* A_d, int stride_row_a, int stride_col_a, float* B_d, int stride_row_b, int stride_col_b, float* X_d, int stride_row_x, int stride_col_x){
	float bsmx=64;
	float* Y;
	cudaMalloc((void**)&Y,sizeof(float)*n*m);
	k_lu_solve_lower_f32<<<ceil(m/bsmx),bsmx>>>(n,m,A_d,stride_row_a,stride_col_a,B_d,stride_row_b,stride_col_b,Y,1,m);
	k_lu_solve_upper_f32<<<ceil(m/bsmx),1>>>(n,m,A_d,stride_row_a,stride_col_a,Y,1,m,X_d,stride_row_x,stride_col_x); //TODO: replace with non kernel version
	cudaFree(Y);
}