#include "gemm.cuh"
#include <stdio.h>


//assumes that diagonal matrix has diagonal of one in reality (see ldl algorithm packed storage)
__global__
void k_choi_diag_lower_gemm_f32(int n, float alpha, const float* D, int stride_row_d, int stride_col_d,const  float* L, int stride_row_l, int stride_col_l, float beta, float* C, int stride_row_c, int stride_col_c){
	const int BLOCK_WIDTH=256; //size of a block
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	
	int col=bx*BLOCK_WIDTH+tx;
	
	if (col<n){
		float d;
		int index;
		for (int i=n-1;i>col;i--){
			d=D[i*stride_col_c+i*stride_row_c];
		
			index=i*stride_col_c+col;
			C[index]*=beta;
			C[index]+=alpha*d*L[index];
		}
		index=col*stride_col_c+col;
		C[index]*=beta;
		C[index]+=alpha*d;
	}
}

//Multiplies D*L
__global__
void k_choi_dl_gemm(int n, int k, const float* D, int stride_row_d, int stride_col_d, const float* L, int stride_row_l, int stride_col_l, float* C, int stride_row_c, int stride_col_c){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;	
	
	int row=bx*TILE_SIZE+tx;
	if (row<n){
		float diag=D[row*stride_row_d+row*stride_col_d];
		for (int i=0;i<k;i++){
			C[row*stride_col_c+i*stride_row_c]=L[row*stride_col_l+i*stride_row_l]*diag;
		}
	}
}

//Multiplies L*D=A whereas L is a lower triangular and D a diagonal matrix. Expects LD to be stored in choi form. Input matrices are padded if they are not a multiple of block size bsmx and bsmy
__host__
void diag_lower_gemm_ldl_f32(int n, float alpha, const float* D_h,const  float* L_h, float beta, float* C_h){
	float* D_d;
	float* L_d;
	float* C_d;
	
	float bsmx=256; //blocksize x
	
	int sizeC=sizeof(float)*n*n;

	cudaMalloc((void**) &C_d, sizeC);

	dim3 threadLayout=dim3(bsmx,1,1);
	dim3 grid=dim3(ceil(n/bsmx),1,1);

	int sizeD=sizeof(float)*n*n;
	int sizeL=sizeof(float)*n*n;	
	
	cudaMalloc((void**) &D_d,sizeD);
	cudaMalloc((void**) &L_d,sizeL);
	
	cudaError_t copy1=cudaMemcpy((void*) D_d, (void*) D_h, sizeD,cudaMemcpyHostToDevice);
	cudaError_t copy2=cudaMemcpy((void*) L_d, (void*) L_h, sizeL,cudaMemcpyHostToDevice);
	
	if ((copy1==cudaSuccess)&& (copy2==cudaSuccess)){
		k_choi_diag_lower_gemm_f32<<<grid,threadLayout>>> (n, alpha, D_d, 1,n,L_d,1,n,beta,C_d,1,n);		
		cudaFree(D_d);
		cudaFree(L_d);		
	}	

	cudaMemcpy((void*) C_h, (void*) C_d,sizeC,cudaMemcpyDeviceToHost);
	cudaFree(C_d);
}

//Solves LX=A, whereas L is a lower triangular matrix and X an unknown Matrix
__global__
void k_solve_lower_f32(int n, int m, const float* L, int stride_row_l,int stride_col_l, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
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
			sum/=L[i*stride_col_l+i*stride_row_l];
			X[bpx+i*stride_col_x]=sum;
		}
	}
}

//Solves (LD)X=A, whereas L is a lower triangular matrix, D a diagonal matrix and X an unknown Matrix. Assumes the following:
//L has ones along the diagonal
//
__global__
void k_choi_solve_lower_f32(int n, int m, const float* L, int stride_row_l,int stride_col_l, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int bpx=(bx*TILE_SIZE+tx)*stride_row_x;
	int bpa=(bx*TILE_SIZE+tx)*stride_row_a;//weitermachen
	if ((bx*TILE_SIZE+tx)<m){
		
		for (int i=0;i<n;i++){
			float sum=A[bpa+i*stride_col_a];
			for (int j=0;j<i;j++){
				sum-=L[i*stride_col_l+j*stride_row_l]*L[j*stride_col_l+j*stride_row_l]*X[bpx+j*stride_col_x];
			}
			sum/=L[i*stride_col_l+i*stride_row_l];
			X[bpx+i*stride_col_x]=sum;
		}
	}
}


__global__
void k_solve_lower_f32_temp(int n, const float* L, int stride_row_l,int stride_col_l, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	for (int a=0;a<n;a++){
		int bpx=(bx*TILE_SIZE+tx+a)*stride_row_x;
		int bpa=(bx*TILE_SIZE+tx+a)*stride_row_a;//weitermachen
		if ((bx*TILE_SIZE+tx)<n){
			for (int i=0;i<n;i++){
				float sum=A[bpa+i*stride_col_a];
				for (int j=0;j<i;j++){
					sum-=L[i*stride_col_l+j*stride_row_l]*X[bpx+j*stride_col_x];
				}
				sum/=L[i*stride_col_l+i*stride_row_l];
				X[bpx+i*stride_col_x]=sum;
			}
		}	
	}
}

//Solves UX=A, whereas U is a upprt triangular matrix and X an unknown Matrix
__global__
void k_solve_upper_f32(int n, int m,const float* U, int stride_row_u,int stride_col_u, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int bpx=(bx*TILE_SIZE+tx)*stride_row_x;
	int bpa=(bx*TILE_SIZE+tx)*stride_row_a;
	if ((bx*TILE_SIZE+tx)<m){
		for (int i=n-1;i>=0;i--){
			float sum=A[bpa+i*stride_col_a];
			
			for (int j=n-1;j>i;j--){
			//printf("index U %d index X %d\n",i*stride_col_u+j*stride_row_u,bpx+j*stride_col_x);
				sum-=U[i*stride_col_u+j*stride_row_u]*X[bpx+j*stride_col_x];
				
			}
			sum/=U[i*stride_col_u+i*stride_row_u];
			X[bpx+i*stride_col_x]=sum;
		}
	}
}


//Solves UX=A, whereas U is a upper triangular matrix coming from ldl decomposition and X an unknown Matrix
__global__
void k_choi_solve_upper_f32(int n, int m,const float* U, int stride_row_u,int stride_col_u, const float* A, int stride_row_a, int stride_col_a, float* X, int stride_row_x, int stride_col_x){
	const int TILE_SIZE=64;
	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int bpx=(bx*TILE_SIZE+tx)*stride_row_x;
	int bpa=(bx*TILE_SIZE+tx)*stride_row_a;
	if ((bx*TILE_SIZE+tx)<m){
		for (int i=n-1;i>=0;i--){
			float sum=A[bpa+i*stride_col_a];
			
			for (int j=n-1;j>i;j--){
			//printf("index U %d index X %d\n",i*stride_col_u+j*stride_row_u,bpx+j*stride_col_x);
				sum-=U[i*stride_col_u+j*stride_row_u]*X[bpx+j*stride_col_x];
				
			}
			X[bpx+i*stride_col_x]=sum;
		}
	}
}



__global__
void k_upper_inverse_ldl_f32(int n, const float* A, int stride_row_a,int stride_col_a, float* C, int stride_row_c, int stride_col_c){
	for (int i=n-1; i>= 0; i--){
		float factor;
		for (int j=i;j<n;j++){
			if (j!=i){
				float sum=0.0;
				for (int k=i;k<j;k++){
					sum-=A[k*stride_col_a+j*stride_row_a]*C[i*stride_col_c+k*stride_row_c];
					//printf("j:%d\n",j);
					//printf("%f times %f",A[i*stride_col_a+j*stride_row_a],C[i*stride_col_c+k*stride_row_c]);
				}
				C[i*stride_col_c+j*stride_row_c]=sum*C[j*stride_col_c+j*stride_row_c];
			}
			else{
				factor=A[i*stride_col_a+i*stride_row_a];	
				C[i*stride_col_c+i*stride_row_c]=1/factor;
			}
		
		}
	
	}
}

__global__
void k_printmatrix(const char* name, int n, int m, const float* A, int stride_row_a, int stride_col_a){
	//printf("matrix:%name\n",name);
	printf("matrix:\n");
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			printf("%.7g \t",A[i*stride_col_a+j*stride_row_a]);
		}
		printf("\n");
	}
}

__global__
void k_choi_single_f32(int n, float* A, int stride_row, int stride_col){

	for (int j=0;j<n;j++){
		float sum=0.0;
		for (int i=0;i<j;i++){
			sum-=A[j*stride_col+i*stride_row]*A[j*stride_col+i*stride_row]*A[i*stride_col+i*stride_row];
		}
		//printf("sum:%f vs. %f and sum %f and diagonal %d\n",sum,A[j*stride_col+j*stride_row],sum+A[j*stride_col+j*stride_row],j);
		A[j*stride_col+j*stride_row]+=sum;
		float D_inv=1.0/A[j*stride_col+j*stride_row];
		for (int i=j+1;i<n;i++){
			float sum=0.0;
			
			for (int t=0;t<j;t++){
				sum-=A[i*stride_col+t*stride_row]*A[j*stride_col+t*stride_row]*A[t*stride_col+t*stride_row];
			}
			
			A[i*stride_col+j*stride_row]+=sum;
			A[i*stride_col+j*stride_row]*=D_inv;
			A[j*stride_col+i*stride_row]=0.0; //can be removed if we do not want to have zeros on upper triangular part
		}
	
	}
}

__global__
void k_dcopy(int m,int n, float* source, int stride_row_source, int stride_col_source, float* dest, int stride_row_dest, int stride_col_dest){
	int TILE_SIZE=32;
	int bx=blockIdx.x;
	int tx=threadIdx.x;
	int row=bx*TILE_SIZE+tx;
	if (row<m){
		for (int i=0;i<n;i++){
			dest[row*stride_col_dest+i*stride_row_dest]=source[row*stride_col_source+i*stride_row_source];
		}
	}
}

//Solves LX=A, whereas L is a lower triangular matrix and X an unknown Matrix
__host__
void solve_lower_f32_v1_device(int n, int m, const float* L_d, int stride_row_l,int stride_col_l, const float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x){
	float bsmx=64; //blocksize x
	dim3 threadLayout=dim3(bsmx,1,1);
	dim3 grid=dim3(ceil(m/bsmx),1,1);
	k_solve_lower_f32<<<grid,threadLayout>>>(n,m, L_d, stride_row_l,stride_col_l, A_d, stride_row_a,stride_col_a, X_d, stride_row_x, stride_col_x);
}

/*Solves LX=A, whereas L is a lower triangular matrix with dimension nxn and an unknown Matrix X with dimension nxm
Transforms value matrix A 
*/
__host__
void solve_lower_f32_device(int n, int m, float* L_d, int stride_row_l,int stride_col_l, float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x){
	int d=3;
	int rem=n%d;
	int k=n/d;
	float bsmx=64; //blocksize x

	float* A1=A_d;
	float* A2=A1+d*stride_col_a;
	float* L11=L_d;
	float* L21=L11+d*stride_col_l;
	float* L22=L21+d*stride_row_l;
	
	float* X1=X_d;
	for (int i=0;i<k;i++){
		dim3 threadLayout=dim3(bsmx,1,1);
		dim3 grid=dim3(ceil(m/bsmx),1,1);
		//k_printmatrix<<<1,1>>>("L_d",d,d,L11,stride_row_l,stride_col_l);		
		k_solve_lower_f32<<<grid,threadLayout>>>(d,m, L11, stride_row_l,stride_col_l, A1, stride_row_a,stride_col_a, X1, stride_row_x, stride_col_x);		
		gemm_f32_device(n-d, m, d, -1.0, L21, stride_row_l,stride_col_l,X1, stride_row_x,stride_col_x, 1.0, A2,stride_row_a, stride_col_a);
		//k_printmatrix<<<1,1>>>("A_d",n-d,m,A2,stride_row_a,stride_col_a);		
		n-=d;
		X1+=d*stride_col_x;
		L11=L22;
		L21=L11+d*stride_col_l;
		L22=L21+d*stride_row_l;
		A1=A2;
		A2+=d*stride_col_a;
	}

	if (rem!=0){
		d=rem;
		dim3 threadLayout=dim3(bsmx,1,1);
		dim3 grid=dim3(ceil(m/bsmx),1,1);
		printf("d:%d\n",d);
		k_solve_lower_f32<<<grid,threadLayout>>>(d,m, L11, stride_row_l,stride_col_l, A1, stride_row_a,stride_col_a, X1, stride_row_x, stride_col_x);		
	}
	
}

/*Solves L*D*X=A, whereas L is a lower triangular matrix with dimension nxn, D a diagonal matrix and an unknown Matrix X with dimension nxm
Transforms value matrix A.
Might not be useful, because we need a domain specific function gemdm that multiplies three matrices
*/
/*
__host__
void choi_solve_lower_f32_device(int n, int m, float* L_d, int stride_row_l,int stride_col_l, float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x){
	int d=3;
	int rem=n%d;
	int k=n/d;
	float bsmx=64; //blocksize x
	dim3 threadLayout=dim3(bsmx,1,1);
	dim3 grid=dim3(ceil(m/bsmx),1,1);

	float* A1=A_d;
	float* A2=A1+d*stride_col_a;
	float* L11=L_d;
	float* L21=L11+d*stride_col_l;
	float* L22=L21+d*stride_row_l;
	float* X1=X_d;
	for (int i=0;i<k;i++){
		//k_printmatrix<<<1,1>>>("L_d",d,d,L11,stride_row_l,stride_col_l);			
		k_choi_solve_lower_f32<<<grid,threadLayout>>>(d,m, L11, stride_row_l,stride_col_l, A1, stride_row_a,stride_col_a, X1, stride_row_x, stride_col_x);		
		//k_printmatrix<<<1,1>>>("L21",n-d,d,L21,stride_row_l,stride_col_l);
		//k_printmatrix<<<1,1>>>("X",d,m,X1,stride_row_x,stride_col_x);
		//k_printmatrix<<<1,1>>>("A2",n-d,m,A2,stride_row_a,stride_col_a);
		gemdm_f32_device(n-d, m, d, -1.0, L21, stride_row_l,stride_col_l,D,stride_row_d,stride_col_d,X1, stride_row_x,stride_col_x, 1.0, A2,stride_row_a, stride_col_a);
		//k_printmatrix<<<1,1>>>("A2",n-d,m,A2,stride_row_a,stride_col_a);
		//k_printmatrix<<<1,1>>>("A_d",4,8,A_d,stride_row_a,stride_col_a);		
		n-=d;
		X1+=d*stride_col_x;
		L11=L22;
		L21=L11+d*stride_col_l;
		L22=L21+d*stride_row_l;
		A1=A2;
		A2+=d*stride_col_a;
	}

	if (rem!=0){
		d=rem;
		//printf("d:%d\n",d);
		k_choi_solve_lower_f32<<<grid,threadLayout>>>(d,m, L11, stride_row_l,stride_col_l, A1, stride_row_a,stride_col_a, X1, stride_row_x, stride_col_x);		
	}
	printf("done choi solve\n");
}
*/


//Solves LX=A, whereas L is a lower triangular matrix and X an unknown Matrix
__host__
void upper_inverse_ldl_f32_device(int n, const float* A_d, int stride_row_a,int stride_col_a, float* C_d, int stride_row_c, int stride_col_c){
	 k_upper_inverse_ldl_f32<<<1,1>>>(n, A_d, stride_row_a,stride_col_a,C_d,stride_row_c,stride_col_c);
}


//Solves UX=A, whereas L is  lower triangular matrix and X an unknown Matrix
__host__
void solve_lower_f32_v2_device(int n, const float* L_d, int stride_row_l,int stride_col_l, const float* A_d, int stride_row_a, int stride_col_a, float* X_d, int stride_row_x, int stride_col_x){
	k_solve_lower_f32_temp<<<1,1>>>(n, L_d, stride_row_l,stride_col_l, A_d, stride_row_a,stride_col_a, X_d, stride_row_x, stride_col_x);
}

//Solves AX=B for matrices A,X and B. A is in LDL format
__host__
void choi_solve_f32_device(int n, int m, const float* A_d, int stride_row_a,int stride_col_a, const float* B_d, int stride_row_b, int stride_col_b, float* X_d, int stride_row_x, int stride_col_x){
	float bsmx=64;
	float* Y;
	cudaMalloc((void**)&Y,sizeof(float)*n*m);
//	printf("bx: %f\n",ceil(m/bsmx));
//	k_printmatrix<<<1,1>>>("A",n,n,A_d,1,n);
	k_choi_solve_lower_f32<<<ceil(m/bsmx),bsmx>>>(n,m,A_d,stride_row_a,stride_col_a,B_d,stride_row_b,stride_col_b,Y,1,m);
//	k_printmatrix<<<1,1>>>("Y",n,m,Y,1,m);
	k_choi_solve_upper_f32<<<ceil(m/bsmx),1>>>(n,m,A_d,stride_row_a,stride_col_a,Y,1,m,X_d,stride_row_x,stride_col_x); //replace with non kernel version
	cudaFree(Y);
}

__host__
void choi_f32_device(int n, float* A_d, int stride_row, int stride_col){

	int BLOCKSIZE=32;
	int d;
	int q;
	int rem;
	
	d=BLOCKSIZE;
	q=n/d;
	rem=n%d;
	
	float* A11=A_d;
	float* A21=A_d+d*stride_col;
	
	
	int sizeT1=sizeof(float)*d*d;
	int sizeT2=sizeof(float)*n*(n-d);
	float* temp1;
	float* temp2;
	
	cudaMalloc((void**) &temp1,sizeT1);
	cudaMalloc((void**) &temp2,sizeT2);
	cudaMemset(temp1,0,sizeT1);
	cudaMemset((void**)&temp2,0,sizeT2);
	
	for (int i=0;i<q;i++){
	
		//Calculate L11
		k_choi_single_f32<<<1,1>>>(d, A11,stride_row, stride_col); 
		
		//Calculate L21
		k_dcopy<<<ceil((n-d)/32.0),32>>>(n-d,d,A21,stride_row,stride_col,temp2,n-d,1);
		//k_printmatrix<<<1,1>>>("temp2", d, n-d, temp2, 1, n-d);
		//choi_solve_lower_f32_device(d, n-d, A11, stride_row,stride_col, temp2, 1, n-d, A21, stride_col, stride_row);
		k_choi_solve_lower_f32<<<ceil((n-d)/64.0),64>>>(d, n-d, A11, stride_row,stride_col, temp2, 1, n-d, A21, stride_col, stride_row);

		//Calculate L22
		k_choi_dl_gemm<<<ceil(d/64.0),64>>>(d, n-d, A11, stride_row, stride_col, A21, stride_col,stride_row,temp2, 1, n-d);
		A11+=d*stride_row+d*stride_col;
		gemm_f32_device(n-d, n-d, d, -1.0, A21, stride_row, stride_col, temp2, 1, n-d, 1.0, A11,stride_row, stride_col);
		A21=A11+d*stride_col;
		n-=d;
	}
	
	if (rem!=0){
		d=rem;
		k_choi_single_f32<<<1,1>>>(d, A11,stride_row, stride_col); 
	}

}

void ldl(float* ){


}