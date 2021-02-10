#include <stdio.h>

/*Kernel for matrix outer product*/
__global__
void k_gemm_f32(float alpha, const float* A, int stride_row_a, int stride_col_a,const  float* B, int stride_row_b, int stride_col_b,float* C, int stride_row_c, int stride_col_c){
	
	const int TILE_WIDTH=16;
	const int VEC_SIZE=4; //multiplies TILE_WIDTH for the b row vectors. 4 means one thread calculated C's entries with a rowlength of 4*16=64 numbers in B. Must be multiple of TILE_WIDTH
	
	float Cc[TILE_WIDTH]={0}; //initializes all elements to zero
	__shared__ float Ac[TILE_WIDTH*TILE_WIDTH]; //buffer that holds columns of a
	
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	
	int a_begin=by*TILE_WIDTH*stride_col_a;
	int a_end=a_begin+stride_col_a;//check if correct
	int b_begin=bx*TILE_WIDTH*VEC_SIZE*stride_row_b; //we multiply by VEC_SIZE because B's tiles have length TILEWIDTH*VEC_SIZE
	
	for (;a_begin < a_end;a_begin+=TILE_WIDTH*stride_row_a){
		//Load elements of A into shared memory
		
		for (int i=0; i< 4;i++){
			Ac[i*4+ty+TILE_WIDTH*tx]=A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a];
		}
			
		__syncthreads();
		
		const float* ptrB=&B[b_begin+(TILE_WIDTH*ty+tx)*stride_row_b];
		float* ptrA=Ac;
		
		#pragma unroll
		for (int i=0;i<TILE_WIDTH;i++){
			float bv=alpha*ptrB[0];
			
			//this loop could be unrolled
			for (int j=0;j<TILE_WIDTH;j++){
				Cc[j]+=ptrA[j]*bv;
			}
			ptrA+=TILE_WIDTH; //next column of A (it is the next column because Ac is a transposed block of A)
			ptrB+=stride_col_b;
		}
		b_begin+=TILE_WIDTH*stride_col_b;
		__syncthreads();
	}
	
	int c=stride_col_c*TILE_WIDTH*by+(TILE_WIDTH*VEC_SIZE*bx+tx+TILE_WIDTH*ty)*stride_row_c;
	
	for (int i=0;i<TILE_WIDTH;i++){
		C[c]+=Cc[i];
		c+=stride_col_c;
	}	
	
}

//Todo!!
/*Kernel for matrix outer product. This version does not require A,B,C to be multiples of the blocksizes*/
__global__
void k_gemm_f32_nonblockmultiple(const int m, const int n, const int k,float alpha, const float* A, int stride_row_a, int stride_col_a,const  float* B, int stride_row_b, int stride_col_b,float* C, int stride_row_c, int stride_col_c){
	
	const int TILE_WIDTH=16;
	const int VEC_SIZE=4; //multiplies TILE_WIDTH for the b row vectors. 4 means one thread calculated with a rowlength of 4*16=64 numbers in B. Must be multiple of TILE_WIDTH
	
	float Cc[TILE_WIDTH]={0}; //initializes all elements to zero
	__shared__ float Ac[TILE_WIDTH*TILE_WIDTH]; //buffer that holds columns of a
	
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int qm=m%TILE_WIDTH;
	//int qn=(VEC_SIZE*TILE_WIDTH)%n;
	int qk=k%TILE_WIDTH;
	int rowA=by*TILE_WIDTH;
	int colB=bx*TILE_WIDTH*VEC_SIZE+TILE_WIDTH*ty+tx;
	int a_begin=by*TILE_WIDTH*stride_col_a;
	int b_begin=bx*TILE_WIDTH*VEC_SIZE*stride_row_b; //we multiply by VEC_SIZE because B's tiles have length TILEWIDTH*VEC_SIZE
	bool does_compute=false;
	//printf("qk:%d\n",qk);
	
		int rk=k/TILE_WIDTH;
		for (int q=0;q<rk;q++){
			//Load elements of A into shared memory
			//printf("i: %d\n",a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a);

			if ((tx<k)&&((rowA+TILE_WIDTH-1)<m)){
				for (int i=0; i< 4;i++){
				//printf("Aci: %d, i: %d and A:%f\n",i*4+ty+TILE_WIDTH*tx,a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a,A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a]);
					Ac[i*4+ty+TILE_WIDTH*tx]=A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a];
					
					
				}
			}
			else{
				for (int i=0; i< 4;i++){
						if((rowA+i*4+ty)<m && (tx<k)){
						//printf("is: %f\n",A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a]);
						
							Ac[i*4+ty+TILE_WIDTH*tx]=A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a];
						//	printf("is:Ac index: %d, index: %d and A:%f\n",i*4+ty+TILE_WIDTH*tx,a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a,A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a]);
						}
						else{
							Ac[i*4+ty+TILE_WIDTH*tx]=0.0;
						}
				}	
			}
			
		/*
			for (int i=0;i<TILE_WIDTH*TILE_WIDTH;i++){
					Ac[i]=-7;
			}
		*/
			__syncthreads();
				
		/*
			if (tx==0 && ty==0){
			
				for (int i=0;i<TILE_WIDTH*TILE_WIDTH;i++){
					printf("%f\t",Ac[i]);
				}
			}	
		*/
			if (colB>=n){
				for (int j=0;j<TILE_WIDTH;j++){
					Cc[j]=0.0;
				}
			}
			else{
			//printf("Id: %d,%d,%d,%d\n",by,ty,bx,tx);
				const float* ptrB=&B[b_begin+(TILE_WIDTH*ty+tx)*stride_row_b];
				float* ptrA=Ac;
				does_compute=true;
				#pragma unroll
				for (int i=0;i<TILE_WIDTH;i++){
					float bv=alpha*ptrB[0];
				
					//this loop could be unrolled
					for (int j=0;j<TILE_WIDTH;j++){
						Cc[j]+=ptrA[j]*bv;
						
					/*	if (ptrA[j]!=0){
					printf("%f vs. %f\n",ptrA[j],bv);
					}
					*/
					}
					ptrA+=TILE_WIDTH; //next column of A (it is the next column because Ac is a transposed block of A)
					ptrB+=stride_col_b;
				}
				b_begin+=TILE_WIDTH*stride_col_b;
			}
		a_begin+=TILE_WIDTH*stride_row_a;
		__syncthreads();
	}

	if (qk>0){
	
		if (tx<qk){
				//printf("rowA:%d, ty:%d\n",rowA,ty);
			a_begin=(by*TILE_WIDTH*stride_col_a)+rk*TILE_WIDTH*stride_row_a;
			for (int i=0; i< 4;i++){
				if((rowA+i*4+ty)<m){
					Ac[i*4+ty+TILE_WIDTH*tx]=A[a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a];
					//printf("Ac index2: %d, index: %d and \n",i*4+ty+TILE_WIDTH*tx,a_begin+stride_col_a*(i*4+ty)+tx*stride_row_a);
					
					
				}
				else{
					Ac[i*4+ty+TILE_WIDTH*tx]=0.0;
				}
			}						
		}
		else{
			for (int i=0; i< 4;i++){
				Ac[i*4+ty+TILE_WIDTH*tx]=0.0;
			}
		}
		__syncthreads();
		//return;
		if (colB<n){
		//	printf("Id: %d,%d,%d,%d\n",by,ty,bx,tx);
			const float* ptrB=&B[b_begin+(TILE_WIDTH*ty+tx)*stride_row_b];
			float* ptrA=Ac;
			does_compute=true;

			for (int i=0;i<qk;i++){
				float bv=alpha*ptrB[0];
			
				//this loop could be unrolled
				for (int j=0;j<TILE_WIDTH;j++){
					Cc[j]+=ptrA[j]*bv;
					
					/*if (ptrA[j]!=0){
					printf("%f vs2. %f\n",ptrA[j],bv);
					}
					*/
				
				}
				ptrA+=TILE_WIDTH; //next column of A (it is the next column because Ac is a transposed block of A)
				ptrB+=stride_col_b;
			}				
		}

	}
	__syncthreads(); //maybe redundant
	if (does_compute){
		int c=stride_col_c*TILE_WIDTH*by+(TILE_WIDTH*VEC_SIZE*bx+tx+TILE_WIDTH*ty)*stride_row_c;
		int c_length=((rowA+TILE_WIDTH)<=m)?TILE_WIDTH:qm;

		for (int i=0;i<c_length;i++){
			C[c]+=Cc[i];
			c+=stride_col_c;
		}	
	}
	
}


__global__
void k_scal_f32(int m, int n, float beta, float* C, int stride_row_c, int stride_col_c){
	const int BLOCK_WIDTH=256; //size of a block
	const int TILE_WIDTH=64; //size of block per single thread
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	//printf("Bin drin mit : bx %d, tx %d, by %d, ty %d \n",bx,tx,by,ty);
	float* c_begin=&C[(by*BLOCK_WIDTH+ty*TILE_WIDTH)*stride_col_c+(bx*BLOCK_WIDTH+tx*TILE_WIDTH)*stride_row_c];
	
	if ((((by+1)*BLOCK_WIDTH)<=m) && (((bx+1)*BLOCK_WIDTH)<=n)){
		for (int i=0;i<TILE_WIDTH;i++){
			for (int j=0;j<TILE_WIDTH;j++){
				c_begin[i*stride_col_c+j*stride_row_c]*=beta;
			}
		}	
	}
	else{
		int column=by*BLOCK_WIDTH+ty*TILE_WIDTH;
		for (int i=0;i<TILE_WIDTH;i++){
			if (column<m){
				int row=bx*BLOCK_WIDTH+tx*TILE_WIDTH;
				for (int j=0;j<TILE_WIDTH;j++){
					if (row<n){
						c_begin[i*stride_col_c+j*stride_row_c]*=beta;
						//printf("Bin hier drin mit %d und %d mit by %d ty %d bx %d tx %d\n",i,j,by,ty,bx,tx);
					}
					row=row+1;
				}
			}
			column=column+1;
		}
	}

}


//matrix matrix multiplication
__host__
void gemm_f32_blockmultiple(int m, int n, int k, float alpha, const float* A_h, const float* B_h, float beta, float* C_h){

	float* A_d;
	float* B_d;
	float* C_d;
	int sizeA=sizeof(float)*m*k;
	int sizeB=sizeof(float)*n*k;
	int sizeC=sizeof(float)*m*n;
	
	float bsmx=16;
	float bsmy=4;
	
	dim3 threadLayout=dim3(bsmx,bsmy,1);
	dim3 grid=dim3(ceil(n/(4.0*bsmx)),ceil(m/bsmx),1);	
	
	cudaMalloc((void**) &C_d,sizeC);
	
	if (beta==0){
		cudaMemset(C_d, 0, sizeC);
	}
	else{
		cudaMemcpy((void*) C_d, (void*) C_h, sizeC,cudaMemcpyHostToDevice);
		k_scal_f32<<<grid,threadLayout>>>(m,n,beta,C_d,1,n);
	}
	
	if (alpha!=0.0){
		cudaMalloc((void**) &A_d,sizeA);
		cudaMalloc((void**) &B_d,sizeB);
		
		cudaError_t copy1=cudaMemcpy((void*) A_d, (void*) A_h, sizeA, cudaMemcpyHostToDevice);
		cudaError_t copy2=cudaMemcpy((void*) B_d, (void*) B_h, sizeB, cudaMemcpyHostToDevice);

		if ((copy1==cudaSuccess)&& (copy2==cudaSuccess)){	
			k_gemm_f32<<<grid,threadLayout>>> (alpha, A_d, 1, k,B_d,1,n,C_d,1,n);
			cudaMemcpy((void*) C_h, (void*) C_d, sizeC, cudaMemcpyDeviceToHost);
			cudaFree(A_d);
			cudaFree(B_d);
		}		
	}
	cudaFree(C_d);
	
}

//General matrix-to-matrix multiplication for 32 bit floats. Input matrices are padded if they are not a multiple of block size bsmx and bsmy
__host__
void gemm_f32_nonblockmultiple(int m, int n, int k, float alpha, const float* A_h, const float* B_h, float beta, float* C_h){
	float* A_d;
	float* B_d;
	float* C_d;
	
	float bsmx=16; //blocksize x
	float bsmy=4; //blocksize y
	
	int mB=ceil(m/bsmx)*bsmx;
	int nB=ceil(n/(4.0*bsmx))*(4.0*bsmx);
	int kB=ceil(k/bsmx)*bsmx;
	int sizeCb=sizeof(float)*mB*nB;

	cudaMalloc((void**) &C_d, sizeCb);

	dim3 threadLayout=dim3(bsmx,bsmy,1);
	dim3 grid=dim3(ceil(nB/(4.0*bsmx)),ceil(mB/bsmx),1);

	if (beta==0){
		cudaMemset(C_d, 0, sizeCb);
	}
	else{
		cudaError_t copy;
		for (int i=0;i<m;i++){
			copy=cudaMemcpy((void*) (C_d+i*nB), (void*) (C_h+i*n), sizeof(float)*n,cudaMemcpyHostToDevice);
		}
		if (copy!=cudaSuccess){
			printf("Copy fehlgeschlagen\n");
		}
	//	printf("Starte nun den Kernel\n");
		dim3 threadsize=dim3(4,4,1);
		dim3 blocksize=dim3(ceil(n/256.0),ceil(m/256.0),1);
		k_scal_f32<<<blocksize,threadsize>>>(m,n,beta,C_d,1,nB);
		//cudaDeviceSynchronize();
	}

	if (alpha!=0.0){
		int sizeAb=sizeof(float)*mB*kB;
		int sizeBb=sizeof(float)*kB*nB;	
		
		cudaMalloc((void**) &A_d,sizeAb);
		cudaMalloc((void**) &B_d,sizeBb);
		cudaMemset(A_d,0.0,sizeAb);
		cudaMemset(B_d,0.0,sizeBb);
		
		cudaError_t copy1;
		cudaError_t copy2;
		
		for (int i=0;i<m;i++){
			copy1=cudaMemcpy((void*) (A_d+i*kB), (void*) (A_h+i*k), sizeof(float)*k,cudaMemcpyHostToDevice);
		}
		for (int i=0;i<k;i++){
			copy2=cudaMemcpy((void*) (B_d+i*nB), (void*) (B_h+i*n), sizeof(float)*n, cudaMemcpyHostToDevice);
		}

		if ((copy1==cudaSuccess)&& (copy2==cudaSuccess)){
			k_gemm_f32<<<grid,threadLayout>>> (alpha, A_d, 1, kB,B_d,1,nB,C_d,1,nB);		
			cudaFree(A_d);
			cudaFree(B_d);		
		}	
	}
	
	for (int i=0;i<m;i++){
			cudaError_t copy=cudaMemcpy((void*) (C_h+i*n), (void*) (C_d+i*nB),sizeof(float)*n,cudaMemcpyDeviceToHost);
			if (copy!=cudaSuccess){
			printf("Copy fehlgeschlagen\n");
			}
	}
	cudaFree(C_d);
}

//General matrix-to-matrix multiplication for 32 bit floats. Input matrices are padded if they are not a multiple of block size bsmx and bsmy
__host__
void gemm_f32(int m, int n, int k, float alpha, const float* A_h, const float* B_h, float beta, float* C_h){
	if ((alpha==0.0) && (beta==1.0)){
		return;
	}
	int res1=m%16;
	int res2=n/(4*4);
	
	if ((res1==0)&&(res2==0)){
		gemm_f32_blockmultiple(m,n,k,alpha,A_h,B_h,beta,C_h);
	}
	else{
//	printf("nonblockmultiple\n");
		gemm_f32_nonblockmultiple(m,n,k,alpha,A_h,B_h,beta,C_h);
	}
}

//General matrix-to-matrix multiplication for 32 bit floats. This assumes that the input parameters are already allocated in device memory
__host__
void gemm_f32_device(int m, int n, int k, float alpha, const float* A_d, int stride_row_a, int stride_col_a, const float* B_d, int stride_row_b, int stride_col_b, float beta, float* C_d,int stride_row_c, int stride_col_c){
	if ((alpha==0.0) && (beta==1.0)){
		return;
	}

	float bsmx=16;
	float bsmy=4;
	dim3 threadLayout=dim3(bsmx,bsmy,1);
	dim3 grid=dim3(ceil(n/(4.0*bsmx)),ceil(m/bsmx),1);	
	
	k_scal_f32<<<grid,threadLayout>>>(m,n,beta,C_d,stride_row_c,stride_col_c);
		
		if (alpha!=0){
	
		int res1=m%(int)bsmx;
		int res2=n%(int)bsmx;
		
		if ((res1==0)&&(res2==0)){
		//	printf("gemm blockmultiple\n");
			k_gemm_f32<<<grid,threadLayout>>>(alpha, A_d, stride_row_a, stride_col_a,B_d,stride_row_b,stride_col_b,C_d,stride_row_c,stride_col_c);		
		}
		else{
		//printf("gemm nonblockmultiple\n");
			k_gemm_f32_nonblockmultiple<<<grid,threadLayout>>>(m,n,k,alpha, A_d, stride_row_a, stride_col_a,B_d,stride_row_b,stride_col_b,C_d,stride_row_c,stride_col_c);		
		}	
		
	}	
}

