module;
#include <ostream>
#include<iostream>
#include <thread>
#include <vector>
export module optimization.transformation;

#if defined(__clang__)
	#include <x86intrin.h> //SIMD for gcc/clang
#elif defined(__GNUC__) || defined(__GNUG__)
	#include <x86intrin.h> //SIMD for gcc/clang
#elif defined(_MSC_VER)
	#include<immintrin.h> //AVX, AVX2, FMA for VS
#endif

namespace opt{
	namespace math{
		namespace cpu{
			
			constexpr size_t MC = 384; //
			constexpr size_t KC = 384;
			constexpr size_t NC = 4096;

			constexpr size_t MR = 32;
			constexpr size_t NR = 32; //probleme bei dieser blockgröße, muss vielfaches von 4 sein und größer 4

			/*Returns a complete panel of dimension MRxKC, which is sliced along
			the horizontal axis MR. Assumes Matrix
			A has dimension MxK.
			For cache efficiency, the values are packed column wise into the
			buffer. See for details http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/day08/page02.html.
			Note: In the tutorial they assume that matrices are stored columns major. */
			template<class T, class F>
			void pack_MRxk(size_t k, T A, F* buffer, size_t stride_rows, size_t stride_cols) {

				for (size_t i = 0; i < k; i++) {
					for (size_t j = 0; j < MR; j++) {
						*(buffer + j) = *(A + j * stride_cols);

					}
					if (i < k - 1) {
						A += stride_rows;
						buffer += MR;
					}

				}
			}

			template<class T>
			void printmat(T v, int N, int M) {
				for (int i = 0; i < N; i++) {
					for (int j = 0; j < M; j++) {
						std::cout << *(v + j + i * M) << "\t";
					}
					std::cout << "\n";
				}

			}

			/*Returns a block of dimension MCxKC, which consists of
			multiple panels of maximum size MRxKC. Assumes Matrix
			A has dimension MxK.
			For cache efficiency, the values are packed column wise into the
			buffer. See for details http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/day08/page02.html.
			Note: In the tutorial they assume that matrices are stored columns major. */
			template<class T, class F>
			void pack_A(size_t mc, size_t kc, T A, size_t stride_rows, size_t stride_cols, F* buffer) {
				size_t r = mc % MR; //number of complete "stripes"
				size_t q = mc / MR;
							F* aux=buffer;
				for (size_t i = 0; i < q; i++) {
					pack_MRxk(kc, A, buffer, stride_rows, stride_cols);
					A += MR * stride_cols;
					buffer += kc * MR;
				}
				
				if (r != 0) {
					std::fill(buffer, buffer + MR * kc, F(0.0));
					for (size_t i = 0; i < r; i++) {
						for (size_t j = 0; j < kc; j++) {
							*(buffer + j * MR) = *(A + j * stride_rows);
							
						}
						if (i < r - 1) {
							A += stride_cols;
							buffer += 1;
						}
					}
				}
			}

			/*Returns a complete panel of dimension KCxNR, which is sliced along
			the vertical axis NR. Assumes Matrix
			B has dimension KxN.
			For cache efficiency, the values are packed column wise into the
			buffer. See for details http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/day08/page02.html.
			Note: In the tutorial they assume that matrices are stored columns major. */
			template<class T, class F>
			void pack_KXnr(size_t k, T B, F* buffer, size_t stride_rows, size_t stride_cols) {
				for (size_t i = 0; i < k; i++) {
					
					for (size_t j = 0; j < NR; j++) {
						buffer[j] = *(B + j * stride_rows);
								//std::cout << *(B + j * stride_rows) << "\n";
					}	
					
					if (i < k - 1) {
						B += stride_cols;
						buffer += NR;
					}				
				}
		
			}

			template<class T, class F>
			void pack_B(size_t kc, size_t nc, T B, size_t stride_rows, size_t stride_cols, F* buffer) {
				size_t r = nc % NR; //number of complete "stripes"
				size_t q = nc / NR;
				for (size_t i = 0; i < q; i++) {
					pack_KXnr(kc, B, buffer, stride_rows, stride_cols);
				//	std::cout << "i:" <<  i<< "\n";
					B += NR;
					buffer += kc * NR;

				}
				if (r != 0) {
					std::fill(buffer, buffer + NR * kc, F(0.0));
					for (size_t i = 0; i < kc; i++) {
						for (size_t j = 0; j < r; j++) {
							buffer[j] = *(B + j * stride_rows);
						}
						if (i < kc - 1) {
							B += stride_cols;
							buffer += NR;
						}		
					}
				}
				
			}
			
			//A is in column major form and B is in row major form (i.e. A was packed by the function pack_A and B by pack_B)

			template<class T>
			void dgemm_micro_kernel(size_t kc, double alpha, const double* A, const double* B, double beta, T C, size_t stride_row_c, int stride_col_c) {
				//std::cout<<A[0]<<"\t"<<A[1]<<"\n"<<A[MR]<<"\t"<<A[MR+1]<<"\n";

				//double* AB =(double*)_aligned_malloc(NR*MR*sizeof(double), 64);
			//	double AB[NR*MR] __attribute__ ((aligned (16)));
				double AB[NR*MR];
				
				
				std::fill(AB, AB + MR * NR, double(0.0));
				
				for (size_t k = 0; k < kc; k++) {
					for (size_t j = 0; j < MR; j++) {
						//__m256d ymm0 = _mm256_loadu_pd(A + j);
						__m256d ymm0 = _mm256_broadcast_sd(A+j); //load scalar
						for (size_t i = 0; i < NR; i += 4) {
							size_t inc = j * NR + i;

							__m256d  ymm1 = _mm256_loadu_pd(B + i);
							__m256d ymm2 = _mm256_loadu_pd(AB+inc); //result
							ymm2=_mm256_fmadd_pd(ymm0, ymm1, ymm2);//FMA fused multiply instruction						
							_mm256_storeu_pd(AB+inc, ymm2);
						}
					}
					if (k < KC - 1) {
						A += MR;
						B += NR;
					}
				}
				
				
				
				/*
				for (size_t k = 0; k < kc; k++) {
					for (size_t j = 0; j < MR; j+=4) {
						__m256d  tmp1 = _mm256_loadu_pd(A+j);	
						double tmp[]={0.0,0.0,0.0,0.0};
						for (size_t i = 0; i < NR; i += 4) {
							size_t inc = j * NR + i;
							double* out=AB+inc;
							__m256d  tmp2 = _mm256_loadu_pd(B+i);			
							
							__m256d tmp3=_mm256_mul_pd(tmp1,tmp2);	
							_mm256_storeu_pd(tmp, tmp3);

							out[0]+=tmp[0];
							out[NR+1]+=tmp[1];
							out[2*NR+2]+=tmp[2];
							out[3*NR+3]+=tmp[3];

							__m256d tmp4 = _mm256_permute_pd (tmp2,0b0101);	// Permutation (1,2,3,4)->(2,1,4,3)
							tmp3=_mm256_mul_pd(tmp1,tmp4);	
							_mm256_storeu_pd(tmp, tmp3);	

							out[1]+=tmp[0];
							out[NR]+=tmp[1];
							out[2*NR+3]+=tmp[2];
							out[3*NR+2]+=tmp[3];

							tmp4 = _mm256_permute4x64_pd(tmp2,0b00011011);	// Permutation (1,2,3,4)->(4,3,2,1). This instruction is roughly three times slower than the other permute ones
							tmp3=_mm256_mul_pd(tmp1,tmp4);	
							_mm256_storeu_pd(tmp, tmp3);	

							out[3]+=tmp[0];
							out[NR+2]+=tmp[1];
							out[2*NR+1]+=tmp[2];
							out[3*NR]+=tmp[3];

							tmp3=_mm256_permute_pd(tmp4,0b0101);	// Permutation (4,3,2,1)->(4,3,1,2)
							tmp4=_mm256_mul_pd(tmp1,tmp3);	
							_mm256_storeu_pd(tmp, tmp4);

							out[2]+=tmp[0];
							out[NR+3]+=tmp[1];
							out[2*NR]+=tmp[2];
							out[3*NR+1]+=tmp[3];
						}
					}
					
					if (k < KC - 1) {
						A += MR;
						B += NR;
					}
				}
				*/

				
				//
				//  Update C <- beta*C
				//

				if (beta == double(0.0)) {
					//std::fill(C,C+MR*NR,F(0.0)); //geht nicht, da C größer ist als MRxNR
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] = 0.0;
						}
					}
				}
				else if (beta != double(1.0)) {
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] *= beta;
						}
					}
				}
				
				//
				//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
				//                                  the above layer dgemm_nn)
				//
				if (alpha == double(1.0)) {
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] += AB[j + i * NR];
						}
					}
				}
				else {
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] += alpha * AB[j + i * NR];
						}
					}
				}
				//free(AB);
				//delete[] AB;
			}

			//A is in column major form and B is in row major form (i.e. A was packed by the function pack_A and B by pack_B)
			template<class T, class F>
			void dgemm_micro_kernel(size_t kc, F alpha, const F* A, const F* B, F beta, T C, size_t stride_row_c, int stride_col_c) {
				/*std::cout<<"zu multiplizierende Matrizen:\n";
									std::cout<<"_A=\n";
									printmat(A,MR,KC);
									std::cout<<"\n _B=\n";
									printmat(B,KC,NR);

									*/
									//std::array<F,MR*NR> AB;	//buffer for result AB
				F* AB=new F[MR * NR];
				//std::array<F,MR*NR>& ABr=&AB;
				std::fill(AB, AB + MR * NR, F(0.0));

				for (size_t k = 0; k < kc; k++) {
					for (size_t j = 0; j < MR; j++) {
						for (size_t i = 0; i < NR; i++) {
							AB[j * NR + i] += A[j] * B[i];

						}
					}
					if (k < KC - 1) {
						A += MR;
						B += NR;
					}
				}

			//
			//  Update C <- beta*C
			//

				if (beta == F(0.0)) {
					//std::fill(C,C+MR*NR,F(0.0)); //geht nicht, da C größer ist als MRxNR
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] = 0.0;
						}
					}
				}
				else if (beta != F(1.0)) {
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] *= beta;
						}
					}
				}

				//
				//  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
				//                                  the above layer dgemm_nn)
				//
				if (alpha == F(1.0)) {
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] += AB[j + i * NR];
						}
					}
				}
				else {
					for (size_t i = 0; i < MR; ++i) {
						for (size_t j = 0; j < NR; ++j) {
							C[i * stride_col_c + j * stride_row_c] += alpha * AB[j + i * NR];
						}
					}
				}
				delete[] AB;
				//std::cout<<"result _C=\n";
				//printmat(C,MR,NR);
				//std::cin.get();

			}

			/*
			Compute Y += alpha*X
			Y and X are m x n matrices
			*/
			template<class T, class F>
			void dgeaxpy(size_t m, size_t n, F alpha, F* X, size_t stride_row_x, size_t stride_col_x, T Y, size_t stride_row_y, size_t stride_col_y) {
				if (alpha != F(1.0)) {
					for (int j = 0; j < m; ++j) {
						for (int i = 0; i < n; ++i) {
							Y[i * stride_row_y + j * stride_col_y] += alpha * X[i * stride_row_x + j * stride_col_x];
						}
					}
				}
				else {
					for (int j = 0; j < m; ++j) {
						for (int i = 0; i < n; ++i) {
							//std::cout<<"in dgeaxpy!\n";
							//printmat(X,m,n);
							//std::cout<<Y[i*stride_row_y+j*stride_col_y]<<" plus "<<X[i*stride_row_x+j*stride_col_x]<<"\n";
							//std::cin.get();
							Y[i * stride_row_y + j * stride_col_y] += X[i * stride_row_x + j * stride_col_x];
						}
					}
				}
			}


			/*
			Compute X*=alpha
			Y and X are m x n matrices
			*/
			template<class T, class F>
			void dgescal(size_t m, size_t n, F alpha, T X, size_t stride_row_x, size_t stride_col_x) {
				//std::cout << "dgescal: m" << m << "n:" << n << "X[0]" << X[0] << "\n";
				if (alpha != F(0.0)) {
					for (int j = 0; j < m; ++j) {
						for (int i = 0; i < n; ++i) {
							X[i * stride_row_x + j * stride_col_x] *= alpha;
							//std::cout << X[i * stride_row_x + j * stride_col_x] << "\n";
						}
					}
				}
				else {
					for (int j = 0; j < m; ++j) {
						for (int i = 0; i < n; ++i) {
							X[i * stride_row_x + j * stride_col_x] = 0;
							//std::cout << X[i * stride_row_x + j * stride_col_x] << "\n";
							//std::cout << "i:" << i << "j" << j << "\n";;
						}
					}
				}
			}
			template<class T, class F>
			void dgemm_macro_kernel(size_t mc, size_t nc, size_t kc, F alpha, F beta, F* _A, F* _B, T C, F* _C, size_t stride_row_c, size_t stride_col_c) {
				size_t q_m = (mc + MR - 1) / MR; //we add MR-1 and then floor the result, so that e.g. a 3 x 3 matrix still has a panel if MR>3
				size_t q_n = (nc + NR - 1) / NR;
				//std::cout << "in macro kernel\n";
				size_t r_m = mc % MR;
				size_t r_n = nc % NR;

				for (size_t j = 0; j < q_n; j++) {
					size_t nr = ((j != q_n - 1) || (r_n == 0)) ? NR : r_n;
					for (size_t i = 0; i < q_m; i++) {
						size_t mr = ((i != q_m - 1) || (r_m == 0)) ? MR : r_m;
					/*
						std::cout<<"zu multiplizierende Matrizen:\n";
						std::cout<<"_A=\n";
						printmat(_A+i*kc*MR,MR,kc);
						std::cout<<"\n _B=\n";
						printmat(_B+j*kc*NR,kc,NR);
						std::cin.get();
						*/
						
						if (mr == MR && nr == NR) {
							//std::cout << "Makro Kernel: mr==MR und nr==NR\n";
							size_t inc_c = i * stride_col_c * MR + j * stride_row_c * NR;
							dgemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, beta, C + inc_c, stride_row_c, stride_col_c);
						}
						else {
							size_t inc_c = i * stride_col_c * MR + j * stride_row_c * NR;
							dgemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, F(0.0), _C, 1, NR);
							dgescal(mr, nr, beta, C + inc_c, stride_row_c, stride_col_c);
							dgeaxpy(mr,nr,F(1.0),_C,1,NR,C+inc_c,stride_row_c,stride_col_c);
						}
					}

				}

			}

			template<class T, class F>
			void dgemm_nn_single(size_t m, size_t n, size_t k, F alpha, T A, size_t stride_row_a, size_t stride_col_a,
				T B, size_t stride_row_b, size_t stride_col_b, F beta, T C, size_t stride_row_c, size_t stride_col_c) {
				
				size_t q_m = (m + MC - 1) / MC; //number of horizontal blocks
				size_t q_n = (n + NC - 1) / NC;
				size_t q_k = (k + KC - 1) / KC;
				size_t r_m = m % MC;
				size_t r_n = n % NC;
				size_t r_k = k % KC;

				//initializing buffers
				
							
				double* _A=new double[KC*MC];
				double* _B =new double[KC*NC];
				double* _C=new double[MC*NC];
				/*
				double _A[KC * MC];
				double _B[KC * NC];
				double _C[MC * NC];
				*/
				/*
				double* _A=(double*)_aligned_malloc(KC*MC*sizeof(double), 32);
				
				double* _B =(double*)_aligned_malloc(KC*NC*sizeof(double), 32);
				double* _C=(double*)_aligned_malloc(MC*NC*sizeof(double), 32);
				*/

				if (alpha == F(0.0) || k == 0) {
					dgescal(m, n, beta, C, stride_row_c, stride_col_c); //there are no matrix A, B to multiply C with
					return;
				}

				for (size_t j = 0; j < q_n; j++) {
					size_t nc = (j != q_n - 1 || r_n == 0) ? NC : r_n;
					for (int l = 0; l < q_k; l++) {
						size_t kc = (l != q_k - 1 || r_k == 0) ? KC : r_k;
						F _beta = (l != 0) ? F(1.0) : beta; //in the first iteration, we need C=1*C+alpha*AB to initalize C
						size_t inc_b = +l * stride_col_b * KC + j * NC * stride_row_b;
						pack_B(kc, nc, B + inc_b, stride_row_b, stride_col_b, _B);
						for (size_t i = 0; i < q_m; i++) {
							size_t mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
							size_t inc_a = i * stride_col_a * MC + l * KC * stride_row_a;
							pack_A(mc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
							size_t inc_c = i * MC * stride_col_c + j * NC * stride_row_c;
							dgemm_macro_kernel(mc, nc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
						}
							
					}
				}
			
				delete[] _A;
				delete[] _B;
				delete[] _C;

			}
			
			//Matrix multiplication with explicitly given buffers and problem dependent blocksizes. 
			template<class T, class F>
			void dgemm_nn_explicit(size_t m, size_t n, size_t k, F alpha, T A, size_t stride_row_a, size_t stride_col_a,
				T B, size_t stride_row_b, size_t stride_col_b, F beta, T C, size_t stride_row_c, size_t stride_col_c,size_t q_m, size_t q_n, size_t q_k, size_t r_m, size_t r_n, size_t r_k) {

				double* _A=new double[KC * MC];
				double* _B= new double[KC * NC];
				double* _C=new double[MC * NC];
				for (size_t j = 0; j < q_n; j++) {
					size_t nc = (j != q_n - 1 || r_n == 0) ? NC : r_n;
					for (int l = 0; l < q_k; l++) {
						size_t kc = (l != q_k - 1 || r_k == 0) ? KC : r_k;
						F _beta = (l != 0) ? F(1.0) : beta; //in the first iteration, we need C=1*C+alpha*AB to initalize C
						size_t inc_b = +l * stride_col_b * KC + j * NC * stride_row_b;
						pack_B(kc, nc, B + inc_b, stride_row_b, stride_col_b, _B);
						for (size_t i = 0; i < q_m; i++) {
							size_t mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
							size_t inc_a = i * stride_col_a * MC + l * KC * stride_row_a;
							pack_A(mc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
							size_t inc_c = i * MC * stride_col_c + j * NC * stride_row_c;
							dgemm_macro_kernel(mc, nc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
						}
							
					}
				}
				delete[] _A;
				delete[] _B;
				delete[] _C;			
			}

			export template<class T, class F>
			void dgemm_nn(size_t m, size_t n, size_t k, F alpha, T A, size_t stride_row_a, size_t stride_col_a,
				T B, size_t stride_row_b, size_t stride_col_b, F beta, T C, size_t stride_row_c, size_t stride_col_c) {
				size_t q_m = (m + MC - 1) / MC; //number of vertical blocks of A
				size_t q_n = (n + NC - 1) / NC;
				size_t q_k = (k + KC - 1) / KC;

				size_t r_m = m % MC;
				size_t r_n = n % NC;
				size_t r_k = k % KC;

				//initializing buffers
				//double* _A[KC * MC];
				//double* _B[KC * NC];
				//double* _C[MC * NC];
		
				
				//double* _A=(double*)_aligned_malloc(KC*MC*sizeof(double), 32);
				
				//double* _B =(double*)_aligned_malloc(KC*NC*sizeof(double), 32);
				//double* _C=(double*)_aligned_malloc(MC*NC*sizeof(double), 32);
				if (alpha == F(0.0) || k == 0) {
					dgescal(m, n, beta, C, stride_row_c, stride_col_c); //there are no matrix A, B to multiply C with
					return;
				}
				else if(m<1 && n<1){
				
				double* _A=new double[KC * MC];
				double* _B= new double[KC * NC];
				double* _C=new double[MC * NC];
				
					for (size_t j = 0; j < q_n; j++) {
						size_t nc = (j != q_n - 1 || r_n == 0) ? NC : r_n;
						for (int l = 0; l < q_k; l++) {
							size_t kc = (l != q_k - 1 || r_k == 0) ? KC : r_k;
							F _beta = (l != 0) ? F(1.0) : beta; //in the first iteration, we need C=1*C+alpha*AB to initalize C
							size_t inc_b = +l * stride_col_b * KC + j * NC * stride_row_b;
							pack_B(kc, nc, B + inc_b, stride_row_b, stride_col_b, _B);
							for (size_t i = 0; i < q_m; i++) {
								size_t mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
								size_t inc_a = i * stride_col_a * MC + l * KC * stride_row_a;
								pack_A(mc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
								size_t inc_c = i * MC * stride_col_c + j * NC * stride_row_c;
								dgemm_macro_kernel(mc, nc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
								//std::cout << "\n C=\n";
								//printmat(C, m, n);
							}

							
						}

					}
					
				delete[] _A;
				delete[] _B;
				delete[] _C;
				}
				else{
		
					int n_threads=std::thread::hardware_concurrency();
				//	std::cout<<"Bin in Fall 1 der Multicoreversion\n";
					/*
					//Although sound in theory, memory is too slow to justify multicore environment here
					if (alpha == F(0.0) || k == 0) {
						
							int chunkM=m;
							int chunkN=n;
							int chunk_offset=0;
							int rem=0;
							int remAdd=0;
							int remAddN=0;
							int remAddM=0;
							if (m>=n){
								if (n_threads>m){
									n_threads=m;
								}
								chunkM=m/n_threads,
								rem=m%n_threads;
								chunk_offset=n*chunkM;	
								remAdd=n;
								remAddM=1;
							}
							else{
								if (n_threads>n){
									n_threads=n;
								}
								chunkN=n/n_threads;
								rem=n%n_threads;
								chunk_offset=chunkN;
								remAdd=1;
								remAddN=1;
							}
							T Cs=&C[0];
							std::vector<std::thread> ts(n_threads);
						//	std::cout<<"n threads: "<<n_threads<<"\n";
						std::cout<<chunkM<<" "<<chunkN<<"  "<<chunk_offset<<"  "<<remAddM<<"\n";
							for (int i=0;i<n_threads;i++){
								if (rem==0){
								//	std::string s=std::to_string(chunk_offset)+"C:"+std::to_string(Cs[0])+"\n";
								//	std::cout<<s;
									ts[i]=std::thread(dgescal<T,F>,chunkM, chunkN, beta, Cs, stride_row_c, stride_col_c); //there are no matrix A, B to multiply C with

									Cs+=chunk_offset;
									
									
								}
								else{
								//	std::string s=std::to_string(chunk_offset+remAdd)+"C:"+std::to_string(Cs[0])+"\n";
								//	std::cout<<s;
									ts[i]=std::thread(dgescal<T,F>,chunkM+remAddM, chunkN+remAddN, beta, Cs, stride_row_c, stride_col_c); //there are no matrix A, B to multiply C with
									Cs+=chunk_offset+remAdd;								
									rem--;
									
								}
							}
							for (int i=0;i<n_threads;i++){
								ts[i].join();
								
							}
							return;
						}
					*/
					std::vector<std::thread> ts(n_threads);	
					int rem=0;
					T Cs=C;	
					
					if (m>=n){
						int chunkM;
						int chunkQM=q_m;
							
						T As=A;							
									
						if (q_m>n_threads){
							chunkQM=q_m/n_threads; //number of vertical blockpieces in A per thread
							rem=q_m%n_threads;
							chunkM=chunkQM*MC;
						}
						else{
							chunkQM=1;
							chunkM=MC;
							n_threads=q_m;
						}
						int leftover=(rem==0)?m-(n_threads-1)*chunkQM*MC:m-(n_threads-1-rem)*chunkQM*MC-rem*(chunkQM+1)*MC;
						
						for (int i=0;i<(n_threads-1);i++){
							if (rem==0){

								ts[i]=std::thread(dgemm_nn_explicit<T,F>,chunkM,n,k, alpha, As,stride_row_a,stride_col_a,B,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,chunkQM,q_n,q_k,0,r_n, r_k); //there are no matrix A, B to multiply C with
								As+=chunkQM*MC*k;
								Cs+=chunkQM*MC*n;
							}
							else{
								ts[i]=std::thread(dgemm_nn_explicit<T,F>,chunkM+MC,n,k, alpha, As,stride_row_a,stride_col_a,B,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,chunkQM+1,q_n,q_k,0,r_n, r_k); //there are no matrix A, B to multiply C with
								As+=(chunkQM+1)*MC*k;
								Cs+=(chunkQM+1)*MC*n;					
								rem--;
								
							}
						}
			
						ts[n_threads-1]=std::thread(dgemm_nn_explicit<T,F>,leftover,n,k, alpha, As,stride_row_a,stride_col_a,B,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,chunkQM,q_n,q_k,leftover%MC,r_n, r_k);
						
						for (int i=0;i<n_threads;i++){
							ts[i].join();
							
						}				
					}
					else{
						int chunkN;
						int chunkQN=q_n;
						T Bs=B;					
						if (q_n>n_threads){
							chunkQN=q_n/n_threads; //number of horizontal blockpieces in B per thread
							rem=q_n%n_threads;
							chunkN=chunkQN*NC;
						}
						else{
							chunkQN=1;
							chunkN=NC;
							n_threads=q_n;
						}
						int leftover=(rem==0)?n-(n_threads-1)*chunkQN*NC:n-(n_threads-1-rem)*chunkQN*NC-rem*(chunkQN+1)*NC;
						
						for (int i=0;i<(n_threads-1);i++){
							if (rem==0){
								ts[i]=std::thread(dgemm_nn_explicit<T,F>,m,chunkN,k, alpha, A,stride_row_a,stride_col_a,Bs,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,q_m,chunkQN,q_k,r_m,0, r_k); //there are no matrix A, B to multiply C with
								Bs+=chunkQN*NC;
								Cs+=chunkQN*NC;
							}
							else{
								ts[i]=std::thread(dgemm_nn_explicit<T,F>,m,chunkN+NC,k, alpha, A,stride_row_a,stride_col_a,Bs,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,q_m,chunkQN+1,q_k,r_m,0, r_k); //there are no matrix A, B to multiply C with
								Bs+=(chunkQN+1)*NC;
								Cs+=(chunkQN+1)*NC;					
								rem--;
							}
						}
						ts[n_threads-1]=std::thread(dgemm_nn_explicit<T,F>,m,leftover,k, alpha, A,stride_row_a,stride_col_a,Bs,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,q_m,chunkQN,q_k,r_m,leftover%NC, r_k);
						
						for (int i=0;i<n_threads;i++){
							ts[i].join();
							
						}	
						
					}
				}
			}
		
		}
	}
}