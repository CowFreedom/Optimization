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
					B += NR*stride_rows;
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
			void gemm_micro_kernel(size_t kc, double alpha, const double* A, const double* B, double beta, T C, size_t stride_row_c, int stride_col_c) {
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
				//                                  the above layer gemm)
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
			void gemm_micro_kernel(size_t kc, F alpha, const F* A, const F* B, F beta, T C, size_t stride_row_c, int stride_col_c) {
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
				//                                  the above layer gemm)
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
			void gemm_macro_kernel(size_t mc, size_t nc, size_t kc, F alpha, F beta, F* _A, F* _B, T C, F* _C, size_t stride_row_c, size_t stride_col_c) {
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
							gemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, beta, C + inc_c, stride_row_c, stride_col_c);
						}
						else {
							size_t inc_c = i * stride_col_c * MR + j * stride_row_c * NR;
							gemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, F(0.0), _C, 1, NR);
							dgescal(mr, nr, beta, C + inc_c, stride_row_c, stride_col_c);
							dgeaxpy(mr,nr,F(1.0),_C,1,NR,C+inc_c,stride_row_c,stride_col_c);
						}
					}

				}

			}

			template<class TA,class TB, class TC, class F>
			void gemm_single(size_t m, size_t n, size_t k, F alpha, TA A, size_t stride_row_a, size_t stride_col_a,
				TB B, size_t stride_row_b, size_t stride_col_b, F beta, TC C, size_t stride_row_c, size_t stride_col_c) {
				
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
							gemm_macro_kernel(mc, nc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
						}
							
					}
				}
			
				delete[] _A;
				delete[] _B;
				delete[] _C;

			}
			
			//Matrix multiplication with explicitly given buffers and problem dependent blocksizes. 
			template<class TA, class TB, class TC, class F>
			void gemm_explicit(size_t m, size_t n, size_t k, F alpha, TA A, size_t stride_row_a, size_t stride_col_a,
				TB B, size_t stride_row_b, size_t stride_col_b, F beta, TC C, size_t stride_row_c, size_t stride_col_c,size_t q_m, size_t q_n, size_t q_k, size_t r_m, size_t r_n, size_t r_k) {

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
							gemm_macro_kernel(mc, nc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
						}
							
					}
				}
				delete[] _A;
				delete[] _B;
				delete[] _C;			
			}

			export template<class TA, class TB, class TC, class F>
			void gemm(size_t m, size_t n, size_t k, F alpha, TA A, size_t stride_row_a, size_t stride_col_a,
				TB B, size_t stride_row_b, size_t stride_col_b, F beta, TC C, size_t stride_row_c, size_t stride_col_c) {
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
				else if(m<500 && n<500){
				
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
								gemm_macro_kernel(mc, nc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
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
					TC Cs=C;	
					
					if (m>=n){
						int chunkM;
						int chunkQM=q_m;
							
						TA As=A;							
									
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

								ts[i]=std::thread(gemm_explicit<TA,TB,TC,F>,chunkM,n,k, alpha, As,stride_row_a,stride_col_a,B,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,chunkQM,q_n,q_k,0,r_n, r_k); //there are no matrix A, B to multiply C with
								As+=chunkQM*MC*stride_col_a;
								Cs+=chunkQM*MC*stride_col_c;
							}
							else{
								ts[i]=std::thread(gemm_explicit<TA,TB,TC,F>,chunkM+MC,n,k, alpha, As,stride_row_a,stride_col_a,B,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,chunkQM+1,q_n,q_k,0,r_n, r_k); //there are no matrix A, B to multiply C with
								As+=(chunkQM+1)*MC*stride_col_a;
								Cs+=(chunkQM+1)*MC*stride_col_c;					
								rem--;
								
							}
						}
			
						ts[n_threads-1]=std::thread(gemm_explicit<TA,TB,TC,F>,leftover,n,k, alpha, As,stride_row_a,stride_col_a,B,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,chunkQM,q_n,q_k,leftover%MC,r_n, r_k);
						
						for (int i=0;i<n_threads;i++){
							ts[i].join();
							
						}				
					}
					else{
						int chunkN;
						int chunkQN=q_n;
						TB Bs=B;					
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
								ts[i]=std::thread(gemm_explicit<TA,TB,TC,F>,m,chunkN,k, alpha, A,stride_row_a,stride_col_a,Bs,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,q_m,chunkQN,q_k,r_m,0, r_k); //there are no matrix A, B to multiply C with
								Bs+=chunkQN*NC*stride_row_b;
								Cs+=chunkQN*NC*stride_row_c;
							}
							else{
								ts[i]=std::thread(gemm_explicit<TA,TB,TC,F>,m,chunkN+NC,k, alpha, A,stride_row_a,stride_col_a,Bs,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,q_m,chunkQN+1,q_k,r_m,0, r_k); //there are no matrix A, B to multiply C with
								Bs+=(chunkQN+1)*NC*stride_row_b;
								Cs+=(chunkQN+1)*NC*stride_row_c;					
								rem--;
							}
						}
						ts[n_threads-1]=std::thread(gemm_explicit<TA,TB,TC,F>,m,leftover,k, alpha, A,stride_row_a,stride_col_a,Bs,stride_row_b,stride_col_b,beta,Cs,stride_row_c,stride_col_c,q_m,chunkQN,q_k,r_m,leftover%NC, r_k);
						
						for (int i=0;i<n_threads;i++){
							ts[i].join();
							
						}	
						
					}
				}
			}
			
						/*DGMV algorithm
			*/
			/*
			Compute y += alpha*x
			y and x are vectors
			See https://www5.in.tum.de/lehre/vorlesungen/parnum/WS10/PARNUM_4.pdf for an explanation
			*/
			template<class T, class F>
			void saxpy(size_t m,F alpha, T x,size_t stride_x, T y, size_t stride_y) {
				for (int j = 0; j < m; ++j) {
							y[j * stride_y] += alpha*x[j * stride_x];
					//		std::cout<<alpha<<" times "<<x[j*stride_x]<<"\n";
				}
			}

			/*
			Compute y += alpha_i*x
			y is a vector and X a matrix
			*/
			template<class T, class F>
			void gaxpy(size_t m, size_t n, F alpha, T X, size_t stride_row_x, size_t stride_col_x,T y, size_t stride_y,  T c, size_t stride_c) {
				//std::cout<<"in gaxpy\n";
				for (int i=0;i<n;i++){
					saxpy(m, alpha*y[i*stride_y], X,stride_col_x,c,stride_c);
					X+=stride_row_x;
				}
						
			}

				
			template<class T, class F>
			void dgmv_explicit(size_t m, size_t n,F alpha, T A, size_t stride_row_a, size_t stride_col_a,
				T y, size_t stride_y, F beta, T c, size_t stride_c){
					//gaxpy(m,n,alpha,A,stride_row_a,stride_col_a,y,stride_y,c,stride_c);
					
					for (int i=0;i<m;i++){
						c[i*stride_c]*=beta;
						for (int j=0;j<n;j++){
							c[i*stride_c]+=alpha*A[i*stride_col_a+j*stride_row_a]*y[j*stride_y];
						}
					}			
			}
		
		
			
			export template<class T, class F>
			void dgmv(size_t m, size_t n,F alpha, T A, size_t stride_row_a, size_t stride_col_a,
				T y, size_t stride_y, F beta, T c, size_t stride_c){
					if (alpha == F(0.0)) {
					return;
					}
					
					if (m<1500 && n<15000){
						for (int i=0;i<m;i++){
							c[i*stride_c]*=beta;
							for (int j=0;j<n;j++){
								c[i*stride_c]+=alpha*A[i*stride_col_a+j*stride_row_a]*y[j*stride_y];
							}
						}						
					}
					else{
						int n_threads=std::thread::hardware_concurrency();
						int rem=0;
						T cs=c;	

						int chunkM;
								
						T As=A;							
									
						if (m>n_threads){
							chunkM=m/n_threads; //number of rows per thread
							rem=m%n_threads;
						}
						else{
							n_threads=m;
							chunkM=1;
						}
						std::vector<std::thread> ts(n_threads);	
						int leftover=(rem==0)?m-(n_threads-1)*chunkM:m-(n_threads-1-rem)*chunkM-rem*(chunkM+1);
						for (int i=0;i<(n_threads-1);i++){
							if (rem==0){
								ts[i]=std::thread(dgmv_explicit<T,F>,chunkM,n,alpha,As,stride_row_a,stride_col_a,y,stride_y,beta,cs,stride_c); //there are no matrix A, B to multiply C with
								As+=stride_col_a*chunkM;
								cs+=chunkM;
							}
							else{
								ts[i]=std::thread(dgmv_explicit<T,F>,chunkM+1,n,alpha,As,stride_row_a,stride_col_a,y,stride_y,beta,cs,stride_c); //there are no matrix A, B to multiply C with
								As+=stride_col_a*(chunkM+1);
								cs+=chunkM+1;				
								rem--;
							}
						}
						
						ts[n_threads-1]=std::thread(dgmv_explicit<T,F>,leftover,n,alpha,As,stride_row_a,stride_col_a,y,stride_y,beta,cs,stride_c);
											

						for (auto& x: ts){
							x.join();
						}
					}
					
			}

			
			/*microkernel for packed triangular matrix (pt) C and nonpacked buffer matrices A,B that represent lower triangular L1 and upper triangular L2.
			Because the result of L1L2 can be dense, it is assumed that C is a panel inside a bigger matrix multiplication, therefore stride_col_c cannot be zero*/
			template<class T, class F>
			void dgeaxpy_pt(size_t m, size_t n, F alpha, F* X, size_t stride_row_x, size_t stride_col_x, T Y, size_t stride_row_y, size_t stride_col_y) {
				if (alpha != F(1.0)) {
					for (int j = 0; j < m; ++j) {
						for (int i = 0; i < n; ++i) {
							Y[i * stride_row_y + j * stride_col_y] += alpha * X[i * stride_row_x + j * stride_col_x];
						}
					}
				}
				else {
					for (int j = 0; j < m; ++j) {
						int offset=0;
						for (int i = 0; i < n; ++i) {
							if (i>=2){
								offset++; //the result C is in the middle of the packed Matrix AB. Therefore we have to add an increasing offset to account for triangular packed form
							}
							size_t ix=j*0.5*(j+1)+j*stride_col_y+offset;
							Y[i * stride_row_y + ix] += X[i * stride_row_x + j * stride_col_x];
						}
					}
				}
			}
			
			/*Copies from array source to dest*/
			template<class TS, class TD>
			void dcopy(int m, int n, TS source, size_t stride_row_source, size_t stride_col_source, TD dest, size_t stride_row_dest,size_t stride_col_dest){
				for (int i=0;i<m;i++){
					for (int j=0;j<n;j++){
						dest[i*stride_col_dest+j*stride_row_dest]=source[i*stride_col_source+j*stride_row_source];
					}
				}
			}



			//rausfinden bedeutung  checken auf korrektheit
			template<class T, class F>
			void truscal(int m, int n, F alpha, T A, int stride_row_a, int stride_col_a) {
				int k = std::min(m, n);
				for (int i = 0; i < k; i++) {
					for (int j = i; j < n; j++) {
						A[i * stride_col_a + j * stride_row_a] *= alpha;
					}
				}

			}

			//Y=alpha*X for upper trianguar X
			template<class T, class F>
			void truaxpy(int m, int n, F alpha, T X, int  stride_row_x, int stride_col_x, T Y, int  stride_row_y, int stride_col_y) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j < std::min(i + 1, m); j++) {
						Y[j * stride_col_y + i * stride_row_y] += alpha * X[j * stride_col_x + i * stride_row_x];

					}
				}
			}

			//A is in column major form and B is in row major form (i.e. A was packed by the function pack_A and B by pack_B)
			template<class T, class F>
			void syurk_micro_kernel(int mr, int nr, int kc, int ic, int jc, F alpha, const F* _A, const F* _B, F beta, T C, int stride_row_c, int stride_col_c) {

				F AB[MR * NR];
				gemm_micro_kernel(kc, alpha, _A, _B, F(0.0), AB, 1, NR);
				//	std::cout<<"AB draußen:\n";
				//	printmat(AB,MR,NR);
					//NR>MR
				if (jc > ic) {
					dgescal(jc - ic, nr, beta, C, stride_row_c, stride_col_c);
					dgeaxpy(jc - ic, nr, F(1.0), AB, 1, NR, C, stride_row_c, stride_col_c);
					truscal(mr - (jc - ic), nr, beta,
						C + (jc - ic) * stride_col_c, stride_row_c, stride_col_c);
					truaxpy(mr - (jc - ic), nr, 1.0,
						AB + (jc - ic) * NR, 1, NR,
						C + (jc - ic) * stride_col_c, stride_row_c, stride_col_c);
					/*
					dgescal(jc-ic, nr, beta, C, stride_row_c, stride_col_c);
					dgeaxpy(jc-ic, nr, F(1.0), AB, 1, NR, C, stride_row_c, stride_col_c);
					truscal(mr-(jc-ic), nr, beta,
							C+(jc-ic)*stride_row_c, stride_row_c, stride_col_c);
					truaxpy(mr-(jc-ic), nr, 1.0,
							AB+(jc-ic), 1, NR,
							C+(jc-ic)*stride_row_c,stride_row_c, stride_col_c);
							*/
							//std::cout<<"jc-ic"<<jc-ic<<"\n";
						//	std::cin.get();
									//MR>NR		
				}
				else {
					truscal(mr, nr - (ic - jc), beta,
						C + (ic - jc) * stride_row_c, stride_row_c, stride_col_c);
					truaxpy(mr, nr - (ic - jc), 1.0,
						AB + (ic - jc), 1, NR,
						C + (ic - jc) * stride_row_c, stride_row_c, stride_col_c);
					/*
					truscal(mr, nr-(ic-jc), beta,
							C, stride_row_c, stride_col_c);
					truaxpy(mr, nr-(ic-jc), 1.0,
							AB, 1, NR,
							C,stride_row_c, stride_col_c);
	*/

	/*
	//original
	truscal(mr, nr-(ic-jc), beta,
			C+(ic-jc)*stride_row_c, stride_row_c, stride_col_c);
	truaxpy(mr, nr-(ic-jc), 1.0,
			AB+(ic-jc), 1, NR,
			C+(ic-jc)*stride_row_c,stride_row_c, stride_col_c);
			*/
				}


				//   std::cin.get();

			}

			template<class T, class F>
			void syurk_macro_kernel(size_t mc, size_t nc, size_t kc, F alpha, F beta, F* _A, F* _B, T C, F* _C, size_t stride_row_c, size_t stride_col_c) {
				size_t q_m = (mc + MR - 1) / MR; //we add MR-1 and then floor the result, so that e.g. a 3 x 3 matrix still has a panel if MR>3
				size_t q_n = (nc + NR - 1) / NR;
				//std::cout << "in macro kernel\n";
				size_t r_m = mc % MR;
				size_t r_n = nc % NR;
				//If MR!=NR, we may have to adjust for the different sizes accordingly
				int ki = (MR < NR) ? NR / MR : 1;  // 2
				int kj = (MR > NR) ? MR / NR : 1;  // 1

				for (size_t j = 0; j < q_n; j++) {
					size_t nr = ((j != q_n - 1) || (r_n == 0)) ? NR : r_n;
					for (size_t i = 0; (i / ki) <= (j / kj); i++) {
						size_t mr = ((i != q_m - 1) || (r_m == 0)) ? MR : r_m;
						/*
						std::cout<<"zu multiplizierende Matrizen:\n";
						std::cout<<"_A=\n";
						printmat(_A+i*kc*MR,MR,kc);
						std::cout<<"\n _B=\n";
						printmat(_B+j*kc*NR,kc,NR);
						std::cin.get();
						*/
						size_t inc_c = i * stride_col_c * MR + j * stride_row_c * NR;
						if ((i / ki) == (j / kj)) {
							syurk_micro_kernel(mr, nr, kc, i * MR, j * NR, alpha, (_A + i * kc * MR), _B + j * kc * NR, beta, C + inc_c, stride_row_c, stride_col_c);
						}
						else {
							if (mr == MR && nr == NR) {
								gemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, beta, C + inc_c, stride_row_c, stride_col_c);
							}
							else {
								gemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, F(0.0), _C, 1, NR);
								dgescal(mr, nr, beta, C + inc_c, stride_row_c, stride_col_c);
								dgeaxpy(mr, nr, F(1.0), _C, 1, NR, C + inc_c, stride_row_c, stride_col_c);
							}
						}

					}

				}
			}

			//https://github.com/michael-lehn/ulmBLAS/blob/master/ulmblas/level3/syurk.tcc
			template<class TA,class TC, class F>
			void syurk_explicit(F alpha, TA A, int stride_row_a, int stride_col_a, F beta, TC C, int stride_row_c, int stride_col_c, int m_max, int r_m_max, int q_m, int q_k, int r_m, int r_k, int offset) {
				//	std::cout<<"explicit:"<<r_m_max<<"\n";
				F* _A = new F[MC * MC]();
				F* _B = new F[MC * MC]();
				F* _C = new double[MC * MC]();
				for (int j = 0; j < m_max; j++) {

					int nc = (j != m_max - 1 || r_m_max == 0) ? MC : r_m_max;
					for (int l = 0; l < q_k; l++) {
						int kc = (l != q_k - 1 || r_k == 0) ? MC : r_k;
						int inc_a = j * stride_col_a * MC + l * MC * stride_row_a;
						/*std::cout<<"rm max"<<r_m_max<<"\n";
						std::cout<<"m_max:"<<m_max<<"\n";
						std::cout<<"qk:"<<q_k<<"l:"<<l<<"\n";
						std::cout<<"nca:"<<nc<<"kc"<<kc<<"\n";

						std::cout<<A[inc_a]<<"\n";
						*/
						pack_A(nc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
						/*
						std::cout<<"_A\n";
						printmat(_A,MC,MC);
						std::cout<<"\n";
						*/
						F _beta = (l == 0) ? beta : F(1.0);
						for (int i = j; i < q_m - offset; i++) {
							int mc = (i != (q_m - (offset)) - 1 || r_m == 0) ? MC : r_m;
							int inc_b = i * stride_col_a * MC + l * MC * stride_row_a;
							pack_B(kc, mc, A + inc_b, stride_col_a, stride_row_a, _B);
							/*std::cout<<"kcb:"<<kc<<"mc:"<<mc<<"q_m-m_max*index"<<offset<<"\n";
							std::cout<<"B[inc]"<<A[inc_b]<<"\n";
							printmat(_B,MC,MC);
							std::cout<<"\n";
							*/
							int inc_c = j * MC * stride_col_c + i * MC * stride_row_c;

							if (i != j) {
								gemm_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
								//std::cout<<"damit fertig\n";
							}
							else {
								syurk_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
							}
						}
					}
					//std::cout<<"C:\n";
	//printmat(C,17,17);


	//std::cin.get();
				}
				delete[] _A;
				delete[] _B;
				delete[] _C;

			}

			//https://github.com/michael-lehn/ulmBLAS/blob/master/ulmblas/level3/syurk.tcc
			template<class TA, class TC, class F>
			void syurk_single(int n, int k, F alpha, TA A, int stride_row_a, int stride_col_a, F beta, TC C, int stride_row_c, int stride_col_c) {
				if (alpha == F(0.0) || k == 0) {
					truscal(n, n, beta, C, stride_row_c, stride_col_c); //there are no matrix A, B to add to C with
					return;
				}

				int q_m = (n + MC - 1) / MC; //number of vertical blocks of A
				int q_k = (k + MC - 1) / MC;

				int r_m = n % MC;
				int r_k = k % MC;

				F* _A = new F[MC * MC]();
				F* _B = new F[MC * MC]();
				F* _C = new double[MC * MC]();
				for (int j = 0; j < q_m; j++) {
					int nc = (j != q_m - 1 || r_m == 0) ? MC : r_m;

					for (int l = 0; l < q_k; l++) {
						int kc = (l != q_k - 1 || r_k == 0) ? MC : r_k;
						int inc_a = j * stride_col_a * MC + l * MC * stride_row_a;
						pack_A(nc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
						F _beta = (l == 0) ? beta : F(1.0);
						for (int i = j; i < q_m; i++) {
							int mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
							int inc_b = i * stride_col_a * MC + l * MC * stride_row_a;
							pack_B(kc, mc, A + inc_b, stride_col_a, stride_row_a, _B);			
							int inc_c = j * MC * stride_col_c + i * MC * stride_row_c;
							if (i != j) {
								gemm_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
							}
							else {
								syurk_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
							}
						}

					}
				}
				delete[] _A;
				delete[] _B;
				delete[] _C;
			}

			export template<class TA, class TC, class F>
			void syurk(int n, int k, F alpha, TA A, int stride_row_a, int stride_col_a, F beta, TC C, int stride_row_c, int stride_col_c) {
				if (alpha == F(0.0) || k == 0) {
					truscal(n, n, beta, C, stride_row_c, stride_col_c); //there are no matrix A, B to add to C with
					return;
				}

				int q_m = (n + MC - 1) / MC; //number of vertical blocks of A
				int q_k = (k + MC - 1) / MC;

				int r_m = n % MC;
				int r_k = k % MC;


				if (n < 200 && k < 200) {
					F* _A = new F[MC * MC]();
					F* _B = new F[MC * MC]();
					F* _C = new double[MC * MC]();

					for (int j = 0; j < q_m; j++) {
						int nc = (j != q_m - 1 || r_m == 0) ? MC : r_m;

						for (int l = 0; l < q_k; l++) {
							int kc = (l != q_k - 1 || r_k == 0) ? MC : r_k;
							int inc_a = j * stride_col_a * MC + l * MC * stride_row_a;
							pack_A(nc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
							F _beta = (l == 0) ? beta : F(1.0);
							for (int i = j; i < q_m; i++) {
								int mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
								int inc_b = i * stride_col_a * MC + l * MC * stride_row_a;
								pack_B(kc, mc, A + inc_b, stride_col_a, stride_row_a, _B);
								int inc_c = j * MC * stride_col_c + i * MC * stride_row_c;
								if (i != j) {

									gemm_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
								}
								else {
									syurk_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);

								}
							}

						}
					}

					delete[] _A;
					delete[] _B;
					delete[] _C;
				}
				int n_threads = std::thread::hardware_concurrency();
				int rem = 0;
				TC Cs = C;
				TA As = A;

				//chunkM is in reality chunkN
				if (true) {
					int chunkM;
					int chunkQM = q_m;



					if (q_m > n_threads) {
						chunkQM = q_m / n_threads; //number of vertical blockpieces in A per thread
						rem = q_m % n_threads;
						chunkM = chunkQM * MC;
					}
					else {
						chunkQM = 1;
						chunkM = MC;
						n_threads = q_m;

					}

					std::vector<std::thread> ts(n_threads);
					int leftover = (rem == 0) ? n - (n_threads - 1) * chunkQM * MC : n - (n_threads - 1 - rem) * chunkQM * MC - rem * (chunkQM + 1) * MC; //number of leftover columns
					int panels_so_far = 0;

					for (int i = 0; i < n_threads - 1; i++) {
						if (rem == 0) {
							ts[i] = std::thread(syurk_explicit<TA,TC, F>, alpha, As, stride_row_a, stride_col_a, beta, Cs, stride_row_c, stride_col_c, chunkQM, 0, q_m, q_k, r_m, r_k, panels_so_far); //there are no matrix A, B to multiply C with
							As += stride_col_a * chunkM;
							Cs += stride_row_c * chunkM + stride_col_c * chunkM;
							panels_so_far += chunkQM;
						}
						else {
							ts[i] = std::thread(syurk_explicit<TA,TC, F>, alpha, As, stride_row_a, stride_col_a, beta, Cs, stride_row_c, stride_col_c, chunkQM + 1, 0, q_m, q_k, r_m, r_k, panels_so_far); //there are no matrix A, B to multiply C with
							As += stride_col_a * (chunkM + MC);
							Cs += stride_row_c * (chunkM + MC) + stride_col_c * (chunkM + MC);
							rem--;
							panels_so_far += chunkQM + 1;
						}

					}
					ts[n_threads - 1] = std::thread(syurk_explicit<TA,TC, F>, alpha, As, stride_row_a, stride_col_a, beta, Cs, stride_row_c, stride_col_c, chunkQM, r_m, q_m, q_k, r_m, r_k, panels_so_far);

					for (int i = 0; i < n_threads; i++) {
						ts[i].join();

					}

				}
				else {


				}


			}

			/*Calculation of Sylrk*/

			//Y=alpha*Y for lower triangular Y
			template<class T, class F>
			void trlscal(int m, int n, F alpha, T A, int stride_row_a, int stride_col_a) {
				int k = std::min(m, n);
				for (int i = 0; i < k; i++) {
					for (int j = 0; j <= i; j++) {
						A[i * stride_col_a + j * stride_row_a] *= alpha;
					}
				}

			}

			//Y=alpha*X for lower trianguar X
			template<class T, class F>
			void trlaxpy(int m, int n, F alpha, T X, int  stride_row_x, int stride_col_x, T Y, int  stride_row_y, int stride_col_y) {
				for (int i = 0; i < n; i++) {
					for (int j = i; j < m; j++) {
						Y[j * stride_col_y + i * stride_row_y] += alpha * X[j * stride_col_x + i * stride_row_x];

					}
				}
			}

			//A is in column major form and B is in row major form (i.e. A was packed by the function pack_A and B by pack_B)
			template<class T, class F>
			void sylrk_micro_kernel(int mr, int nr, int kc, int ic, int jc, F alpha, const F* _A, const F* _B, F beta, T C, int stride_row_c, int stride_col_c) {

				F AB[MR * NR];
				gemm_micro_kernel(kc, alpha, _A, _B, F(0.0), AB, 1, NR);

				if (jc < ic) {
					dgescal(mr, ic - jc, beta, C, stride_row_c, stride_col_c);
					dgeaxpy(mr, ic - jc, F(1.0), AB, 1, NR, C, stride_row_c, stride_col_c);
					trlscal(mr, nr - (ic - jc), beta,
						C + (ic - jc) * stride_row_c, stride_row_c, stride_col_c);
					trlaxpy(mr, nr - (ic - jc), F(1.0),
						AB + (ic - jc), 1, NR,
						C + (ic - jc) * stride_row_c, stride_row_c, stride_col_c);

				}
				else {
					trlscal(mr - (jc - ic), nr, beta,
						C + (jc - ic) * stride_col_c, stride_row_c, stride_col_c);
					trlaxpy(mr - (jc - ic), nr, F(1.0),
						AB + (jc - ic) * NR, 1, NR,
						C + (jc - ic) * stride_col_c, stride_row_c, stride_col_c);
				}

			}

			template<class T, class F>
			void sylrk_macro_kernel(size_t mc, size_t nc, size_t kc, F alpha, F beta, F* _A, F* _B, T C, F* _C, size_t stride_row_c, size_t stride_col_c) {
				size_t q_m = (mc + MR - 1) / MR; //we add MR-1 and then floor the result, so that e.g. a 3 x 3 matrix still has a panel if MR>3
				size_t q_n = (nc + NR - 1) / NR;
				//std::cout << "in macro kernel\n";
				size_t r_m = mc % MR;
				size_t r_n = nc % NR;
				//If MR!=NR, we may have to adjust for the different sizes accordingly
				int ki = (MR < NR) ? NR / MR : 1;  // 2
				int kj = (MR > NR) ? MR / NR : 1;  // 1

				for (size_t j = 0; j < q_n; j++) {
					size_t nr = ((j != q_n - 1) || (r_n == 0)) ? NR : r_n;
					for (size_t i = ki * (j / kj); i < q_m; i++) {
						size_t mr = ((i != q_m - 1) || (r_m == 0)) ? MR : r_m;
						/*
						std::cout<<"zu multiplizierende Matrizen:\n";
						std::cout<<"_A=\n";
						printmat(_A+i*kc*MR,MR,kc);
						std::cout<<"\n _B=\n";
						printmat(_B+j*kc*NR,kc,NR);
						std::cin.get();
						*/
						size_t inc_c = i * stride_col_c * MR + j * stride_row_c * NR;
						if ((i / ki) == (j / kj)) {
							sylrk_micro_kernel(mr, nr, kc, i * MR, j * NR, alpha, (_A + i * kc * MR), _B + j * kc * NR, beta, C + inc_c, stride_row_c, stride_col_c);
						}
						else {
							if (mr == MR && nr == NR) {
								gemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, beta, C + inc_c, stride_row_c, stride_col_c);
							}
							else {
								gemm_micro_kernel(kc, alpha, (_A + i * kc * MR), _B + j * kc * NR, F(0.0), _C, 1, NR);
								dgescal(mr, nr, beta, C + inc_c, stride_row_c, stride_col_c);
								dgeaxpy(mr, nr, F(1.0), _C, 1, NR, C + inc_c, stride_row_c, stride_col_c);
							}
						}

					}

				}
			}
			template<class TA, class TC, class F>
			void sylrk_explicit(F alpha, TA A, int stride_row_a, int stride_col_a, F beta, TC C, int stride_row_c, int stride_col_c, int m_max, int r_m_max, int q_m, int q_k, int r_m, int r_k, int offset_blocks) {
				//	std::cout<<"explicit:"<<r_m_max<<"\n";
				F* _A = new F[MC * MC]();
				F* _B = new F[MC * MC]();
				F* _C = new double[MC * MC]();
				for (int j = 0; j < m_max; j++) {

					int nc = (j != m_max - 1 || r_m_max == 0) ? MC : r_m_max;
					for (int l = 0; l < q_k; l++) {
						int kc = (l != q_k - 1 || r_k == 0) ? MC : r_k;
						int inc_a = j * stride_col_a * MC + l * MC * stride_row_a + offset_blocks * MC * stride_col_a;
						pack_A(nc, kc, A + inc_a, stride_row_a, stride_col_a, _A);

						F _beta = (l == 0) ? beta : F(1.0);
						for (int i = 0; i <= j + offset_blocks; i++) {
							int mc = (i != (q_m)-1 || r_m == 0) ? MC : r_m;
							int inc_b = i * stride_col_a * MC + l * MC * stride_row_a;
							pack_B(kc, mc, A + inc_b, stride_col_a, stride_row_a, _B);

							int inc_c = j * MC * stride_col_c + i * MC * stride_row_c;

							if (i != (j + offset_blocks)) {
								gemm_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
							}
							else {
								sylrk_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
							}
						}
					}
					//std::cout<<"C:\n";
	//printmat(C,17,17);


	//std::cin.get();
				}
				delete[] _A;
				delete[] _B;
				delete[] _C;

			}
			template<class TA, class TC, class F>
			void sylrk_single(int n, int k, F alpha, TA A, int stride_row_a, int stride_col_a, F beta, TC C, int stride_row_c, int stride_col_c) {

				if (alpha == F(0.0) || k == 0) {
					trlscal(n, n, beta, C, stride_row_c, stride_col_c); //there are no matrix A, B to add to C with
					return;
				}

				int q_m = (n + MC - 1) / MC; //number of vertical blocks of A
				int q_k = (k + MC - 1) / MC;

				int r_m = n % MC;
				int r_k = k % MC;

				F* _A = new F[MC * MC]();
				F* _B = new F[MC * MC]();
				F* _C = new double[MC * MC]();
				for (int j = 0; j < q_m; j++) {
					int nc = (j != q_m - 1 || r_m == 0) ? MC : r_m;

					for (int l = 0; l < q_k; l++) {
						int kc = (l != q_k - 1 || r_k == 0) ? MC : r_k;
						int inc_a = j * stride_col_a * MC + l * MC * stride_row_a;
						pack_A(nc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
						F _beta = (l == 0) ? beta : F(1.0);
						for (int i = 0; i <= j; i++) {
							int mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
							int inc_b = i * stride_col_a * MC + l * MC * stride_row_a;

							pack_B(kc, mc, A + inc_b, stride_col_a, stride_row_a, _B);

							int inc_c = j * MC * stride_col_c + i * MC * stride_row_c;
							if (i != j) {

								gemm_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);

							}
							else {
								sylrk_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
							}
						}

					}
				}
				delete[] _A;
				delete[] _B;
				delete[] _C;
			}
			
			export template<class TA, class TC, class F>
			void sylrk(int n, int k, F alpha, TA A, int stride_row_a, int stride_col_a, F beta, TC C, int stride_row_c, int stride_col_c) {
				if (alpha == F(0.0) || k == 0) {
					trlscal(n, n, beta, C, stride_row_c, stride_col_c); //there are no matrix A, B to add to C with
					return;
				}

				int q_m = (n + MC - 1) / MC; //number of vertical blocks of A
				int q_k = (k + MC - 1) / MC;

				int r_m = n % MC;
				int r_k = k % MC;


				if (n < 200 && k < 200) {
					F* _A = new F[MC * MC]();
					F* _B = new F[MC * MC]();
					F* _C = new double[MC * MC]();

					for (int j = 0; j < q_m; j++) {
						int nc = (j != q_m - 1 || r_m == 0) ? MC : r_m;

						for (int l = 0; l < q_k; l++) {
							int kc = (l != q_k - 1 || r_k == 0) ? MC : r_k;
							int inc_a = j * stride_col_a * MC + l * MC * stride_row_a;
							pack_A(nc, kc, A + inc_a, stride_row_a, stride_col_a, _A);
							F _beta = (l == 0) ? beta : F(1.0);
							for (int i = 0; i <= j; i++) {
								int mc = (i != q_m - 1 || r_m == 0) ? MC : r_m;
								int inc_b = i * stride_col_a * MC + l * MC * stride_row_a;

								pack_B(kc, mc, A + inc_b, stride_col_a, stride_row_a, _B);

								int inc_c = j * MC * stride_col_c + i * MC * stride_row_c;
								if (i != j) {

									gemm_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);

								}
								else {
									sylrk_macro_kernel(nc, mc, kc, alpha, _beta, _A, _B, C + inc_c, _C, stride_row_c, stride_col_c);
								}
							}

						}
					}

					delete[] _A;
					delete[] _B;
					delete[] _C;
				}
				int n_threads = std::thread::hardware_concurrency();
				int rem = 0;
				TC Cs = C;
				TA As = A;

				//chunkM is in reality chunkN
				if (true) {
					int chunkM;
					int chunkQM = q_m;

					if (q_m > n_threads) {
						chunkQM = q_m / n_threads; //number of vertical blockpieces in A per thread
						rem = q_m % n_threads;
						chunkM = chunkQM * MC;
					}
					else {
						chunkQM = 1;
						chunkM = MC;
						n_threads = q_m;

					}

					std::vector<std::thread> ts(n_threads);
					int leftover = (rem == 0) ? n - (n_threads - 1) * chunkQM * MC : n - (n_threads - 1 - rem) * chunkQM * MC - rem * (chunkQM + 1) * MC; //number of leftover columns
					//int panels_so_far=(rem==0)?(n_threads-1)*chunkQM:(n_threads-1-rem)*chunkQM+rem*(chunkQM+1); //number of leftover columns
					int panels_so_far = 0;

					for (int i = 0; i < n_threads - 1; i++) {
						if (rem == 0) {
							//F alpha, T A, int stride_row_a, int stride_col_a,F beta, T C, int stride_row_c, int stride_col_c, int m_max,int q_m, int q_k, int r_m, int r_k
							ts[i] = std::thread(sylrk_explicit<TA,TC, F>, alpha, As, stride_row_a, stride_col_a, beta, Cs, stride_row_c, stride_col_c, chunkQM, 0, q_m, q_k, r_m, r_k, panels_so_far); //there are no matrix A, B to multiply C with
							Cs += stride_col_c * chunkM;
							panels_so_far += chunkQM;
						}
						else {
							//	std::cout<<"hier\n";
							ts[i] = std::thread(sylrk_explicit<TA,TC, F>, alpha, As, stride_row_a, stride_col_a, beta, Cs, stride_row_c, stride_col_c, chunkQM + 1, 0, q_m, q_k, r_m, r_k, panels_so_far); //there are no matrix A, B to multiply C with
							Cs += stride_col_c * (chunkM + MC);
							rem--;
							panels_so_far += chunkQM + 1;
						}

					}
					ts[n_threads - 1] = std::thread(sylrk_explicit<TA,TC, F>, alpha, As, stride_row_a, stride_col_a, beta, Cs, stride_row_c, stride_col_c, chunkQM, r_m, q_m, q_k, r_m, r_k, panels_so_far);


					for (int i = 0; i < n_threads; i++) {
						ts[i].join();
					}

				}
				else {


				}


			}
			
			
			/*Solves A=LDL^T with diagonal D and lower triangular L. Result is stored back into A (in place)
			This function calculates a variant of the Cholesky decomposition. The traversal for the factors is 
			column wise (Cholesky–Crout traversal).*/
			export template<class T,class F>
			void choi_single(size_t n, T A, int stride_row, int stride_col){
				F* D=new F[n]();
				for (int j=0;j<n;j++){
					for (int i=0;i<j;i++){
						D[j]-=A[j*stride_col+i*stride_row]*A[j*stride_col+i*stride_row]*D[i];
					}
					A[j*stride_col+j*stride_row]+=D[j];
					D[j]=A[j*stride_col+j*stride_row];
					F D_inv=1/D[j];
					for (int i=j+1;i<n;i++){	
						F sum=0.0;		
						for (int t=0;t<j;t++){
							sum-=A[i*stride_col+t*stride_row]*A[j*stride_col+t*stride_row]*D[t];
						}
						
						A[i*stride_col+j*stride_row]+=sum;
						A[i*stride_col+j*stride_row]*=D_inv;
						A[j*stride_col+i*stride_row]=0.0; //can be removed if I don't want to have zeros
					}
				}
				delete[] D;
			}
			
			/*Solves A=LDL^T with diagonal D and lower triangular L. Result is stored back into A (in place).
			Important: In contrast to choi, this function assumes that the input matrix A is in triangular packed format,
			i.e. only the lower triangle including the diagonal is stored.
			This function calculates a variant of the Cholesky decomposition. The traversal for the factors is 
			column wise (Cholesky–Crout traversal).
			TODO: Matrix strides not working yet*/
			export template<class T,class F>
			void choip_single(size_t n, T A, int stride_row, int stride_col){
				F* D=new F[n]();
				for (int j=0;j<n;j++){
					int ix1=j*0.5*(j+1)+j*stride_col;
					for (int i=0;i<j;i++){
						D[j]-=A[ix1+i]*A[ix1+i]*D[i];
						
					}
					A[ix1+j]+=D[j];
					D[j]=A[ix1+j];
					F D_inv=1/D[j];
					for (int i=j+1;i<n;i++){
						F sum=0.0;
						int ix2=i*0.5*(i+1)+i*stride_col;
						for (int t=0;t<j;t++){
							sum-=A[ix2+t]*A[ix1+t]*D[t];
						}
						A[ix2+j]+=sum;
						A[ix2+j]*=D_inv;
					}
				}
				delete[] D;
			}


			//inverse of lowertriangular matrix L, i.e. calculates L^{-1}
			template<class TA, class TC, class F>
			void lower_inv(int n, TA A, int stride_row_a, int stride_col_a, TC C, int stride_row_c, int stride_col_c) {
				for (int i = 0; i < n; i++) {
					for (int j = 0; j <= i; j++) {
						if (j = i) {
							C[i * stride_col_c + i*stride_row_c] = 1 / A[i*stride_col_a + i*stride_row_a];
						}
						else {
							F sum = 0.0;
							for (int k = j; k <= i - 1; k++) {
								sum -= A[i*stride_col_a + k*stride_row_a] + C[k * stride_col_c + j*stride_row_c];
							}
							sum /= A[i * stride_col_a + i*stride_row_a];
							C[i * stride_col_c + j*stride_row_a] = sum;
						}
					}
				}
			}

			//inverse of A=D*L,whereas D is diagonal and L lower triangular
			template<class TA, class TC, class F>
			void diag_lower_inv(int n, TA A, int stride_row_a, int stride_col_a, TC C, int stride_row_c, int stride_col_c) {
				for (int i=0;i<n;i++){
					for (int j=0;j<=i;j++){
						if (j==i){
						
							C[i*stride_col_c+i*stride_row_c]=F(1.0)/A[i*stride_col_a+i*stride_row_a];
						}
						else{
							F sum=0.0;
							for (int k=j;k<=i-1;k++){
								sum-=A[i*stride_col_a+k*stride_row_a]*C[k*stride_col_c+j*stride_row_c];
							}
							C[i*stride_col_c+j*stride_row_c]=sum;
						}
					}
				}
			}		

			
			//Calculates D*L, whereas D is diagonal and L is lower triangular.
			template<class TD, class TA, class TC,class F>
			void dl_gemm_micro_kernel(int n, int k,TD D, int stride_row_d, int stride_col_d, TA A, int stride_row_a, int stride_col_a, TC C, int stride_row_c, int stride_col_c){
				
				F* _D=new double[n];
				
				//Copy the diagonal elements for caching
				for (int i=0;i<n;i++){
					_D[i]=D[i*stride_col_d+i*stride_row_d];
				}

				for (int i=0;i<n;i++){
					for (int j=0;j<k;j++){
						C[i*stride_col_c+j*stride_row_c]=A[i*stride_col_a+j*stride_row_a]*_D[i];
					}
				}
				delete _D;
			}			

			//Calculates L*D*y=b for=LDL^T
			template<class TA, class TB, class F>
			void choi_forward_sub(int n, TA A, int stride_row_a, int stride_col_a, TB b, int  stride_b, F* y) {

				for (int i = 0; i < n; i++) {
					F sum = b[i*stride_b];
					for (int j = 0; j < i; j++) {
						sum -= A[i * stride_col_a + j * stride_row_a] * A[j * stride_col_a + j * stride_row_a] * y[j];
					}
					sum /= A[i * stride_col_a + i * stride_row_a];
					y[i] = sum;
				}
			}
			
			
			/*Calculates L*D*Y=B for=LDL^T and matrices Y,B. The matrix A=L*D is already in packed storage. The following is assumed:
			* the diagonal of L consists of ones
			* D and L are stored in a single matrix A
			* the diagonal of L is implicitly assumed and replaced by the diagonal of D, ergo A[i][i] indicates D[i][i].
			*/
			template<class TA, class TB, class TY,class F>
			void choi_forward_sub(int n, int m, TA A, int stride_row_a, int stride_col_a, TB B, int  stride_row_b, int stride_col_b, TY Y, int stride_row_y, int stride_col_y) {
				F* y=&Y[0];
				F* b=&B[0];
				for (int a=0;a<m;a++){
					for (int i = 0; i < n; i++) {
						F sum = b[i*stride_col_b];
						
						for (int j = 0; j < i; j++) {
							
							sum -= A[i * stride_col_a + j * stride_row_a] * A[j * stride_col_a + j * stride_row_a] * y[j*stride_col_y];
							
						}
						sum /= A[i * stride_col_a + i * stride_row_a];
						y[i*stride_col_y] = sum;
					}
					y+=stride_row_y;
					b+=stride_row_b;
				}
			}		

			template<class TA, class TX, class F>
			void choi_backward_sub(int n, TA A, int stride_row_a, int stride_col_a, F* y, TX x, int stride_x) {
				for (int i = n - 1; i >= 0; i--) {
					x[i * stride_x] += y[i];
					for (int j = i - 1; j >= 0; j--) {
						x[j * stride_x] -= A[j * stride_col_a + i * stride_row_a] * x[i * stride_x];
					}
				}
			}

			export template<class TA, class TB, class TX, class F>
			void choi_solve(int n, TA A, int stride_row_a, int stride_col_a, TB b, int stride_b, TX x, int stride_x) {
				for (int i = 0; i < n; i++) {
					x[i * stride_x] = F(0.0);
				}
				F* y=new F[n];
				choi_forward_sub<TA,TB,F>(n, A, stride_row_a, stride_col_a, b, stride_b, y);
				choi_backward_sub<TA,TX,F>(n, A, stride_col_a, stride_row_a, y, x, stride_x);
				delete y;
			}
			
			export template<class T,class F>
			void choi(int n, T A, const int stride_row_a, const int stride_col_a){
				int BLOCKSIZE=200;
				int d;
				int q;
				int rem;
				if (n<BLOCKSIZE){
					d=n;
					q=1;
					rem=0;
				}
				else{
					d=BLOCKSIZE;
					q=n/d;
					rem=n%d;
				}
				T A11=A;
				T A21=A+d*stride_col_a;
				T A22=A+d*stride_col_a+d*stride_row_a;

				F* temp1=new F[d*d]();
				
				F* temp2=new F[(n-d)*d]();

				for (int i=0;i<q;i++){

					choi_single<T,F>(d, A11, stride_row_a,stride_col_a);
					//diag_upper_inv_p<T,F*,F>(d,A11,stride_col_a,stride_row_a,temp1,1,d);
					//choi_forward_sub<T,T,F>(d, A11, stride_row_a, stride_col_a, b, stride_b, y);
					
					/*Instead of calculating A21*(D1*L11^T)^-1=L21 we are solving A21^T=L11*D1*X with X=L21
					*/
					dcopy(n-d,d,A21,stride_row_a,stride_col_a,temp2,n-d,1);
					choi_forward_sub<T,F*,T,F>(d, n-d, A11, stride_row_a, stride_col_a,temp2, 1, n-d, A21, stride_col_a, stride_row_a);
				
					//Now calculate L22
					dl_gemm_micro_kernel<T,T,F*,F>(d,n-d,A11,stride_row_a, stride_col_a, A21, stride_col_a, stride_row_a,temp2,1,n-d);
					A11+=d*stride_row_a+d*stride_col_a;
					gemm(n-d,n-d,d,F(-1.0),A21,stride_row_a,stride_col_a,temp2,1,n-d,F(1.0),A11,stride_row_a,stride_col_a);
					A21=A11+d*stride_col_a;	
					n-=d;
				}

				if (rem){
					d=rem;
					choi_single<T,F>(d, A11, stride_row_a,stride_col_a);
					dcopy(n-d,d,A21,stride_row_a,stride_col_a,temp2,d,1);
					choi_forward_sub<T,F*,T,F>(d, n-d, A11, stride_row_a, stride_col_a,temp2, 1, n-d, A21, stride_col_a, stride_row_a);
					
					dl_gemm_micro_kernel<T,T,F*,F>(d,n-d,A11,stride_row_a, stride_col_a, A21, stride_col_a, stride_row_a,temp2,1,n-d);
					A11+=d*stride_row_a+d*stride_col_a;
					gemm(n-d,n-d,d,F(-1.0),A21,stride_row_a,stride_col_a,temp2,1,n-d,F(1.0),A11,stride_row_a,stride_col_a);				
				}
				
				delete[] temp1;
				delete[] temp2;
				
			}
		
		}
	}
}