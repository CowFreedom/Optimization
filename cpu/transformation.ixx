module;
#include <ostream>
export module optimization.transformation.cpu;

#if defined(__clang__)
	#include <x86intrin.h> //SIMD for gcc/clang
#elif defined(__GNUC__) || defined(__GNUG__)
	#include <x86intrin.h> //SIMD for gcc/clang
#elif defined(_MSC_VER)
	#include<immintrin.h> //AVX, AVX2, FMA for VS
#endif

namespace opt{
	namespace math{
		
		/*These are the blocksizes for blocked matrix
		multiplication./

			/*
			constexpr size_t MC = 8; //
			constexpr size_t KC = 12;
			constexpr size_t NC = 12;

			constexpr size_t MR = 4;
			constexpr size_t NR = 6;
			*/
			
			constexpr size_t MC = 64; //
			constexpr size_t KC = 128;
			constexpr size_t NC = 96;

			constexpr size_t MR = 32;
			constexpr size_t NR = 48;
			
			
			

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
						*(buffer + j) = *(B + j * stride_rows);
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
					//std::cout << "i:" <<  << "\n";
					if (i < q - 1) {
						B += NR;
						buffer += kc * NR;
					}

				}
				if (r != 0) {
					std::fill(buffer, buffer + NR * kc, F(0.0));
					for (size_t i = 0; i < kc; i++) {
						for (size_t j = 0; j < r; j++) {
							*(buffer + j) = *(B + j * stride_rows);

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
				/*std::cout<<"zu multiplizierende Matrizen:\n";
									std::cout<<"_A=\n";
									printmat(A,MR,KC);
									std::cout<<"\n _B=\n";
									printmat(B,KC,NR);

									*/
									//std::array<F,MR*NR> AB;	//buffer for result AB
				double AB[MR * NR];
				
				//std::array<F,MR*NR>& ABr=&AB;
				std::fill(AB, AB + MR * NR, double(0.0));
				for (size_t k = 0; k < kc; k++) {
					for (size_t j = 0; j < MR; j++) {
						//std::cout << "bis hier\n";
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
				F AB[MR * NR];
				//std::array<F,MR*NR>& ABr=&AB;
				std::fill(AB, AB + MR * NR, F(0.0));

				for (size_t k = 0; k < kc; k++) {
					for (size_t j = 0; j < MR; j++) {
						for (size_t i = 0; i < NR; i++) {
							AB[j * NR + i] += A[j] * B[i];
							/*if (isFirst){
								std::cout << A[j] << " vs. " << B[i] << "\n";
								}
								*/
							//

						}
						//std::cin.get();
						//std::cout<<"\n";
					}
					if (k < KC - 1) {
						A += MR;
						B += NR;
					}
				}
				/*
				if (isFirst) {
					std::cout << "Temp A:\n";
					printmat(A, MR, KC);
					std::cout << "\nTemp B:\n";
					printmat(B, KC, NR);
					std::cout<<"result _AB=\n";
					printmat(AB,MR,NR);
					std::cin.get();
				}

				*/
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

			template<class T>
			void test1(size_t kc, T C, size_t stride_row_c, int stride_col_c, int num) {
				for (size_t i = 0; i < MR; ++i) {
					for (size_t j = 0; j < NR; ++j) {
						C[i * stride_col_c + j * stride_row_c] = num;
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
				double iter = 1;
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

			export template<class T, class F>
			void dgemm_nn(size_t m, size_t n, size_t k, F alpha, T A, size_t stride_row_a, size_t stride_col_a,
				T B, size_t stride_row_b, size_t stride_col_b, F beta, T C, size_t stride_row_c, size_t stride_col_c) {
				size_t q_m = (m + MC - 1) / MC; //number of horizontal blocks
				size_t q_n = (n + NC - 1) / NC;
				size_t q_k = (k + KC - 1) / KC;

				size_t r_m = m % MC;
				size_t r_n = n % NC;
				size_t r_k = k % KC;

				//initializing buffers
				F _A[KC * MC];
				F _B[KC * NC];
				F _C[MC * NC];
				
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

			}


		
		/*Matrix times scalar*/
		template<class T,class F>
			void dgms(T A, T res, int n, int m, F s){
				for (int i=0;i<n;i++){
					for (int j=0;j<m;j++){
						res[i*m+j]=s*A[i*m+j];					
				}			
			}
		}

		

	}

}