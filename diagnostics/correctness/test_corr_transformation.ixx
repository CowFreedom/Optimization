module;
#include <vector>
#include <random>
#include <ostream>
#include<cmath>
#include <filesystem>
#include<fstream> //checken warum das importiert werden muss

export module tests.correctness:transformation;

import tests;

import optimization.transformation;
import optimization.solvers;

namespace opt{
	namespace test{

		namespace corr{
			
			template<class T, class F>
			bool verify_calculation(T A, T B, int n_elems, F tol){
			
				for (int i=0;i<n_elems;i++){
					F val=A[i]-B[i];
					val=(val<0)?-val:val;
					if (val>tol){
						return false;
					}
				}
				return true;
			}
			

			template<class T, class F>
			bool verify_nonpacked_vs_packed(const T nonpacked_mat, const T packed_mat, int n, F tol){
				for (int i=0;i<n;i++){
					for (int j=0;j<=i;j++){
						int ix1=i*n+j;
						int ix2=i*0.5*(i+1)+i*0+j;
						
						F val=nonpacked_mat[ix1]-packed_mat[ix2];
						val=(val<0)?-val:val;
						if (val>tol){
							//	std::cout<<nonpacked_mat[ix1]<<" vs. "<<packed_mat[ix2]<<"at position "<<i*n+j<<"\n";
								return false;
							}
						}				
					}

				return true;
			}
			
			
			export bool matrix_multiplication_1(std::ostream& os, CorrectnessTest& v){
				int m=2;
				int n=2;
				int k=2;
				double alpha=1.0;
				double beta=0;

				double* A1=new double[m*k];
				double* B1=new double[n*k];
				double* res=new double[m*n];

				opt::test::parse_csv("correctness/test_data/matrix_mul/A1.txt",A1, m*k,"	");
				opt::test::parse_csv("correctness/test_data/matrix_mul/B1.txt",B1, k*n,"	");	
				
				opt::math::cpu::dgemm_nn(m,n,k,alpha,A1,1,k,B1,1,n,beta,res,1,n);
	
				double* np_res=new double[m*n];
				opt::test::parse_csv("correctness/test_data/matrix_mul/C1.txt",np_res, m*n,"	");
	
				bool is_correct=verify_calculation(np_res,res,m*n,0.0000001);
				delete[] A1;
				delete[] B1;
				delete[] res;
				delete[] np_res;
				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}
			
			export bool matrix_multiplication_2(std::ostream& os, CorrectnessTest& v){
				int m=1000;
				int n=1000;
				int k=1000;
				double alpha=1.0;
				double beta=0;

				double* A1=new double[m*k];
				double* B1=new double[n*k];
				double* res=new double[m*n];

				opt::test::parse_csv("correctness/test_data/matrix_mul/A2.txt",A1, m*k,"	");
				opt::test::parse_csv("correctness/test_data/matrix_mul/B2.txt",B1, k*n,"	");	
				
				opt::math::cpu::dgemm_nn(m,n,k,alpha,A1,1,k,B1,1,n,beta,res,1,n);
	
				double* np_res=new double[m*n];
				opt::test::parse_csv("correctness/test_data/matrix_mul/C2.txt",np_res, m*n,"	");
	
				bool is_correct=verify_calculation(np_res,res,m*n,0.00001);
				delete[] A1;
				delete[] B1;
				delete[] res;
				delete[] np_res;
				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}
			
			export bool matrix_multiplication_3(std::ostream& os, CorrectnessTest& v){
				int m=425;
				int n=17;
				int k=31;
				double alpha=1.0;
				double beta=0;

				double* A1=new double[m*k];
				double* B1=new double[n*k];
				double* res=new double[m*n];

				opt::test::parse_csv("correctness/test_data/matrix_mul/A4.txt",A1, m*k,"	");
				opt::test::parse_csv("correctness/test_data/matrix_mul/B4.txt",B1, k*n,"	");	
				
				opt::math::cpu::dgemm_nn(m,n,k,alpha,A1,1,k,B1,1,n,beta,res,1,n);
	
				double* np_res=new double[m*n];
				opt::test::parse_csv("correctness/test_data/matrix_mul/C4.txt",np_res, m*n,"	");
	
				bool is_correct=verify_calculation(np_res,res,m*n,0.00001);
				delete[] A1;
				delete[] B1;
				delete[] res;
				delete[] np_res;
				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}
			
			//Tests is choi and choip are equal and calculate the correct result
			export bool cholesky_1(std::ostream& os, CorrectnessTest& v){
				int n=3;
				double* A=new double[n*n]();
				double* A_packed=new double[(n*(n+1))/2]();	
				A[0]=1;
				A[1*n+0]=4;
				A[1*n+1]=7;
				A[2*n+0]=7;
				A[2*n+1]=8;
				A[2*n+2]=9;
				A_packed[0]=1;
				A_packed[1]=4;
				A_packed[2]=7;
				A_packed[3]=7;
				A_packed[4]=8;
				A_packed[5]=9;	
				
				double* B=new double[n*n](); //manually calculated cholesky decomposition
				B[0]=1;
				B[1*n+0]=4;
				B[1*n+1]=-9;
				B[2*n+0]=7;
				B[2*n+1]=20.0/9.0;
				B[2*n+2]=40.0/9.0;
				
				opt::math::cpu::choi<double*,double>(n,A,1,n);
				opt::math::cpu::choip<double*,double>(n,A_packed,1,0);
				bool is_correct=verify_nonpacked_vs_packed(B,A_packed,n,0.000001);
				bool are_same=verify_nonpacked_vs_packed(A,A_packed,n,0.0);
				
				delete[] A;
				delete[] B;
				delete[] A_packed;
				
				is_correct=are_same&&is_correct;
	
				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}
		
	
		}
	
	}

}

