module;
#include <vector>
#include <random>
#include <ostream>
#include<cmath>
#include <filesystem>
#include <iostream>
#include <iostream>
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
					F val=A[i]-=B[i];
					val=(val<0)?-val:val;
					if (val>tol){
						std::cout<<A[i]<<" vs. "<<B[i]<<"\n";
						return false;
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
		
	
		}
	
	}

}

