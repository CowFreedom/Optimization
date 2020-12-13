module;
#include <vector>
#include <random>
#include <ostream>
#include <iostream> //remove
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
					if (val>tol || (std::isinf(A[i])) || (std::isinf(B[i])) ||(std::isnan(A[i])) || (std::isnan(B[i]))){
					//std::cout<<"Error at:"<<i<<"\n";
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
			
			template<class T, class F>
			bool verify_upperlower(const T C1, const T C2, int n, const char selection, F tol){
				switch (selection){
					case 'U':{
							for (int i=0;i<n;i++){
								for (int j=i;j<n;j++){
									if (std::abs(C1[i*n+j]-C2[i*n+j])>tol){
									//std::cin.get();
									return false;
								}
								}
								
							}
							
							break;		
					}
					
					case 'L':{
							for (int i=0;i<n;i++){
								for (int j=0;j<i;j++){
									if (std::abs(C1[i*n+j]-C2[i*n+j])>tol){
									return false;
								}
								}	
							}
							
							break;		
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
				
				opt::math::cpu::gemm(m,n,k,alpha,A1,1,k,B1,1,n,beta,res,1,n);
	
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
				
				opt::math::cpu::gemm(m,n,k,alpha,A1,1,k,B1,1,n,beta,res,1,n);
	
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
				double beta=1.0;

				double* A1=new double[m*k];
				double* B1=new double[n*k];
				double* res=new double[m*n];

				opt::test::parse_csv("correctness/test_data/matrix_mul/A4.txt",A1, m*k,"	");
				opt::test::parse_csv("correctness/test_data/matrix_mul/B4.txt",B1, k*n,"	");	
				
				opt::math::cpu::gemm(m,n,k,alpha,A1,1,k,B1,1,n,beta,res,1,n);
	
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
			
			export bool matrix_multiplication_4(std::ostream& os, CorrectnessTest& v){

				std::random_device rd;				
					
				std::uniform_real_distribution<> dist(1,300); // distribution in range [1, 6];		

				bool is_correct;
				for (int i=0;i<20;i++){
					int n=dist(rd);
					int k=dist(rd);
					double beta=0;		
					double alpha=-2.0;
					double* A=new double[n*k];
					double* control=new double[n*n];
					double* res=new double[n*n];

					opt::test::fill_container_randomly<double*,double>(rd, A,n*k);
					opt::math::cpu::syurk(n, k,alpha, A, 1, k,beta,res,1,n);	
					opt::math::cpu::gemm(n,n,k,alpha,A,1,k,A,k,1,beta,control,1,n); //calculate control result, AA^T=C
					is_correct=verify_upperlower(control, res,n, 'U', 0.0001);
					delete[] A;
					delete[] control;
					delete[] res;	
					if (is_correct==false){
						return false;
					}
		
				}

				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}	

			export bool matrix_multiplication_5(std::ostream& os, CorrectnessTest& v) {

				std::random_device rd;

				std::uniform_real_distribution<> dist(1, 300); // distribution in range [1, 6];		

				bool is_correct;
				for (int i = 0; i < 20; i++) {
					int n = dist(rd);
					int k = dist(rd);
					double beta = 0;
					double alpha = -2.0;
					double* A = new double[n * k];
					double* control = new double[n * n];
					double* res = new double[n * n];

					opt::test::fill_container_randomly<double*, double>(rd, A, n * k);
					opt::math::cpu::sylrk(n, k, alpha, A, 1, k, beta, res, 1, n);
					opt::math::cpu::gemm(n, n, k, alpha, A, 1, k, A, k, 1, beta, control, 1, n); //calculate control result, AA^T=C
					is_correct = verify_upperlower(control, res, n, 'L', 0.0001);
					delete[] A;
					delete[] control;
					delete[] res;
					if (is_correct == false) {
						return false;
					}

				}

				if (is_correct) {
					v.test_successful = true;
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
				
				opt::math::cpu::choi_single<double*,double>(n,A,1,n);
				opt::math::cpu::choip_single<double*,double>(n,A_packed,1,0);
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
			
			//Tests is choi and choip are equal and calculate the correct result
			export bool cholesky_2(std::ostream& os, CorrectnessTest& v){
	
				std::random_device rd;
				std::uniform_real_distribution<> dist(1, 300); // distribution in range [1, 6];		
				bool is_correct;
				for (int i = 0; i < 20; i++) {
					int n = dist(rd);
					int k = dist(rd);;
					int m=1; //lower threshold for eigenvalue of 0.5*(A+A^T)+m*I eigenvalue
					double* A = new double[n * n]();
					double* L = new double[n*n]();
					double* D = new double[n*n]();
					double* C = new double[n * n]();

					double* temp = new double[n * n]();
					opt::test::fill_container_randomly<double*, double>(rd, A, n * n);
					//Create symmetric matrix C
					for (int i=0;i<n;i++){
						for (int j=0;j<n;j++){
							if (i!=j){
								C[i*n+j]=0.5*(A[i*n+j]+A[i+j*n]);
							}
							else{
								C[i*n+j]=0.5*(A[i*n+j]+A[i+j*n])+m;
							}
						}
					
					}
					
					delete[] A;
					double* C_copy=new double[n*n]();
					for (int i = 0; i < n * n; i++) {
						C_copy[i] = C[i];
					}

					opt::math::cpu::choi<double*, double>(n, C, 1, n);

					for (int i=0; i<n;i++){
						for (int j=0;j<i;j++){
							L[i*n+j]=C[i*n+j];
						}
						L[i*n+i]=1;
					}
					for (int i=0;i<n;i++){
						D[i*n+i]=C[i*n+i];
					}
					opt::math::cpu::gemm(n, n, n, 1.0, D, 1, n, L, n, 1,0.0, temp, 1, n); //calculate intermediate result D*L^T from LDL^T
					opt::math::cpu::gemm(n, n, n, 1.0, L, 1, n, temp, 1, n,0.0, C, 1, n);
					is_correct = verify_calculation(C,C_copy, n*n, 0.1);
					delete[] C;
					delete[] C_copy;
					delete[] temp;
					delete[] L;
					delete[] D;
					
					if (is_correct==false){
						return v.test_successful=false;
						return false;
					}
					
				 }
	
				v.test_successful=true;	
				return true;
			}
			export bool cholesky_solve(std::ostream& os, CorrectnessTest& v) {

				std::random_device rd;

				std::uniform_real_distribution<> dist(1, 20); // distribution in range [1, 6];		

				bool is_correct;
				for (int i = 0; i < 20; i++) {
					int n = dist(rd);
					int m=1;//lower threshold for eigenvalue of 0.5*(A+A^T)+m*I eigenvalue

					double* A = new double[n * n];
					double* C = new double[n * n];
					double* C_copy = new double[n * n];
					opt::test::fill_container_randomly<double*, double>(rd, A, n * n);
					//Create symmetric matrix C
					for (int i=0;i<n;i++){
						for (int j=0;j<n;j++){
							if (i!=j){
								C[i*n+j]=0.5*(A[i*n+j]+A[i+j*n]);
							}
							else{
								C[i*n+j]=0.5*(A[i*n+j]+A[i+j*n])+m;
							}
						}
					}
					delete[] A;
					for (int i = 0; i < n * n; i++) {
						C_copy[i] = C[i];
					}
					//printmat("AAt",C,n,n,std::cout);
					opt::math::cpu::choi<double*, double>(n, C, 1, n);
					
					double* b = new double[n];
					double* x = new double[n];
					double* res = new double[n];
					opt::test::fill_container_randomly<double*, double>(rd, b, n);
					opt::math::cpu::choi_solve<double*,double>(n, C, 1, n, b, 1, x, 1);
					//printmat("C",C,n,n,std::cout);
					//printmat("b",b,n,1,std::cout);
					//printmat("x",x,n,1,std::cout);
					
					//Test if LDL^T composition is valid
					
					double* L=new double[n*n]();
					double* D=new double[n*n]();
					double* temp=new double[n*n]();
					
					for (int i=0; i<n;i++){
						for (int j=0;j<i;j++){
							L[i*n+j]=C[i*n+j];
						}
						L[i*n+i]=1;
					}
					for (int i=0;i<n;i++){
						D[i*n+i]=C[i*n+i];
					}
					opt::math::cpu::gemm(n, n, n, 1.0, D, 1, n, L, n, 1,0.0, temp, 1, n); //calculate intermediate result D*L^T from LDL^T
					opt::math::cpu::gemm(n, n, n, 1.0, L, 1, n, temp, 1, n,0.0, C, 1, n);
					//printmat("LDL^T",C,n,n,std::cout);
					bool ldl_is_correct = verify_calculation(C,C_copy, n*n, 0.01);
					//std::cout<<"Ldl is correct: "<<ldl_is_correct<<"\n";
					delete[] L;
					delete[] D;
					delete[] temp;
					
					//end ldl test
					opt::math::cpu::dgmv(n,n, 1.0, C_copy, 1, n,x,1, 0.0, res,1);
					//printmat("Ax:",res,n,1,std::cout);
					is_correct = verify_calculation(b,res, n, 0.1);

					
					delete[] C;
					delete[] b;
					delete[] C_copy;
					delete[] x;
					delete[] res;
					if (is_correct == false) {
						return false;
					}

				}

				if (is_correct) {
					v.test_successful = true;
					return true;
				}
				return false;
			}
		
	
		}
	
	}

}

