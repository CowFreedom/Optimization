module;
#include <random>
#include <cstring>
#include <iostream>
#include "../../gpu/linear_algebra/gemm.cuh"
export module tests.correctness:transformation.gpu;

import :utility;

import tests;

import optimization.transformation;
import optimization.solvers;


namespace opt{
	namespace test{
		namespace corr{
			export bool matrix_multiplication_1_f32_gpu(std::ostream& os, CorrectnessTest& v){
				std::random_device os_seed;
				const uint32_t seed=os_seed();
				std::mt19937 rng(seed);
				int reps=10;
				int m=64;
				int n=64;
				int k=64;
				float* A=new float[m*k];
				float* B=new float[n*k];
				float* C=new float[m*n];
				float* control=new float[m*n];
				bool is_correct=false;

				for (int i=0;i<reps;i++){
					float alpha=0.5;
					float beta=1.0;			
					fill_container_randomly<float*, float>(rng, A,m*k);
					fill_container_randomly<float*, float>(rng, B,k*n);
					fill_container_randomly<float*, float>(rng, C,m*n);			
					std::memcpy(control,C,m*n*sizeof(float));
					gemm_f32(m, n, k, alpha, A, B, beta, C);
					opt::math::cpu::gemm(m,n,k,alpha,A,1,k,B,1,n,beta,control,1,n);
					is_correct=verify_calculation(control,C,m*n,0.001);
					
					//printmat("GPU",C,m,n,std::cout);
				//	printmat("CPU",control,m,n,std::cout);
					
					
					if (is_correct==false){
						break;
					}
				}
	
				delete[] A;
				delete[] B;
				delete[] C;
				delete[] control;
				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}
			
			export bool matrix_multiplication_2_f32_gpu(std::ostream& os, CorrectnessTest& v){
				std::random_device os_seed;
				const uint32_t seed=os_seed();
				std::mt19937 rng(seed);
				std::uniform_int_distribution<> dist(0,1024); // distribution in range [0, range];
				int reps=10;
				bool is_correct=false;

				for (int i=0;i<reps;i++){
					int m=dist(rng);
					int n=dist(rng);
					int k=dist(rng);
					float* A=new float[m*k];
					float* B=new float[n*k];
					float* C=new float[m*n];
					float* control=new float[m*n];
					float alpha=0.5;
					float beta=1.0;			
					fill_container_randomly<float*, float>(rng, A,m*k);
					fill_container_randomly<float*, float>(rng, B,k*n);
					fill_container_randomly<float*, float>(rng, C,m*n);			
					std::memcpy(control,C,m*n*sizeof(float));
					gemm_f32(m, n, k, alpha, A, B, beta, C);
					opt::math::cpu::gemm(m,n,k,alpha,A,1,k,B,1,n,beta,control,1,n);
					is_correct=verify_calculation(control,C,m*n,0.1);
					
					//printmat("GPU",C,m,n,std::cout);
				//	printmat("CPU",control,m,n,std::cout);
					delete[] A;
					delete[] B;
					delete[] C;
					delete[] control;
					
					if (is_correct==false){
						break;
					}
				}
	

				if (is_correct){
					v.test_successful=true;
					return true;
				}	
				return false;
			}
		}
	}

}