module;
#include <vector>
#include <random>
#include <ostream>
#include <chrono>
#include "../../gpu/linear_algebra/gemm.cuh"
export module tests.performance:transformation.gpu;

import tests;

namespace opt{
	namespace test{

		namespace perf{
			
			bool gemm_test_square_f32_gpu(std::ostream& os, PerformanceTest& v, int n){
				float* A=new float[n*n];
				float* B=new float[n*n];
				float* C=new float[n*n];				
				std::random_device os_seed;
				const uint32_t seed=os_seed();
				std::mt19937 rng(seed);
				opt::test::fill_container_randomly<float*,float>(rng,A,n*n);
				opt::test::fill_container_randomly<float*,float>(rng,B,n*n);
					
				// Get starting timepoint 
			  	auto start = std::chrono::high_resolution_clock::now(); 
				//Function to measure
				gemm_f32(n,n,n,1.0,A,B,0.0,C);
				// Get ending timepoint 
				auto stop = std::chrono::high_resolution_clock::now(); 

				auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start); 
			  	
				delete[] A;
				delete[] B;
				delete[] C;
				os << v.name<<" took "
					 << duration.count() << " seconds \n";
			
				v.time_taken=duration;
				
				return true;
			}		
			

			export bool gemm_test_square_3200_f32_gpu(std::ostream& os, PerformanceTest& v){
				return gemm_test_square_f32_gpu(os,v,3200);
			}
		
		
		
		}
	
	}

}
