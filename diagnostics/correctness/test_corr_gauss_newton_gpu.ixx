module;
#include <vector>
#include <random>
#include <ostream>
#include<cmath>
#include<thread>
#include <functional>
#include <iostream>
#include <array>
export module tests.correctness:gauss_newton.gpu;

import :utility;

import tests;

import optimization.transformation;
import optimization.gpu.solvers;
import optimization.solvers;

namespace opt{
	namespace test{
		namespace corr{
		
			export bool gauss_newton_example3_gpu(std::ostream& os, CorrectnessTest& v){
			int reps=10;
			std::random_device os_seed;
			const uint32_t seed=os_seed();
			std::mt19937 rng(seed);
			std::uniform_real_distribution<float> dist(-10,10);

			for (int i=0;i<reps;i++){
			
				float tol=1e-10;
				int x1=dist(rng);
				int x2=dist(rng);
				CircleB<const float*, float*,float> c(x1,x2);
				float x0[]={dist(rng),dist(rng)};
				float output[2];
				using std::placeholders::_1;
				using std::placeholders::_2;
				
				std::function<void(const float*, float*)> f1=std::bind(&CircleB<const float*, float*,float>::residual,c,_1,_2);
				std::function<void(const float*, float*)> f2=std::bind(&CircleB<const float*, float*,float>::jacobian,c,_1,_2);
				
				opt::solvers::gpu::GNSGPU<std::function<void(const float*, float*)>,float> gns(f1,f2,c.xdim,c.rdim,std::cout.rdbuf());
				
				gns.tol=tol;
				auto result=gns.optimize(x0,output,c.xdim);
				
				if (result){
					float p1=(*result)[0];
					float p2=(*result)[1];
					float x1=output[0];
					float x2=output[1];
					float val=(p1-x1>=0.0)?p1-x1:p1+x1;
					if(val>0.001){
							return false;						
					}
					val=(p2-x2>=0.0)?p2-x2:p2+x2;
					if(val>0.001){
							return false;						
					}
				}
				else{
					
					return false;
				}		
				
			}
			
			v.test_successful=true;
			return true;
			
			}
		}
	}

}