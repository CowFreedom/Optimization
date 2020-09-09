module;
#include <vector>
#include <random>
#include <ostream>
#include <chrono>
export module tests.performance:transform.cpu;

import tests;
import optimization.transformation.cpu;


namespace opt{
	namespace test{

		namespace perf{
		
		
		

			export template<class T>
			bool dgmm_nn_test_square_n(int n){
				std::vector<T> A(n*n);
				std::vector<T> B(n*n);
				std::vector<T> C(n*n);
				
				std::random_device dev;
				opt::test::fill_container_randomly<std::vector<T>::iterator,T>(dev,A.begin(),n*n);
				opt::test::fill_container_randomly<std::vector<T>::iterator,T>(dev,B.begin(),n*n);

				return true;
			}		
			
			//Test how quickly a square 5000x5000 matrix with is multiplied
			export bool dgmm_nn_test_square_5000_d(std::ostream& os, TestInterface& v){
		
				int n=5000;

				std::vector<double> A(n*n);
				std::vector<double> B(n*n);
				std::vector<double> C(n*n);
				
				std::random_device dev;
				opt::test::fill_container_randomly<std::vector<double>::iterator,double>(dev,A.begin(),n*n);
				opt::test::fill_container_randomly<std::vector<double>::iterator,double>(dev,B.begin(),n*n);
		
					
				// Get starting timepoint 
				auto start = std::chrono::high_resolution_clock::now(); 
				os<<"Start now\n";
			  
				//Function to measure
				opt::math::dgemm_nn(n,n,n,1.0,A.begin(),1,n,B.begin(),1,n,double(0.0),C.begin(),1,n);
			  os<<"Stop now \n";
				// Get ending timepoint 
				auto stop = std::chrono::high_resolution_clock::now(); 
			  
				// Get duration. Substart timepoints to  
				// get durarion. To cast it to proper unit 
				// use duration cast method 
				auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start); 
			  
				os << v.name<<" took "
					 << duration.count() << " seconds \n";
			
				
			
				return true;
			}
		
		}
	
	}

}





export int mul(){
	return 7;
}
