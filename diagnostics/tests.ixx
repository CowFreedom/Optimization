module;
#include <string>
#include <ostream>
#include <random>
#include <chrono>
#include <iostream>
export module tests;

namespace opt{
	namespace test{
	

	
		export class TestInterface{
			public:
			const std::string name;
			
			bool run_test(){
				
				
				return test_function(std::cout,*this);
			
			}
			
			bool (*test_function)(std::ostream& os, TestInterface& v);

			TestInterface(std::string _name, bool (*f)(std::ostream&, TestInterface& v)): name(_name), test_function(f){
			}
			
		};
		

		export template <class T, class F>
		void fill_container_randomly(std::random_device& dev, T v,int n){
			std::mt19937 rng(dev());
			std::uniform_real_distribution<> dist(1,6); // distribution in range [1, 6];
			for (int i=0;i<n;i++){

				*(v+i)=F(dist(rng));
			
			}
		}
		
		export template<class T>
		void printmat(T v, int N, int M, std::ostream& os) {
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					os << *(v + j + i * M) << "\t";
				}
				os << "\n";
			}
		}

	
	}


}

