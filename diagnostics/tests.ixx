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

			private:

			bool (*test_function)(std::ostream& os, TestInterface& v);

			public:
			const std::string name;
			
			virtual bool run_test()=0;
			
			

			TestInterface(std::string _name, bool (*f)(std::ostream&, TestInterface& v)): name(_name), test_function(f){
			}
			
		};

		export class PerformanceTest:public TestInterface{
			private:
			bool (*test_function)(std::ostream& os, PerformanceTest& v);

			public:
			std::chrono::seconds time_taken;

			//PerformanceTest(std::string _name, bool (*f)(std::ostream&, TestInterface& v)): TestInterface(_name,f){	
			//}


			PerformanceTest(std::string _name, bool (*f)(std::ostream&, PerformanceTest& v)): TestInterface(_name,nullptr), test_function(f){	
			}	

			bool run_test() override{

				return test_function(std::cout,*this);

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

