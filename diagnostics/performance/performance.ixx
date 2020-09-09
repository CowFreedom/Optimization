module;
#include<string>
#include <ostream>
#include <vector>
export module tests.performance;
export import :transform.cpu;

import tests;

//import string;
namespace opt{
	namespace test{
	
		export class PerformanceTest:public TestInterface{
			private:
			
			public:

			PerformanceTest(std::string _name, bool (*_f)(std::ostream&, TestInterface& v)): TestInterface(_name,_f){
			
				
			}	
		};
		
		
		export std::vector<PerformanceTest> get_performance_tests(){
			std::vector<PerformanceTest> v;
			v.push_back(PerformanceTest("dgmm_nn_test_square_5000_d",opt::test::perf::dgmm_nn_test_square_5000_d));
			return v;
		}
		
	}
	
	

}



	

