module;

#include<string>
#include <ostream>
export module tests.performance;
export import :transform.cpu;

export import tests;

//import string;

export class PerformanceTest:public TestInterface{
	public:

	PerformanceTest(std::string _name, bool (*_f)(std::ostream&)): TestInterface(_name,_f){
	}


	
};
