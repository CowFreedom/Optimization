import tests.performance;
import tests;
#include <iostream>
//#include <chrono>

bool test1(std::ostream& os, opt::test::TestInterface& v){
	os<<"All good\n";
	return true;
}

int main(){
	auto performance_tests=opt::test::get_performance_tests();
	performance_tests[0].run_test();
	
}