import tests.performance;
import tests;
#include <iostream>
#include<vector>
#include<sstream>
#include <iomanip>
#include <ostream>
//#include <chrono>


bool run_performance_tests(std::ostream& os, bool save_metrics){
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y_%Hh-%Mm-%Ss");
	

	std::vector<opt::test::PerformanceTest> v;
	v.push_back(opt::test::PerformanceTest("dgmm_nn_test_square_800_d",opt::test::perf::dgmm_nn_test_square_800_d));
	v.push_back(opt::test::PerformanceTest("dgmm_nn_test_square_1600_d",opt::test::perf::dgmm_nn_test_square_1600_d));
	v.push_back(opt::test::PerformanceTest("dgmm_nn_test_square_3200_d",opt::test::perf::dgmm_nn_test_square_3200_d));
	//v.push_back(opt::test::PerformanceTest("dgmm_nn_test_square_5000_d",opt::test::perf::dgmm_nn_test_square_5000_d));
	bool result=true;
	
	for (int i=0; i<v.size(); i++){
		bool temp=v[i].run_test();
		if (temp!= true){
			result=temp;
		}
		if ((i%3)==0){
			os<<i+1<<" out of "<<v.size()<<" performance tests finished \n";
		}
	}
	if (save_metrics==true){
		opt::test::perf::save_metrics(v,oss.str());
	}
	
	return result;

}


bool run_tests(std::ostream& os,bool save_stats){

	bool test1=run_performance_tests(os, save_stats); //run performance tests
	
	return test1;
}

int main(){
	bool save_stats=true; //save test logs to file
	bool test_result=run_tests(std::cout,save_stats);
		
	if (test_result){
		std::cout<<"Tests finished without errors\n";
		return 1;
	}
	else{
		std::cout<<"At least one test finished erroneously\n";
		return 0;
	}
	
	
}