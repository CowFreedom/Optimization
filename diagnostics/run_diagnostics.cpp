import tests.performance;
import tests.correctness;
import tests;
#include <iostream>
#include<vector>
#include<sstream>
#include <iomanip>
#include <ostream>


bool run_performance_tests(std::ostream& os, bool save_metrics){
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y_%Hh-%Mm-%Ss");
	
	std::vector<opt::test::PerformanceTest> v;
	v.push_back(opt::test::PerformanceTest("gemm_test_square_800_d",opt::test::perf::gemm_test_square_800_d));
	v.push_back(opt::test::PerformanceTest("gemm_test_square_1600_d",opt::test::perf::gemm_test_square_1600_d));
	v.push_back(opt::test::PerformanceTest("gemm_test_square_3200_d",opt::test::perf::gemm_test_square_3200_d));
	//v.push_back(opt::test::PerformanceTest("gemm_test_square_5000_d",opt::test::perf::gemm_test_square_5000_d));
	#ifdef opt_use_cuda
		v.push_back(opt::test::PerformanceTest("gemm_test_square_3200_f32_gpu",opt::test::perf::gemm_test_square_3200_f32_gpu));
	#endif
	
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



bool run_correctness_tests(std::ostream& os, bool save_metrics){
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%d-%m-%Y_%Hh-%Mm-%Ss");
	

	std::vector<opt::test::CorrectnessTest> v;
	
	//v.push_back(opt::test::CorrectnessTest("gauss_newton_example1",opt::test::corr::gauss_newton_example1));
	//v.push_back(opt::test::CorrectnessTest("gauss_newton_example1_fptr",opt::test::corr::gauss_newton_example1_fptr));
	//v.push_back(opt::test::CorrectnessTest("gauss_newton_example2",opt::test::corr::gauss_newton_example2));
	v.push_back(opt::test::CorrectnessTest("gauss_newton_example3",opt::test::corr::gauss_newton_example3));
	v.push_back(opt::test::CorrectnessTest("gauss_newton_example4",opt::test::corr::gauss_newton_example4));	
//	v.push_back(opt::test::CorrectnessTest("gemm dim(A)=(2,2), dim(B)=(2,2)",opt::test::corr::matrix_multiplication_1));
//	v.push_back(opt::test::CorrectnessTest("gemm dim(A)=(1000,1000), dim(B)=(1000,1000)",opt::test::corr::matrix_multiplication_2));
//	v.push_back(opt::test::CorrectnessTest("gemm dim(A)=(425,17), dim(B)=(17,31)",opt::test::corr::matrix_multiplication_3));

	v.push_back(opt::test::CorrectnessTest("syurk",opt::test::corr::matrix_multiplication_4));
	v.push_back(opt::test::CorrectnessTest("sylrk", opt::test::corr::matrix_multiplication_5));
	v.push_back(opt::test::CorrectnessTest("gemm various dims double cpu)",opt::test::corr::matrix_multiplication_6));
	v.push_back(opt::test::CorrectnessTest("gemm identity matrix times identity matrix various dims double cpu)",opt::test::corr::matrix_multiplication_7));
	
	v.push_back(opt::test::CorrectnessTest("cholseky LDL dim(A)=(3,3)",opt::test::corr::cholesky_1));
	
	v.push_back(opt::test::CorrectnessTest("cholseky LDL various dimensions)",opt::test::corr::cholesky_2));
	
	v.push_back(opt::test::CorrectnessTest("cholseky solve various dimensions)",opt::test::corr::cholesky_solve));

	//GPU Diagnostics
	#ifdef opt_use_cuda
		v.push_back(opt::test::CorrectnessTest("gauss_newton_example3 float gpu",opt::test::corr::gauss_newton_example3_gpu));
		v.push_back(opt::test::CorrectnessTest("gemm dim(A)=(64,64), dim(B)=(64,64) float gpu",opt::test::corr::matrix_multiplication_1_f32_gpu));
		v.push_back(opt::test::CorrectnessTest("gemm various dims float gpu",opt::test::corr::matrix_multiplication_2_f32_gpu));
	#endif
	
	bool result=true;
	
	for (int i=0; i<v.size(); i++){
		bool temp=v[i].run_test();
		if (temp!= true){
			result=temp;
		}
		
		if (v[i].test_successful){
			os<<v[i].name<<" finished successfully\n";
		}
		else{
			os<<v[i].name<<" finished erroneously\n";
		}
		
		if ((i%5)==0){
			os<<i+1<<" out of "<<v.size()<<" correctness tests finished \n";
		}
	}
	if (save_metrics==true){
		//opt::test::corr::save_metrics(v,oss.str());
	}
	
	return result;

}

bool run_tests(std::ostream& os,bool save_stats){

	//bool test1=run_performance_tests(os, save_stats); //run performance tests
	bool test2=run_correctness_tests(os,save_stats); //run correctness tests
	return test2;
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