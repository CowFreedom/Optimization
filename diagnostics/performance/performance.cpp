module;
#include<string>
#include <ostream>
#include <fstream>
#include <vector>
#include <filesystem>
export module tests.performance;
export import :transformation;

#ifdef opt_use_cuda
	export import :transformation.gpu;
#endif

import tests;

//import string;
namespace opt{
	namespace test{
		namespace perf{
			export bool save_metrics(std::vector<opt::test::PerformanceTest>& values, std::string timestring){
				std::filesystem::path path("performance/logs/");
				std::string output="Results of the performance test from ";
				output+=timestring;
				output+=":\n \n";
				output+="Name\t\t\tTime taken (seconds)\n";
				for (auto& v: values){
					output+=v.name;
					output+="\t\t\t";
					output+=std::to_string(v.time_taken.count());
					output+="\n";
				}
				std::ofstream file_output;	
				std::filesystem::create_directories("performance/logs");
				

				timestring +="_performance_log.txt";
				file_output.open(path/timestring);
				file_output<<output;
				file_output.close();
				return true;
			}
		}
	}
}



	

