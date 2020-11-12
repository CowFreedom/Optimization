module;
#include <string>
#include <ostream>
#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
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
		
		export class CorrectnessTest:public TestInterface{
			private:
			bool (*test_function)(std::ostream& os, CorrectnessTest& v);

			public:

			CorrectnessTest(std::string _name, bool (*f)(std::ostream&, CorrectnessTest& v)): TestInterface(_name,nullptr), test_function(f){	
			}	
			bool test_successful=false;
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
		
		
		/*Merges files in a given path. Merging is done sequentially in order of the input files.
		in: path of the folder with files to be merged

		fix: fixed columnnames
		out: Container object that implements push_back
		delimiter: delimiting string
		ParseType: Type to be parsed*/
		export template<class T>
		bool parse_csv(std::string filepath, T out, int length, std::string delimiter) {
			std::ifstream file(filepath);
			if (file.fail()) {
				std::cerr << "Couldn't open  " << filepath << "\n";
				return false;
			}
			
			std::string line;
			//std::array<char,64> numbuf;
			char numbuf[64];
			bool in_digit = 0;
			
			int curr=0;

			while (std::getline(file, line)) {
				//Skip if line starts with a comment #
				if (line.front() == '#') {
					continue;
				}
				size_t bufsize = 0;
				line.push_back('\n'); //add last delimiter so that number parsing is correct
				for (auto& x : line) {
					if (in_digit) {
						if (isdigit(x) || x == '.' || x == 'e' || x == '-' || x == '+') {
							numbuf[bufsize] = x;
							bufsize++;
						}
						else {
							in_digit = false;
							numbuf[bufsize] = '\0';
							out[curr]=std::atof(numbuf);
							curr++;
							bufsize = 0;
						}
					}
					else {
						if (isdigit(x) || x == '.' || x == '-' || x == '+') {
							in_digit = true;
							numbuf[bufsize] = x;
							bufsize++;
						}
					}
				}

			}
			return true;
		}

	
	}


}

