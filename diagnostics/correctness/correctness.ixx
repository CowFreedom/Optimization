module;
#include<string>
#include <ostream>
#include <fstream>
#include <vector>
#include <filesystem>
export module tests.correctness;
import :utility;
export import :gauss_newton.cpu;
export import :transformation;

//Import GPU code
#ifdef opt_use_cuda
	export import :gauss_newton.gpu;
#endif





import tests;

//import string;
namespace opt{
	namespace test{
		namespace corr{
		
		}
	}
}



	

