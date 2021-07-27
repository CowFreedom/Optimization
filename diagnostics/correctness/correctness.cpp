module;
#include<string>
#include <ostream>
#include <fstream>
#include <vector>
#include <filesystem>
//#include <iostream>
export module tests.correctness;

import :utility;
export import :gauss_newton.cpu;
export import :transformation;

//Import GPU code
#ifdef opt_use_cuda
	export import :gauss_newton.gpu;
	export import :transformation.gpu;
#endif



