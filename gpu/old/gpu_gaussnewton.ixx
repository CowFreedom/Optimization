module;
#include <vector>
export module optimization.solvers:gpu_gaussnewton;

import optimization.transformation;

import :gaussnewton;


extern "C"{



	struct ResidualGPU{
		void* res; //pointer to Residual class, see PIMPL pattern http://www.olivierlanglois.net/idioms_for_using_cpp_in_c_programs.html
	} ;

	void ResidualGPU_get_residual(ResidualGPU* a, float* x,float* residual){

	//	(static_cast<opt::solvers::gns::ResidualPure<std::vector<float>>*>(a->res))->r(x,residual);
	}

}



	namespace solvers{
		namespace gpu{
		//	using ResidualGPU=ResidualGPU_f;
		
		}	
	}