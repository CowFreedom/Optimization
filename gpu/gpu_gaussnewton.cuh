extern "C"{

	//pointer to function object, see PIMPL pattern http://www.olivierlanglois.net/idioms_for_using_cpp_in_c_programs.html
	struct EvalGNSGPU{
	
	};
	
	void EvalGNSGPU_eval(EvalGNSGPU* r, const float* x, float* storage);

	void EvalGNSGPU_destroy(EvalGNSGPU* r);

}



