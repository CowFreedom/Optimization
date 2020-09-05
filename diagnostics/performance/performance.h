export module Tests:Performance;
import string;


export class PerformanceTest{
	public:
	const std::string name;
	
	bool (&run_test)();
	PerformanceTest(std::string _name, bool (&f)()): name(_name), run_test(f){
	}
	
};
