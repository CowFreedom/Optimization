module;
#include <string>
#include <ostream>
export module tests;
//export import tests.performance;

export enum class TestResult{
	Ok,
	Error
};


export class TestInterface{
	public:
	const std::string name;

	bool (*run_test)(std::ostream& os);

	TestInterface(std::string _name, bool (*f)(std::ostream&)): name(_name), run_test(f){
	}
	
};