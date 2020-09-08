import tests.performance;
#include <iostream>

bool test1(std::ostream& os){
	os<<"All good\n";
	return true;
}

int main(){
	//int num=mul();
	//std::cout << "Num ist: "<<num<<"\n";
	PerformanceTest a("Test 1",test1);
	std::cout<<a.run_test(std::cout);


}