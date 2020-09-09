#include <chrono>

namespace opt{

namespace test{
template<class T>
void f(){
	auto stop = std::chrono::high_resolution_clock::now(); 

	//std::cout<<std::chrono::duration_cast<std::chrono::seconds>(stop);

}

}



}


int main(){
opt::test::f<double>();
return 0;

}