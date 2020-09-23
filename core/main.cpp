import optimization.transformation;
import gpubindings;
import optimization.solvers;
#include <iostream>
#include <vector>

template<class T>
typename T::value_type sos(T begin, int n){
	auto sum=*begin;
	auto curr=begin+1;
	for (int i=0;i<n;i++){
		sum+=*curr;
		curr++;
	}
		return sum;
}





int main(){
	std::vector<double> v={1,-0.3,3};
	std::cout<<sos(v.begin(),v.size());
	opt::solvers::gns::ResidualPure<std::vector<double>::iterator> res(sos);
	opt::solvers::GNSCPU gns(res);
	
	
}