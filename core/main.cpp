import optimization.transformation;
import optimization.gpubindings;
import optimization.solvers;
#include <iostream>
#include <vector>
#include<array>
#include <deque>
#include <functional>

template<class T>
typename T::value_type sos(T& storage){
	auto sum=*(storage.begin());
	auto curr=storage.begin()+1;
	for (int i=0;i<storage.size();i++){
		sum+=*curr;
		curr++;
	}
		return sum;
}



template<class C, class T>
void res_circle(C params, T storage){
	typename T::value_type x0=*params;
	typename T::value_type x1=*(params+1);
	*storage=x0-1;
	*(storage+1)=x1+0.3;
}

/*Transposed jacobian*/
template<class C, class T>
void j_t_circle(C x, T storage){
	typename T::value_type x0=*x;
	typename T::value_type x1=*(x+1);
	*storage=1;
	*(storage+1)=0;
	*(storage+2)=0;
	*(storage+3)=1;	
}

template<class C, class T>
void j_t_j_inv_circle(C x, T storage){
	typename T::value_type x0=*x;
	typename T::value_type x1=*(x+1);
	*storage=1;
	*(storage+1)=0;
	*(storage+2)=0;
	*(storage+3)=1;	
}

template<class C, class T>
class Circle{

	public:
	
void res_circle(C params, T storage){
	typename T::value_type x0=*params;
	typename T::value_type x1=*(params+1);
	*storage=x0-1;
	*(storage+1)=x1+0.3;
}
	
void j_t_circle(C x, T storage){
	typename T::value_type x0=*x;
	typename T::value_type x1=*(x+1);
	*storage=1;
	*(storage+1)=0;
	*(storage+2)=0;
	*(storage+3)=1;	
}
	
void j_t_j_inv_circle(C x, T storage){
	typename T::value_type x0=*x;
	typename T::value_type x1=*(x+1);
	*storage=1;
	*(storage+1)=0;
	*(storage+2)=0;
	*(storage+3)=1;	
}	

	int dim=2;
	
};




int main(){
	std::deque<double> v={2.3,-0.3};
	std::array<double,3> v2={1,-0.3,4};
	//std::cout<<sos(v2);
	
	int dim_residual=2;
	//opt::solvers::gns::ResidualPure<std::vector<double>> res(residual,dim_residual);
//	opt::solvers::gns::ResidualPureJI<std::deque<double>> res_ji(residual,j_t_circle,j_t_j_inv_circle,dim_residual);
//	opt::solvers::A a(res);
	
	
//	opt::solvers::GNSCPU<std::deque<double>,void(*)(std::deque<double>::const_iterator, std::deque<double>::iterator),opt::solvers::gns::HasJacInv::Yes> gns(res_circle,j_t_circle,j_t_j_inv_circle,dim_residual,std::cout);
	
	Circle<std::deque<double>::const_iterator, std::deque<double>::iterator> c;
	using std::placeholders::_1;
	using std::placeholders::_2;

	std::function<void(std::deque<double>::const_iterator, std::deque<double>::iterator)> f1=std::bind(&Circle<std::deque<double>::const_iterator, std::deque<double>::iterator>::res_circle,c,_1,_2);
	std::function<void(std::deque<double>::const_iterator, std::deque<double>::iterator)> f2=std::bind(&Circle<std::deque<double>::const_iterator, std::deque<double>::iterator>::j_t_circle,c,_1,_2);
	std::function<void(std::deque<double>::const_iterator, std::deque<double>::iterator)> f3=std::bind(&Circle<std::deque<double>::const_iterator, std::deque<double>::iterator>::j_t_j_inv_circle,c,_1,_2);
	//opt::solvers::GNSCPU<std::deque<double>,void(*)(std::deque<double>::const_iterator, std::deque<double>::iterator),opt::solvers::gns::HasJacInv::Yes> gns(f1,f2,f3,c.dim,std::cout);
	opt::solvers::GNSCPU<std::deque<double>,std::function<void(std::deque<double>::const_iterator, std::deque<double>::iterator)>,opt::solvers::gns::HasJacInv::Yes> gns(f1,f2,f3,c.dim,std::cout);
	
	//opt::solvers::gns::Residual<opt::solvers::gns::ResidualSpec::Pure,std::vector<double>,opt::solvers::gns::HasJacInv::No> res(residual,dim_residual);
	//opt::solvers::GNSCPU gns(res,std::cout);
	
	//<std::vector<double>,opt::solvers::gns::ResidualSpec::Pure,opt::solvers::gns::HasJacInv::No>
	auto result=gns.run(v);
	if (result){
		std::cout<<"Gauss-Newton procedure finished successfully\n";
		std::cout<<"The estimated parameters are:\n{";
		
		for (const auto& x: *result){
			std::cout<<x<<",";
			
		}
		std::cout<<"}\n";
		
	}
	else{
		std::cout<<"Gauss-Newton procedure terminated with error.\n";
	}

	
}