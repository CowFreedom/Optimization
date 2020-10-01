module;
#include <vector>
#include <random>
#include <ostream>
#include<cmath>
#include <functional>
#include <iostream>
#include <thread> //warum brauche ich das? prüfen!
#include <array>
export module tests.correctness:gauss_newton.cpu;

import tests;

import optimization.transformation;
import optimization.solvers;

namespace opt{
	namespace test{

		namespace corr{
			
			template<class C, class T>
			class Circle{
			
				public:
				Circle(int x, int y){
					offset[0]=x;
					offset[1]=y;
				}
				std::array<double,2> offset={0,0};
				void res_circle(C params, T storage){
					typename T::value_type x0=*params;
					typename T::value_type x1=*(params+1);
					*storage=x0-offset[0];
					*(storage+1)=x1-offset[1];
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
		

			export bool gauss_newton_example1(std::ostream& os, CorrectnessTest& v){
				std::vector<double> x0={2.3,-0.3};
				
				Circle<std::vector<double>::const_iterator, std::vector<double>::iterator> c(-4,2.32);
				using std::placeholders::_1;
				using std::placeholders::_2;

				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f1=std::bind(&Circle<std::vector<double>::const_iterator, std::vector<double>::iterator>::res_circle,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f2=std::bind(&Circle<std::vector<double>::const_iterator, std::vector<double>::iterator>::j_t_circle,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f3=std::bind(&Circle<std::vector<double>::const_iterator, std::vector<double>::iterator>::j_t_j_inv_circle,c,_1,_2);

				opt::solvers::GNSCPU<std::vector<double>,std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)>,opt::solvers::gns::HasJacInv::Yes> gns(f1,f2,f3,c.dim,std::cout);
				
				auto result=gns.run(x0);
				if (result){
					
					for (int i=0;i<x0.size();i++){
						if (std::abs((*result)[i]-c.offset[i])>0.02){
							return false;						
						}
								
					}
					v.test_successful=true;
					return true;
					
				}
				else{
					return false;
				}				
			}
		}
	}

}

