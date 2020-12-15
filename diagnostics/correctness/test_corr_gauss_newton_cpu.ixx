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
				std::array<typename T::value_type,2> offset={0,0};
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
				int x=-4;
				int y=2.32;
				Circle<std::vector<double>::const_iterator, std::vector<double>::iterator> c(x,y);
				using std::placeholders::_1;
				using std::placeholders::_2;

				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f1=std::bind(&Circle<std::vector<double>::const_iterator, std::vector<double>::iterator>::res_circle,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f2=std::bind(&Circle<std::vector<double>::const_iterator, std::vector<double>::iterator>::j_t_circle,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f3=std::bind(&Circle<std::vector<double>::const_iterator, std::vector<double>::iterator>::j_t_j_inv_circle,c,_1,_2);

				opt::solvers::GNSCPU<std::vector<double>,std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)>,opt::solvers::gns::HasJacInv::Yes> gns(f1,f2,f3,c.dim,std::cout);
				
				double tol=0.001;
				auto result=gns.run(x0);
				if (result){
					double sum=((*result)[0]-x)*((*result)[0]-x)+((*result)[1]-y)*((*result)[1]-y);
					if(sum>tol){
							return false;						
					}
					else{
			
						v.test_successful=true;
						return true;
					}
				}	
				return false;
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

			export bool gauss_newton_example1_fptr(std::ostream& os, CorrectnessTest& v){
				std::vector<double> x0={2.3,-0.3};
				
				opt::solvers::GNSCPU<std::vector<double>,void(*)(std::vector<double>::const_iterator, std::vector<double>::iterator),opt::solvers::gns::HasJacInv::Yes> gns(res_circle,j_t_circle,j_t_j_inv_circle,2,std::cout);
				double tol=0.001;
				auto result=gns.run(x0);
				if (result){
					double sum=((*result)[0]-1)*((*result)[0]-1)+((*result)[1]+0.3)*((*result)[1]+0.3);
					if(sum>tol){
							return false;						
					}
					else{
			
						v.test_successful=true;
						return true;
					}
					
				}
				else{
					return false;
				}				
				
			}
			
			template<class C, class T>
			class Example2{
			
				public:
				Example2(){
				}
				std::array<double,2> offset={0,0};
				void residual(C params, T storage){
					typename T::value_type x0=*params;
					typename T::value_type x1=*(params+1);
					*storage=std::exp(x0*x1);
					*(storage+1)=1/x0;
					*(storage+2)=1/x1;
				}
				
				void j_t(C x, T storage){
					typename T::value_type x0=*x;
					typename T::value_type x1=*(x+1);
					*storage=x0*std::exp(x0*x1);
					*(storage+1)=-1/(x0*x0);
					*(storage+2)=0;
					*(storage+3)=x1*std::exp(x0*x1);
					*(storage+4)=0;
					*(storage+5)=-1/(x1*x1);
				}
				
				//inverse, see https://www.wolframalpha.com/input/?i=inverse+%7B%7Bx%5E%28-4%29+%2B+E%5E%282+x+y%29+x%5E2%2C+E%5E%282+x+y%29+x+y%7D%2C+%7BE%5E%282+x+y%29+x+y%2C+y%5E%28-4%29+%2B+E%5E%282+x+y%29+y%5E2%7D%7D
				void j_t_j_inv(C x, T storage){
					typename T::value_type x0=*x;
					typename T::value_type x1=*(x+1);
					
					typename T::value_type den=1.0/(std::exp(2*x0*x1)*(x0*x0*x0*x0*x0*x0+x1*x1*x1*x1*x1*x1+1));
					
					*storage=den*(x0*x0*x0*x0*x1*x1*x1*x1*x1*x1*std::exp(2*x0*x1)+x0*x0*x0*x0);
					*(storage+1)=den*(x0*x0*x0*x0*x0*x1*x1*x1*x1*x1*(-std::exp(2.0*x0*x1)));
					*(storage+2)=den*(x0*x0*x0*x0*x0*x1*x1*x1*x1*x1*(-std::exp(2.0*x0*x1)));
					*(storage+3)=den*(x0*x0*x0*x0*x0*x0*x1*x1*x1*x1*std::exp(2.0*x0*x1)+x1*x1*x1*x1);	
				}	

				int rdim=3; //dimension of residual
				
			};
			
			export bool gauss_newton_example2(std::ostream& os, CorrectnessTest& v){
				std::vector<double> x0={-0.3,-0.3};
				Example2<std::vector<double>::const_iterator, std::vector<double>::iterator> c;
				using std::placeholders::_1;
				using std::placeholders::_2;
				
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f1=std::bind(&Example2<std::vector<double>::const_iterator, std::vector<double>::iterator>::residual,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f2=std::bind(&Example2<std::vector<double>::const_iterator, std::vector<double>::iterator>::j_t,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f3=std::bind(&Example2<std::vector<double>::const_iterator, std::vector<double>::iterator>::j_t_j_inv,c,_1,_2);

				opt::solvers::GNSCPU<std::vector<double>,std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)>,opt::solvers::gns::HasJacInv::Yes> gns(f1,f2,f3,c.rdim,std::cout);
				auto result=gns.run(x0);
				if (result){
					return false;
					
				}
				else{
					v.test_successful=true;
					return true;
				}
			}
			
			//Tests of gaussnewton cpu
			
			template<class C, class T>
			class CircleB{
			
				public:
				CircleB(int x, int y){
					offset[0]=x;
					offset[1]=y;
				}
				std::array<typename T::value_type,2> offset={0,0};
				void residual(C params, T storage){
					typename T::value_type x0=*params;
					typename T::value_type x1=*(params+1);
					*storage=x0-offset[0];
					*(storage+1)=x1-offset[1];
				}

				void jacobian(C x, T storage){
					typename T::value_type x0=*x;
					typename T::value_type x1=*(x+1);
					*storage=1;
					*(storage+1)=0;
					*(storage+2)=0;
					*(storage+3)=1;	
				}

				const int rdim=2;
				const int xdim=2;
				
			};
			
			export bool gauss_newton_example3(std::ostream& os, CorrectnessTest& v){
				std::vector<double> x0={-0.3,-0.3};
				double tol=1e-10;
				double x1=5;
				double x2=-90;
				CircleB<std::vector<double>::const_iterator, std::vector<double>::iterator> c(x1,x2);
				using std::placeholders::_1;
				using std::placeholders::_2;
				
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f1=std::bind(&CircleB<std::vector<double>::const_iterator, std::vector<double>::iterator>::residual,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f2=std::bind(&CircleB<std::vector<double>::const_iterator, std::vector<double>::iterator>::jacobian,c,_1,_2);
				
				opt::solvers::GNSCPU<std::vector<double>,std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)>,opt::solvers::gns::HasJacInv::No> gns(f1,f2,c.xdim,c.rdim,std::cout);
				gns.tol=tol;
				auto result=gns.run(x0);
				
				if (result){
					double sum=((*result)[0]-x1)*((*result)[0]-x1)+((*result)[1]-x2)*((*result)[1]-x2);
					std::cout<<"Endsum:"<<(*result)[0]<<" and"<<(*result)[1]<<"\n";
					if(sum>tol){
							return false;						
					}
					else{
			
						v.test_successful=true;
						return true;
					}
					
				}
				else{
					return false;
				}
				
			}			
			
		}
	
	}

}

