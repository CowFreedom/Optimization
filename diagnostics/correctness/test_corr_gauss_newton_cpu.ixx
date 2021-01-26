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

import :utility;

import tests;

import optimization.transformation;
import optimization.solvers;

namespace opt{
	namespace test{

		namespace corr{		

			//Tests of gaussnewton cpu
			//CHANGE
			export bool gauss_newton_example2(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				std::random_device os_seed;
				const uint32_t seed=os_seed();
				std::mt19937 rng(seed);
				std::uniform_real_distribution<double> dist(-10,10);

				v.test_successful=true;
				return true;
				
			}
			
			export bool gauss_newton_example3(std::ostream& os, CorrectnessTest& v){
				std::vector<double> x0={-0.3,-0.3};
				double tol=1e-10;
				double x1=5;
				double x2=-90;
				CircleB<std::vector<double>::const_iterator, std::vector<double>::iterator,double> c(x1,x2);
				using std::placeholders::_1;
				using std::placeholders::_2;
				
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f1=std::bind(&CircleB<std::vector<double>::const_iterator, std::vector<double>::iterator,double>::residual,c,_1,_2);
				std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)> f2=std::bind(&CircleB<std::vector<double>::const_iterator, std::vector<double>::iterator,double>::jacobian,c,_1,_2);
				
				opt::solvers::GNSCPU<std::vector<double>,std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)>> gns(f1,f2,c.xdim,c.rdim,std::cout.rdbuf());
				gns.tol=tol;
				auto result=gns.run(x0);
				
				if (result){
					float p1=(*result)[0];
					float p2=(*result)[1];
					
					float val=(p1-x1>=0.0)?p1-x1:p1+x1;
				//	std::cout<<"Endsum:"<<(*result)[0]<<" and"<<(*result)[1]<<"\n";
					if(val>0.001){
							return false;						
					}
					val=(p2-x2>=0.0)?p2-x2:p2+x2;
					if(val>0.001){
							return false;						
					}
					
					v.test_successful=true;
					return true;
					
				}
				else{
					return false;
				}	
			}	
			
			export bool gauss_newton_example4(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				std::random_device os_seed;
				const uint32_t seed=os_seed();
				std::mt19937 rng(seed);
				std::uniform_real_distribution<double> dist(-10,10);
				
				for (int i=0;i<reps;i++){
					std::vector<double> x0={dist(rng),dist(rng)};
					double tol=1e-1; //Higher values lead to instablity
					double a=1;
					double b=1;

					using std::placeholders::_1;
					using std::placeholders::_2;
					Convex1<std::vector<double>::const_iterator, std::vector<double>::iterator,double> c1(a,b);
	
					auto f1=std::bind(&Convex1<std::vector<double>::const_iterator, std::vector<double>::iterator,double>::residual,c1,_1,_2);
					auto f2=std::bind(&Convex1<std::vector<double>::const_iterator, std::vector<double>::iterator,double>::jacobian,c1,_1,_2);
					
					opt::solvers::GNSCPU<std::vector<double>,std::function<void(std::vector<double>::const_iterator, std::vector<double>::iterator)>> gns(f1,f2,c1.xdim,c1.rdim,std::cout.rdbuf());
					gns.tol=0.1; //lower values lead to instability in the approximated hesse matrix calculations
					auto result=gns.run(x0);
			
					if (result){
						double x=(*result)[0];
						double y=(*result)[1];
						double val_f=a*a*x*x+a*b*x*y+b*b*y*y;
						double val_dfx=2*a*a*x+a*b*y;
						double val_dfy=2*b*b*y+a*b*x;
						if(val_f<tol || (val_dfx<tol && val_dfy<tol)){
						}
						else{
							//std::cout<<"val_f:"<<val_f<<" val_dfx: "<<val_dfx<<" val_dfy: "<<val_dfy<<"\n";
							return false;
						}
					}
					else{
						
						return false;
					}		
				}	
				v.test_successful=true;
				return true;
				
			}
			
		
			
		}
	
	}

}

