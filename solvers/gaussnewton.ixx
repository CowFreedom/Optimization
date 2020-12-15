module;
#include <ostream>
#include <optional>
#include <thread>
#include <iostream>
#include <vector>
#include <algorithm>
export module optimization.solvers:gaussnewton;

import optimization.transformation;

namespace opt{
	namespace solvers{
	
		/*Concept that describes which member function a container must
		contain for the algorithms to work.*/
		template<class T>
		concept ValidContainer=requires(T& t) {
		 t.push_back((typename T::value_type)(0.0));
		 t.size();
		};
		
		namespace gns{
			
			export enum class ResidualSpec{
				Pure,
				Data
			};
			
			export enum class HasJacInv{
				Yes,
				No
			};
			
			
				/*
			template<class T>
			std::string print_info<EFloat64>(T vars, T J_T_J_inv, int iteration, typename T::value_type f0x){
				std::ostringstream res;
				res.precision(15);
				size_t m=vars;
				const std::vector<EVar<EFloat64>>& params=vars.get_params();
				const std::vector<std::string>& names=vars.get_names();
				
				std::vector<double> std_devs_inv(m,0.0);
				
				//Calculate variances for each parameter
				for (int i=0;i<m;i++){
					std_devs_inv[i]=1.0/sqrt(J_T_J_inv_scaled[i+i*m].get_v());
				}
				
				res<<std::setw(30)<<std::left<<"Newton iteration: "<<iteration<<"\n";
				res<<std::setw(30)<<std::left<<"Squared Error: "<<squared_error.get_v()<<"\n";
				res<<std::setw(30)<<std::left<<"Parameter"<<"|"<<std::setw(25)<<"Estimate"<<"|"<<std::setw(30)<<"Approx. Standard Error"<<"|"<<std::setw(20)<<"Approx. Correlation Matrix\n";
				for (int i=0;i<m;i++){
					res<<std::setw(30)<<std::left<<names[i]<<"|"<<std::setw(25)<<std::right<<params[i].get_value_as_double()<<"|"<<std::setw(30)<<sqrt(J_T_J_inv_scaled[i+i*m].get_v())<<"|";
					for (int j=0;j<=i;j++){
						res<<"\t"<<std::setw(13)<<std_devs_inv[i]*std_devs_inv[j]*J_T_J_inv_scaled[j+i*m].get_v();
					}
					res<<"\n";
				}
				
				return res.str();
				
			}			

*/			
		}
	
		export template<ValidContainer T,class U,gns::HasJacInv F>
		class GNSCPU{
				GNSCPU(U _r,U _j_t, U _j_t_j_inv,int _xdim, std::ostream& _os){
			
			}
		};

		export template<ValidContainer T,class U>
		class GNSCPU<T,U,gns::HasJacInv::Yes> {
		
			using gfloat=typename T::value_type;
			
			private:
			U r;
			U j_t;
			U j_t_j_inv;
			int rdim;
			gfloat lambda=0.5; //determines, how much the step size is reduced at each iteration of the wolfe conditions
			gfloat tol=0.001;
			int n_threads=1;
			int max_step_iter=60/n_threads; //maximum number of iterations during the stepsize finding process
			int max_iter=300; //maximum number of iterations
			gfloat c1=0.5;
			
			gfloat f0(typename T::const_iterator params,typename T::iterator residuals){
			
				T result(1);
				r(params,residuals);	
				opt::math::cpu::gemm(1,1,rdim,gfloat(1.0),residuals,1,rdim,residuals,1,1,gfloat(0.0),result.begin(),1,1);
				
				return *(result.begin());
			}
			
			void gemm_and_residual(size_t xdim ,gfloat beta,typename T::iterator A, typename T::iterator x_source, typename T::iterator x_dest, typename T::iterator res, gfloat& f0_res){
					T residual(rdim);
				std::copy(x_source,x_source+xdim,x_dest);
					
							/*
					
					
						os<<"x_dest davor\n";
						for (int i=0;i<d;i++){
							os<<*(x_dest+i)<<"\t";
						}
					os<<"\n";
				
					os<<"r_i\n";
						for (int i=0;i<d;i++){
							os<<*(residual.begin()+i)<<"\t";
						}
					os<<"\n";
					
					os<<"M:\n";
					
					for (int i=0;i<d;i++){
						for (int j=0;j<d;j++){
							os<<A[i*d+j]<<"t";
						}
						os<<"\n";
					
					}
					
				
				
					*/
					
					
					opt::math::cpu::gemm(xdim,1,rdim,-beta,A,1,rdim,res,1,1,gfloat(1.0),x_dest,1,1);	
					f0_res=f0(x_dest,residual.begin());		
					
				
				/*
				
					
					os<<"r_i+beta*(J_T_J)^-1*J_T*r_i\n";
						for (int i=0;i<d;i++){
							os<<*(x_dest+i)<<"\t";
						}
					os<<"\n";
						std::cin.get();					
					*/
			}
			
			std::ostream& os;
			
			//backtracking line search
			bool bls(gfloat fnew, gfloat fmin, gfloat beta, typename T::iterator grad, typename T::iterator xb,typename T::iterator xe,typename T::iterator xpb, int xdim){
		//	std::cout<<"BLS: beta: "<<beta<<"\n";
				T direction(xdim);
				std::transform(xb,xe,xpb, direction.begin(),std::minus<double>()); //not numerically stable but faster than doing the matrix calculations again
			/*			
				std::cout<<"xb:\n";
				for (int i=0;i<xdim;i++){
				std::cout<<*(xb+i)<<"\t";
				}
				
				std::cout<<"\nxp:\n";
				for (int i=0;i<xdim;i++){
				std::cout<<*(xpb+i)<<"\t";
				}
				
				std::cout<<"\n grad:\n";
				for (int i=0;i<xdim;i++){
				std::cout<<*(grad+i)<<"\t";
				}*/			
				
				T fk(1,fmin);
				opt::math::cpu::gemm(1,1,xdim,c1,direction.begin(),1,xdim,grad,1,1,gfloat(1.0),fk.begin(),1,1);	
				//std::cout<<"f new: "<<fnew <<" f(xk)"<<*(fk.begin())<<"\n";
				if (fnew<=*(fk.begin())){
					return true;
				}
				else{
					return false;
				}
			}
			
			public:
			
			GNSCPU(U _r,U _j_t, U _j_t_j_inv,int _rdim, std::ostream& _os): r(_r), j_t(_j_t), j_t_j_inv(_j_t_j_inv),rdim(_rdim),os(_os){
				//os<<"Hat alles geklappt, alter\n";
			}
			
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Gauss-Newton procedure.
			*/
			std::optional<T> run(T x0){
				
				bool run_finished=false;
				int xdim=x0.size();
				//Test is parameters already minimize f0 according to tol
				T residuals(rdim);
				gfloat fmin=f0(x0.begin(),residuals.begin()); //current minimum
				
				if (fmin<tol){
					return {x0};
				}
				else{
					T J_t(xdim*rdim);
					T grad(xdim);
					T J_t_J_inv(xdim*xdim);

					T C(xdim*rdim);
					T xi(x0.begin(),x0.end());
					T xs(xdim*n_threads);
					gfloat alpha=0.7;		
					int iter=0;
					
					std::vector<std::thread> ts(n_threads);
					gfloat curr_min;
					
					while (run_finished==false && (iter<max_iter)){
						j_t(xi.begin(),J_t.begin());
						j_t_j_inv(xi.begin(),J_t_J_inv.begin());
					
						opt::math::cpu::gemm(xdim,rdim,xdim,gfloat(1.0),J_t_J_inv.begin(),1,xdim,J_t.begin(),1,rdim,gfloat(0.0),C.begin(),1,rdim);
						
						/*Calculate gradient from indices of jacobi matrix. Used for backtracking stepsize finding later*/
						for (int i=0;i<xdim;i++){
							gfloat sum=0;
							for (int j=0;j<rdim;j++){
								
								sum+=*(J_t.begin()+j+i*rdim)*gfloat(2.0)* *(residuals.begin()+j);
							}
							*(grad.begin()+i)=sum;
						}
			//			std::cout<<"current x:\n";
						
						gfloat beta=alpha;
						
						std::vector<gfloat> f0_vals(n_threads);
						std::vector<gfloat> betas(n_threads);
						int step_iter=0;
						bool step_size_found=false;
						
								
						while(!step_size_found && step_iter<=max_step_iter){
						step_iter++;
							
							for (int i=0;i<n_threads;i++){
						//	std::cout<<"beta:"<<beta<<"lambda:"<<lambda<<"\t \t";
								ts[i]=std::thread(&GNSCPU::gemm_and_residual,this,xdim,beta,C.begin(),xi.begin(),xs.begin()+i*xdim,residuals.begin(),std::ref(f0_vals[i]));
								betas[i]=beta;
								beta*=lambda;
								
							}
							
							/*
							os<<"(J_T_J)^-1\n";
							
							for (int i=0;i<d;i++){
								for (int k=0; k<d;k++){
									os<<*(J_t_J_inv.begin()+d*i+k)<<"\t";
								
								}
								os<<"\n";
							}
						
						*/
						
							for (auto& t: ts){
								t.join();
							}
					
							//Find step size that minimizes f0 the most						
							curr_min=f0_vals[0];
							int min_index=0;
							for (int i=1;i<n_threads;i++){
								if (f0_vals[i]<curr_min){
									curr_min=f0_vals[i];
									min_index=i;
								}
							}		
						//	std::cout<<"curr_min:"<<curr_min<<" fmin:"<<fmin<<"\n";
						//	std::cout<<"min index:"<<min_index<<"\n";
				
							if (bls(curr_min,fmin,betas[min_index],grad.begin(),xs.begin()+min_index,xs.begin()+min_index+xdim,xi.begin(),xdim)){
								fmin=curr_min;
								std::copy(xs.begin()+min_index,xs.begin()+min_index+xdim,xi.begin());
								r(xi.begin(),residuals.begin());
							//	std::cout<<"Current error: "<<fmin<<"\n";
								step_size_found=true;
							}
										
						}
														
						if (!step_size_found){
							return {};
						}
						
						if (fmin<=tol){
							run_finished=true;
							return {xi};
						}
				
						iter++;

					}
					
				}
				return {};
				
			}

		};
		
		export template<ValidContainer T,class U>
		class GNSCPU<T,U,gns::HasJacInv::No> {
			
			using gfloat=typename T::value_type;
			
			private:
			U r; //represents residual function
			U j; //represents jacobian function
			U j_inv; //represents scheme to calculate jacobian inverse, like finite differences 
			int rdim;
			int xdim;

			std::ostream& os;
			
			//Evaluates sum of squares objective function f0
			gfloat f0(typename T::const_iterator params,typename T::iterator residuals){
				gfloat result=0.0;
				r(params,residuals);
				
				for (int i=0;i<rdim;i++){
					result+=residuals[i] * residuals[i];
				}

				return result;
			}
			
						//backtracking line search
			bool bls(gfloat fnew, gfloat fcurrent, gfloat beta, typename T::iterator grad, typename T::iterator direction, int xdim){
			
				T fk(1,fcurrent);
				std::cout<<"F current:"<<fcurrent<<"\n";
				std::cout<<"d1:"<<direction[0]<<" d2:"<<direction[1]<<"\n";
				std::cout<<"c1:"<<c1<<"beta"<<beta<<"\n";
				std::cout<<"grad1:"<<grad[0]<<"grad2:"<<grad[1]<<"\n";
				opt::math::cpu::dgmv(1,xdim,-c1*beta,direction,1,xdim,grad,1,gfloat(1.0),fk.begin(),1);	
				std::cout<<"fnew:"<<fnew<<"f(xi)+gradf"<<fk[0]<<"\n";
				std::cin.get();
				if (fnew<=fk[0]){
					return true;
				}
				else{
					return false;
				}
			}
			
			
			public:
			gfloat lambda=0.5; //determines, how much the step size is reduced at each iteration of the wolfe conditions
			int n_threads=std::thread::hardware_concurrency();
			int max_step_iter=60/n_threads; //maximum number of iterations during the stepsize finding process
			int max_iter=300; //maximum number of iterations
			gfloat c1=0.5;
			gfloat alpha=0.7;	//decrease factor in the newton stepsize iterations	
			gfloat tol=0.001;
			
			GNSCPU(U _r,U _j,int _xdim, int _rdim, std::ostream& _os): r(_r), j(_j),rdim(_rdim),xdim(_xdim),os(_os){
				
			}		
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Gauss-Newton procedure.
			*/
			std::optional<T> run(T x0){
				
				bool run_finished=false;
				if (x0.size()<xdim){
					os<<"Size of the input container does not fit the dimension of the problem";
					return {};
				}
				//Test is parameters already minimize f0 according to tol
				T residuals(rdim*n_threads); //collection of residuals for each thread
				int c_r=0; //current best residual
				gfloat fmin=f0(x0.begin(),residuals.begin()+c_r); //current minimum
				std::cout<<"Start newton"<<fmin<<"\n";
				if (fmin<tol){
					return {x0};
				}
				else{
					T J(xdim*rdim);
					T grad(xdim);
					T b(xdim); //output vector in J^{T}*J*v=b
					T v(xdim);//unknown vector in J^{T}*J*v=b
					T J_t_J(xdim*xdim);
					T xi(x0.begin(),x0.end()); //current parameters
					int iter=0;
					
					std::vector<std::thread> ts(n_threads);
					gfloat curr_min;
					
					do{
					
						j(xi.begin(),J.begin()); //get current value of Jacobian
						
						/*Calculate gradient from indices of jacobi matrix. Used for backtracking stepsize finding later*/
						//used for sum calculation. not necessary but probably faster due cache misses in the traversal of array
						for(int i=0;i<xdim;i++){
							grad[i]=0.0;
						}
						for (int i=0;i<rdim;i++){
							gfloat sum=0.0;
							for (int j=0;j<xdim;j++){
								grad[j]+=J[i*xdim+j]*gfloat(2.0)* residuals[i];
							}
						}
						
						
						opt::math::cpu::gemm(xdim,xdim,rdim,gfloat(1.0),J.begin(),xdim,1,J.begin(),1,xdim,gfloat(0.0),J_t_J.begin(),1,xdim); //calculate J^{T}*J
						opt::math::cpu::dgmv(xdim,rdim,gfloat(1.0),J.begin(),xdim,1,residuals.begin()+c_r,1,gfloat(0.0),b.begin(),1); //calculate vector for J^{T}*J*v=b
						opt::math::cpu::choi<typename T::iterator, gfloat>(xdim, J_t_J.begin(), 1, xdim); //calculate cholesky diagonal A=LDL^{T} decomposition
						opt::math::cpu::choi_solve<typename T::iterator,typename T::iterator,typename T::iterator,gfloat>(xdim, J_t_J.begin(), 1, xdim, b.begin(), 1, v.begin(), 1);
						gfloat beta=alpha;
						
						std::vector<gfloat> f0_vals(n_threads);
						std::vector<gfloat> betas(n_threads);
						int step_iter=0;
						bool step_size_found=false;
						
						//maybe take U by reference
						auto eval=[](U r, typename T::iterator residual, int rdim, gfloat beta, typename T::iterator v, typename T::iterator xi, int xdim, gfloat alpha, gfloat& f0_val){
							T _x(xdim); //new parameters
							
							for (int i=0;i<rdim;i++){
								_x[i]=gfloat(-1.0)*beta*v[i]+xi[i]; //get new parameters
							}
							f0_val=0.0;
							r(_x.begin(),residual); //evaluate objective function
							gfloat temp=0.0;
							for (int i=0;i<rdim;i++){
								temp+=residual[i] * residual[i];
							}
							f0_val=temp;
							
						};
						
						do{
							
							
							for (int i=0;i<n_threads;i++){
								ts[i]=std::thread(eval,r,residuals.begin()+i*rdim,rdim,beta, v.begin(),xi.begin(), xdim, alpha,std::ref(f0_vals[i]));
						
								betas[i]=beta;
								beta*=lambda;
							}
							for (auto& t: ts){
								t.join();
							}
							
							//Find step size that minimizes f0 the most						
							curr_min=f0_vals[0];
							int min_index=0;
							for (int i=1;i<n_threads;i++){
								if (f0_vals[i]<curr_min){
									curr_min=f0_vals[i];
									min_index=i;
								}
							}
							
							std::cout<<"Current parameters\n";
							for (int i=0;i<xdim;i++){
								std::cout<<xi[i]<<"\n";
							}
							
							//recover the stepsize factor that minimized f0 the most in this iteration, betaopt=alpha*lambda^{step_iter*n_threads+min_index}
							gfloat betaopt=alpha;
							for (int i=1;i<step_iter*n_threads+min_index;i++){
								betaopt*=lambda;
							}
							
							if (bls(curr_min, fmin, gfloat(1.0)/betaopt, grad.begin(), v.begin(),xdim)){
								std::cin.get();
								c_r=min_index*rdim;
								fmin=curr_min;
								std::cin.get();
								
								for (int i=0;i<rdim;i++){
									xi[i]=gfloat(-1.0)*betaopt*v[i]+xi[i]; //get new parameters
								}
								
								std::cout<<"Best parameters\n";
								for (int i=0;i<xdim;i++){
									std::cout<<xi[i]<<"\n";
								}
								
							//	std::cout<<"Current error: "<<fmin<<"\n";
								step_size_found=true;
							}
							step_iter++;
							
						}
						while(!step_size_found && step_iter<=max_step_iter);
														
						if (!step_size_found){
							return {};
						}
						
						if (fmin<=tol){
						std::cout<<"fmin:"<<fmin<<"tol:"<<tol<<"\n";
							run_finished=true;
							return {xi};
						}
						else{
							std::cout<<"fmin nicht kleiner tol\n";
						}
				
						iter++;
					}
					while (run_finished==false && (iter<max_iter));
					
				}
				return {};
				
			}


		
		};


		
	}
}

