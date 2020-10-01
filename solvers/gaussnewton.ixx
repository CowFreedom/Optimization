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
			
			export template<ResidualSpec K,ValidContainer T, HasJacInv>
			class Residual{
		
			};
			
			export template<ValidContainer T>
			class Residual<ResidualSpec::Pure,T,HasJacInv::No>{
			
			
				
				public:
				Residual(void (&_r)(typename T::const_iterator params, typename T::iterator storage), int _dim):r(_r),dim(_dim){
				
				}
				
				void (&r)(typename T::const_iterator params, typename T::iterator storage); //the container contains a vector of residuals.
				
				const int dim;
			
			};
			
			template<ValidContainer T>
			class Residual<ResidualSpec::Data,T,HasJacInv::No>{
			
				T v;
				
				public:
				Residual(void (&_r)(typename T::const_iterator params, typename T::iterator storage),T _v, int _dim):r(_r),v(_v), dim(_dim){
				
				}
			
				void (&r)(typename T::const_iterator params,typename T::iterator storage); //the container contains a vector of residuals.
				
				
				const int dim;
			
			};
			
			
			template<ValidContainer T>
			class Residual<ResidualSpec::Pure,T,HasJacInv::Yes>{
			
				T v;
				public:
				Residual(void (&_r)(typename T::const_iterator params, typename T::iterator storage),void (&_j_t)(typename T::const_iterator, typename T::iterator),void (&_j_t_j_inv)(typename T::const_iterator, typename T::iterator),int _dim):r(_r),j_t(_j_t),j_t_j_inv(_j_t_j_inv),dim(_dim){
				
				}
			
				void (&r)(typename T::const_iterator params,typename T::iterator storage); //the container contains a vector of residuals.
				
				//jacobi matrix
				void (&j_t)(typename T::const_iterator, typename T::iterator);
				
				void (&j_t_j_inv)(typename T::const_iterator, typename T::iterator);
				
				const int dim;
			
			};
			template<ValidContainer T>
			class Residual<ResidualSpec::Data,T,HasJacInv::Yes>{
			
				T v;
				public:
				Residual(void (&_r)(typename T::const_iterator params, typename T::iterator storage),void (&_j_t)(typename T::const_iterator, typename T::iterator),void (&_j_t_j_inv)(typename T::const_iterator, typename T::iterator),T _v, int _dim):r(_r),v(_v), j_t(_j_t),j_t_j_inv(_j_t_j_inv),dim(_dim){
				
				}
			
				void (&r)(typename T::const_iterator params,typename T::iterator storage); //the container contains a vector of residuals.
				
				//jacobi matrix
				void (&j_t)(typename T::const_iterator, typename T::iterator);
				
				void (&j_t_j_inv)(typename T::const_iterator, typename T::iterator);
				
				const int dim;
			
			};
			
			export template<ValidContainer T>
			using ResidualPure=Residual<ResidualSpec::Pure,T,HasJacInv::No>;

			export template<ValidContainer T>
			using ResidualData=Residual<ResidualSpec::Data,T,HasJacInv::No>;		

			export template<ValidContainer T>
			using ResidualPureJI=Residual<ResidualSpec::Pure,T,HasJacInv::Yes>;
			
			export template<ValidContainer T>
			using ResidualDataJI=Residual<ResidualSpec::Data,T,HasJacInv::Yes>;
			
			
			
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
	
		export template<ValidContainer T,gns::ResidualSpec K,gns::HasJacInv F>
		class GNSCPU{
			GNSCPU(gns::Residual<K,T,F>& _r,std::ostream& _os){
			
			}
		};
		
		export template<ValidContainer T,gns::ResidualSpec K, gns::HasJacInv F>
		class A
		{
		public:
			A(gns::Residual<K,T,F>& r){
				std::cout<<"A standard \n";
			}
		};
		
		
		export template<ValidContainer T,gns::ResidualSpec K>
		class A<T,K,gns::HasJacInv::Yes>
		{
		public:
			A(gns::Residual<K,T, gns::HasJacInv::Yes>& r){
				std::cout<<"A yes \n";
			}
		
		
		};
		
		export template<ValidContainer T,gns::ResidualSpec K>
		class A<T,K,gns::HasJacInv::No>
		{
		private:
			gns::Residual<K,T,gns::HasJacInv::No>& r;
	
		public:

			A(gns::Residual<K,T, gns::HasJacInv::No>& _r):r(_r){
				std::cout<<"A no \n";
			}
		
		
		};

		export template<ValidContainer T,gns::ResidualSpec K>
		class GNSCPU<T,K,gns::HasJacInv::Yes> {
		
			using gfloat=typename T::value_type;
			
			private:
			gns::Residual<K,T,gns::HasJacInv::Yes>& r;
			gfloat lambda=0.5; //determines, how much the step size is reduced at each iteration of the wolfe conditions
			gfloat tol=0.001;
			int max_step_iter=2; //maximum number of iterations during the stepsize finding process
			gfloat c1=0.5;
			gfloat f0(typename T::const_iterator params,typename T::iterator residuals){
			
				T result(1);
				r.r(params,residuals);	
				opt::math::cpu::dgemm_nn(1,1,r.dim,gfloat(1.0),residuals,1,r.dim,residuals,1,1,gfloat(0.0),result.begin(),1,1);
				
				return *(result.begin());
			}
			
			void dgemm_and_residual(size_t d ,gfloat beta,typename T::iterator A, typename T::iterator x_source, typename T::iterator x_dest, typename T::iterator res, gfloat& f0_res){
					T residual(r.dim);
				std::copy(x_source,x_source+d,x_dest);
					
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
					
					
					opt::math::cpu::dgemm_nn(d,1,r.dim,-beta,A,1,r.dim,res,1,1,gfloat(1.0),x_dest,1,1);	
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
			bool bls(gfloat fnew, gfloat fmin, gfloat beta, typename T::iterator grad, typename T::iterator xb,typename T::iterator xe,typename T::iterator xpb, int d){
		//	std::cout<<"BLS: beta: "<<beta<<"\n";
				T direction(d);
				std::transform(xb,xe,xpb, direction.begin(),std::minus<double>()); //not numerically stable but faster than doing the matrix calculations again
			/*			
				std::cout<<"xb:\n";
				for (int i=0;i<d;i++){
				std::cout<<*(xb+i)<<"\t";
				}
				
				std::cout<<"\nxp:\n";
				for (int i=0;i<d;i++){
				std::cout<<*(xpb+i)<<"\t";
				}
				
				std::cout<<"\n grad:\n";
				for (int i=0;i<d;i++){
				std::cout<<*(grad+i)<<"\t";
				}*/			
				
				T fk(1,fmin);
				opt::math::cpu::dgemm_nn(1,1,d,c1,direction.begin(),1,d,grad,1,1,gfloat(1.0),fk.begin(),1,1);	
				//std::cout<<"f new: "<<fnew <<" f(xk)"<<*(fk.begin())<<"\n";
				if (fnew<=*(fk.begin())){
					return true;
				}
				else{
					return false;
				}
			}
			
			public:
			
			GNSCPU(gns::Residual<K,T,gns::HasJacInv::Yes>& _r,std::ostream& _os): r(_r),os(_os){
				//os<<"Hat alles geklappt, alter\n";
			}
			
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Gauss-Newton procedure.
			*/
			std::optional<T> run(T x0){
				
				bool run_finished=false;
				
				//Test is parameters already minimize f0 according to tol
				T residuals(r.dim);
				gfloat fmin=f0(x0.begin(),residuals.begin()); //current minimum
				
				if (fmin<tol){
					return {x0};
				}
				else{
					int d=x0.size();
					T J_t(r.dim*d);
					T grad(d);
					T J_t_J_inv(d*d);
					int n_threads=1;
					T C(r.dim*d);
					T xi(x0.begin(),x0.end());
					T xs(d*n_threads);
					gfloat alpha=0.7;		
					int iter=0;
					
					std::vector<std::thread> ts(n_threads);
					gfloat curr_min;
					
					while (run_finished==false){
						r.j_t(xi.begin(),J_t.begin());
						r.j_t_j_inv(xi.begin(),J_t_J_inv.begin());
		
						opt::math::cpu::dgemm_nn(d,r.dim,d,gfloat(1.0),J_t_J_inv.begin(),1,d,J_t.begin(),1,r.dim,gfloat(0.0),C.begin(),1,r.dim);
						
						
						/*Calculate gradient from indices of jacobi matrix. Used for backtracking stepsize finding later*/
						for (int i=0;i<d;i++){
							gfloat sum=0;
							for (int j=0;j<r.dim;j++){
								
								sum+=*(J_t.begin()+j+i*d)*gfloat(2.0)* *(residuals.begin()+j);
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
							//	t[i]=std::thread(opt::math::cpu::dgemm_nn(d,1,r.dim,beta,C.begin(),1,r.dim,residuals.begin(),1,r.dim,gfloat(1.0),(xi.begin()),1,d);
							
			
								ts[i]=std::thread(&GNSCPU::dgemm_and_residual,this,d,beta,C.begin(),xi.begin(),xs.begin()+i*d,residuals.begin(),std::ref(f0_vals[i]));
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
				
							if (bls(curr_min,fmin,betas[min_index],grad.begin(),xs.begin()+min_index,xs.begin()+min_index+d,xi.begin(),d)){
								fmin=curr_min;
								std::copy(xs.begin()+min_index,xs.begin()+min_index+d,xi.begin());
								r.r(xi.begin(),residuals.begin());
								std::cout<<"Current error: "<<fmin<<"\n";
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
						if (iter==4){
							return {};
						}

					}
					
				}
				return {};
				
			}

		};


	
		export template<ValidContainer T,gns::ResidualSpec K>
		class GNSCPU<T,K,gns::HasJacInv::No> {
		
			using gfloat=typename T::value_type;
			
			private:
			gns::Residual<K,T,gns::HasJacInv::No>& r;
			
			gfloat tol=0.001;
			
			bool f0(T& params){
			
				T result(1);
				T residuals(r.dim);
				r.r(params,residuals.begin());	
				opt::math::cpu::dgemm_nn(1,1,r.dim,gfloat(1.0),residuals.begin(),1,r.dim,residuals.begin(),1,1,gfloat(0.0),result.begin(),1,1);

				if (result[0]>tol){
					return false;
				}
				else{
					return true;
				}
			}
			
			std::ostream& os;
			
			
			public:
			
			GNSCPU(gns::Residual<K,T,gns::HasJacInv::No>& _r,std::ostream& _os): r(_r),os(_os){
				//os<<"Hat alles geklappt, alter\n";
			}
			
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Gauss-Newton procedure.
			*/
			std::optional<T> run(T x0){
				
	
				bool run_finished=false;
				
				//Test is parameters already minimize f0 according to tol

				
				
			}

		};	
	}
}

