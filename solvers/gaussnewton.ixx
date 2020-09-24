module;
#include <ostream>
#include <optional>
#include <thread>
#include <iostream>
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
				Residual(void (&_r)(const T& params, typename T::iterator storage), int _dim):r(_r),dim(_dim){
				
				}
				
				void (&r)(const T& params, typename T::iterator storage); //the container contains a vector of residuals.
				
				const int dim;
			
			};
			
			template<ValidContainer T>
			class Residual<ResidualSpec::Data,T,HasJacInv::No>{
			
				T v;
				
				public:
				Residual(void (&_r)(const T& params, typename T::iterator storage),T _v, int _dim):r(_r),v(_v), dim(_dim){
				
				}
			
				void (&r)(const T& params,typename T::iterator storage); //the container contains a vector of residuals.
				
				
				const int dim;
			
			};
			
			
			template<ValidContainer T>
			class Residual<ResidualSpec::Pure,T,HasJacInv::Yes>{
			
				T v;
				public:
				Residual(void (&_r)(const T& params, typename T::iterator storage),void (&_j_t)(const T&, typename T::iterator),void (&_j_t_j_inv)(const T&, typename T::iterator),int _dim):r(_r),j_t(_j_t),j_t_j_inv(_j_t_j_inv),dim(_dim){
				
				}
			
				void (&r)(const T& params,typename T::iterator storage); //the container contains a vector of residuals.
				
				//jacobi matrix
				void (&j_t)(const T&, typename T::iterator);
				
				void (&j_t_j_inv)(const T&, typename T::iterator);
				
				const int dim;
			
			};
			template<ValidContainer T>
			class Residual<ResidualSpec::Data,T,HasJacInv::Yes>{
			
				T v;
				public:
				Residual(void (&_r)(const T& params, typename T::iterator storage),void (&_j_t)(const T&, typename T::iterator),void (&_j_t_j_inv)(const T&, typename T::iterator),T _v, int _dim):r(_r),v(_v), j_t(_j_t),j_t_j_inv(_j_t_j_inv),dim(_dim){
				
				}
			
				void (&r)(const T& params,typename T::iterator storage); //the container contains a vector of residuals.
				
				//jacobi matrix
				void (&j_t)(const T&, typename T::iterator);
				
				void (&j_t_j_inv)(const T&, typename T::iterator);
				
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
			
			gfloat tol=0.001;
			
			bool f0(T& params,T& residuals){
			
				T result(1);
				r.r(params,residuals.begin());	
				opt::math::cpu::dgemm_nn(1,1,r.dim,gfloat(1.0),residuals.begin(),1,r.dim,residuals.begin(),1,1,gfloat(0.0),result.begin(),1,1);
				std::cout<<"Error now: "<<result[0]<<"\n";
				if (result[0]>tol){
					return false;
				}
				else{
					return true;
				}
			}
			
			std::ostream& os;
			
			
			public:
			
			GNSCPU(gns::Residual<K,T,gns::HasJacInv::Yes>& _r,std::ostream& _os): r(_r),os(_os){
				//os<<"Hat alles geklappt, alter\n";
			}
			
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Gauss-Newton procedure.
			*/
			std::optional<T> run(T x0){
				
				//, void (&jacobian)(const T&, typename T::iterator)
				//	size_t j_n=target_data.size(); //height of jacobi matrix
				//size_t j_m=parameters[0].len(); //length of jacobi matrix
					//x_n=evaluator.eval(initial_params,target_times)[0];
				//	size_t iter=0;
				//	bool run_finished=false;
				//	T e_g2=T(0.0); ;//adagrad like adaptive stepsize factor
					
				bool run_finished=false;
				
				//Test is parameters already minimize f0 according to tol
				T residuals(r.dim);
				if (f0(x0,residuals)){
					return {x0};
				}
				else{
					int d=x0.size();
					T J_t(r.dim*d);
					T J_t_J_inv(d*d);
					
					T C(r.dim*d);
					int iter=0;
									
					while (run_finished==false){
						
						r.j_t(x0,J_t.begin());
						r.j_t_j_inv(x0,J_t_J_inv.begin());
						
						opt::math::cpu::dgemm_nn(d,r.dim,d,gfloat(1.0),J_t_J_inv.begin(),1,d,J_t.begin(),1,r.dim,gfloat(0.0),C.begin(),1,1);
						
						gfloat alpha=-1.0;		
						
						for (auto& x:C){
						os<<x<<"\t";
						}
						std::cout<<"\n\n";
						
						opt::math::cpu::dgemm_nn(d,1,r.dim,alpha,C.begin(),1,r.dim,residuals.begin(),1,r.dim,gfloat(1.0),x0.begin(),1,d);
						
						
						f0(x0,residuals);
						//J_t_J
						
						for (auto& x:x0){
						os<<x<<"\t";
						}
						
						/*

						for (auto& x:J_t){
							os<<x<<"\t";
						}
						for (auto& t: ts){
							r.r(x0,9)
						}
						for (auto& t: ts){
							t.join();
						}
						*/
						//run_finished=true;
						iter++;
						if (iter==50){
							run_finished=true;
						}
					//	return {};
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
				
				//, void (&jacobian)(const T&, typename T::iterator)
				//	size_t j_n=target_data.size(); //height of jacobi matrix
				//size_t j_m=parameters[0].len(); //length of jacobi matrix
					//x_n=evaluator.eval(initial_params,target_times)[0];
				//	size_t iter=0;
				//	bool run_finished=false;
				//	T e_g2=T(0.0); ;//adagrad like adaptive stepsize factor
					
				bool run_finished=false;
				
				//Test is parameters already minimize f0 according to tol
				if (f0(x0)){
					return {x0};
				}
				else{
					int d=x0.size();
					T J(r.dim*d);
								
					
					while (f0(x0) && run_finished==false){
						
						//jacobian(x0,J.begin());
						
						
						//J_t_J
						
						for (auto& x:J){
						os<<x<<"\t";
						}
						/*

						for (auto& t: ts){
							r.r(x0,9)
						}
						for (auto& t: ts){
							t.join();
						}
						*/
						run_finished=true;
						return {};
					}
					

					
				}
				
				
			}

		};	
	}
}

