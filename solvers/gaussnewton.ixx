module;
#include <ostream>
#include <optional>
#include <thread>
#include <future>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip> //std setprecision
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
			
			export enum class GNSError{
				Success,
				Numerical,
				InputError,
				NoStepSizeFound,
				NoConvergence
			};
			
			export enum class InfoType{
				Jacobian,
				ApproxHessian, //corresponds to A=(J^{T}*J)
				CholeskyApproxHessian, //corresponds to A=(J^{T}*J)=LDL^{T}
			};
			
			export template<class T, class F>
			void stream_iteration_info(int iteration,T vars, int xdim, F fmin,	std::streambuf* buf, std::string* names, int precision=15){
				std::ostream os(buf);
				os.precision(precision);
				os<<std::setw(17)<<std::left<<"Newton Iteration"<<std::setw(25)<<std::right<<iteration<<"\n"<<std::setw(17)<<std::left<<"f0"<<std::setw(25)<<std::right<<fmin<<"\n";	
				
				if (names){			
					const int name_width=30;
					const int num_width=25;
					const int num_fields=2; //number of table entries above
					const std::string separator=" |";
					const int total_width=name_width+num_width+separator.size()*num_fields;
					std::string line=separator + std::string(total_width-1,'-')+'|';
					os<<line<<"\n"<<separator;
					os<<std::setw(name_width)<<"Parameter"<<separator<<std::setw(num_width)<<"Estimate"<<separator<<"\n"<<line<<"\n";
					
					for (int i=0;i<xdim;i++){
						os<<separator<<std::setw(name_width)<<names[i]<<separator<<std::setw(num_width)<<vars[i]<<separator<<"\n"<<line<<"\n";
					
					}
					
				}				
			}			
		
	
			
			//l1 vector norm tolerance
			export template<class T,class F>
			bool abs_tol(int dim, T x, int stride_x, F tol){
				for (int i=0;i<dim;i++){
					float val=(x[i*stride_x]>=0.0)?x[i*stride_x]:-x[i*stride_x];
					if (val>tol){
						return false;
					}
				}
				return true;
			}
			
			export template<class T, class F>
			bool check_diagonal_has_zero(int n,T A, int stride_row, int stride_col){
				for (int i=0;i<n;i++){
					F val=A[i*stride_row+i*stride_col];

					if (val==0){
						return true;
					}
				}
				return false;
			}
			
			export template<class T, class F>
			void perturb_diagonal(int n,T A, int stride_row, int stride_col,F eps){
				for (int i=0;i<n;i++){
					A[i*stride_row+i*stride_col]+=eps;				
				}
			}
			
			template<class T>
			bool calculate_correlation(InfoType info, int n, int m, typename T::iterator A, int stride_col, int stride_row, std::streambuf* buf){
				if (info==InfoType::Jacobian){
						//TODO: ADD LOGIC
					if (buf){
						std::ostream os(buf);
					
					}
				}
				else{
						return false;
				}	
			}
		}
	
		export template<ValidContainer T,class U>
		class GNSCPU{
		};

	
		export template<ValidContainer T,class U>
		class GNSCPU<T,U> {
			
			using gfloat=typename T::value_type;
			
			private:
			U r; //represents residual function
			U j; //represents jacobian function
			U j_inv; //represents scheme to calculate jacobian inverse, like finite differences 
			int rdim;
			int xdim;
			bool (&xtol)(int,typename T::iterator, int, gfloat)=gns::abs_tol<typename T::iterator,gfloat>;
			std::streambuf* buf;
			std::ostream os;
			
			//Evaluates sum of squares objective function f0
			gfloat f0(typename T::const_iterator params,typename T::iterator residuals){
				gfloat result=0.0;
				r(params,residuals);
				
				for (int i=0;i<rdim;i++){
					result+=residuals[i] * residuals[i];
				}

				return result;
			}
			
			bool has_nans_or_infs(int dim,typename T::iterator x, int stride){
				for (int i=0;i<dim;i++){
					if (std::isnan(x[i*stride])||std::isinf(x[i*stride])){
						return true;
					}
				}
				return false;		
			}
			
			//backtracking line search
			bool bls(gfloat fnew, gfloat fcurrent, gfloat beta, typename T::iterator grad, typename T::iterator direction, int xdim){
			
				T fk(1,fcurrent);
				opt::math::cpu::dgmv(1,xdim,-c1*beta,direction,1,xdim,grad,1,gfloat(1.0),fk.begin(),1);	
				//std::cout<<std::setprecision(17)<<"fnew: "<<fnew<<"fcurrent:"<<fcurrent<<"\n";
				if (fnew<fk[0]){
					return true;
				}
				else{
					return false;
				}
			}
			
			
			public:
			gfloat lambda=0.5; //determines, how much the step size is reduced at each iteration of the wolfe conditions
			int n_threads=std::thread::hardware_concurrency();
			int max_step_iter=600/n_threads; //maximum number of iterations during the stepsize finding process
			int max_iter=300; //maximum number of iterations
			int step_nthreads=(60>n_threads)?n_threads:60;
			gfloat c1=0.0001;
			gfloat alpha=0.8;	//decrease factor in the newton stepsize iterations	
			gfloat tol=0.0001;
			gfloat lu_pertubation=5; //If LDL and LU decompositions fail, the approximated hessian J^{T}*J is perturbed along the diagonal by this factor
			std::string* parameter_names=nullptr;//Optionally, parameter names can be set for later display
			
			bool (*send_numerical_info)(gns::InfoType, int n, int m, typename T::iterator, int stride_col, int stride_row)=nullptr; 
			
			GNSCPU(U _r,U _j,int _xdim, int _rdim, std::streambuf* _buf): r(_r), j(_j),rdim(_rdim),xdim(_xdim),buf(_buf),os(_buf){
				
			}	
			GNSCPU(U _r,U _j,int _xdim, int _rdim): r(_r), j(_j),rdim(_rdim),xdim(_xdim),buf(nullptr),os(nullptr){
				
			}				
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Gauss-Newton procedure.
			*/
			std::optional<T> optimize(T x0, gns::GNSError* gns_error=nullptr){
	
				if (x0.size()<xdim){
					os<<"Size of the input container does not fit the dimension of the problem";
					if(gns_error){
						*gns_error=gns::GNSError::InputError;
					}
					return {};
				}
				//Test is parameters already minimize f0 according to tol
				T residuals(rdim*n_threads); //collection of residuals for each thread
				typename T::iterator residual=residuals.begin();
				int c_r=0; //current best residual
				gfloat fmin=f0(x0.begin(),residuals.begin()); //current minimum
				T J(xdim*rdim);
				T grad(xdim);
				T b(xdim); //output vector in J^{T}*J*v=b
				T v(xdim);//unknown vector in J^{T}*J*v=b
				T J_t_J(xdim*xdim);
				T xi(x0.begin(),x0.end()); //current parameters
				int iter=0;
				
				std::vector<std::thread> ts(n_threads);
				std::future<bool> future;
				gfloat curr_min;
				
				bool* result_valid=new bool[step_nthreads]();	
				std::vector<gfloat> f0_vals(n_threads);
				std::vector<gfloat> betas(n_threads);				
				do{
					
					j(xi.begin(),J.begin()); //get current value of Jacobian
										
					/*Calculate gradient from indices of jacobi matrix. Used for backtracking stepsize finding later*/
					//used for sum calculation. not necessary but probably faster due cache misses in the traversal of array
					for(int i=0;i<xdim;i++){
						grad[i]=0.0;
					}
					
					//If specified, send jacobian information for further processing
					if (send_numerical_info){
						future = std::async(std::launch::async, send_numerical_info,gns::InfoType::Jacobian,rdim,xdim,J.begin(),1,xdim);
					}
					
					gfloat all_zeros=0.0;
					
					for (int i=0;i<rdim;i++){
						for (int j=0;j<xdim;j++){
							grad[j]+=J[i*xdim+j]*gfloat(2.0)* residual[i];
							all_zeros+=grad[j];
						}
					}
					if (all_zeros==0.0){
						if (gns_error){
							*gns_error=gns::GNSError::Success;
						}
						delete[] result_valid;
						return{xi};
					}
					
					opt::math::cpu::gemm(xdim,xdim,rdim,gfloat(1.0),J.begin(),xdim,1,J.begin(),1,xdim,gfloat(0.0),J_t_J.begin(),1,xdim); //calculate J^{T}*J
					opt::math::cpu::dgmv(xdim,rdim,gfloat(1.0),J.begin(),xdim,1,residual,1,gfloat(0.0),b.begin(),1); //calculate vector for J^{T}*J*v=b
					opt::math::cpu::choi<typename T::iterator, gfloat>(xdim, J_t_J.begin(), 1, xdim); //calculate cholesky diagonal A=LDL^{T} decomposition				
					bool diagonal_has_zero=gns::check_diagonal_has_zero<T::iterator,gfloat>(xdim,J_t_J.begin(),1,xdim);
					
					if(!diagonal_has_zero){
						opt::math::cpu::choi_solve<typename T::iterator,typename T::iterator,typename T::iterator,gfloat>(xdim, J_t_J.begin(), 1, xdim, b.begin(), 1, v.begin(), 1);
						
					}
					else{
						//Using LU decomposition
						if(buf){
							os<<"In the calculation of the descent direction LU decomposition was used instead of LDL^{T}. This was due a zero on the diagonal in the LDL^{T} decomposition\n";
						}
						
						//Recreate the initial state of J_t_J_d;
						opt::math::cpu::gemm(xdim,xdim,rdim,gfloat(1.0),J.begin(),xdim,1,J.begin(),1,xdim,gfloat(0.0),J_t_J.begin(),1,xdim); //calculate J^{T}*J
						opt::math::cpu::lu_single<typename T::iterator, gfloat>(xdim,J_t_J.begin(),1,xdim);
						diagonal_has_zero=gns::check_diagonal_has_zero<T::iterator,gfloat>(xdim,J_t_J.begin(),1,xdim);
						
						/*If matrix still singular, slightly perturb diagonal and redo LU decomposition*/
						if (diagonal_has_zero){
							if(buf){
								os<<"Diagonal still zero after LU decomposition. Perturbing the diagonal of approximated Hessian and redoing decomposition.\n";
							}
							opt::math::cpu::gemm(xdim,xdim,rdim,gfloat(1.0),J.begin(),xdim,1,J.begin(),1,xdim,gfloat(0.0),J_t_J.begin(),1,xdim); //recalculate J^{T}*J
							gns::perturb_diagonal<T::iterator,gfloat>(xdim,J_t_J.begin(),1,xdim,lu_pertubation);

						}
						opt::math::cpu::lu_single<typename T::iterator, gfloat>(xdim,J_t_J.begin(),1,xdim);
						
						opt::math::cpu::lu_solve<typename T::iterator,typename T::iterator,typename T::iterator,gfloat>(xdim,1, J_t_J.begin(), 1, xdim, b.begin(), 1,1, v.begin(), 1,1);

						if (has_nans_or_infs(xdim, J_t_J.begin(), 1)){
							if(buf){
								os<<"The descent direction contains at least a NaN or infinity value. This is often caused by numerical errors (limited digit precision).\n";
							}
							if (gns_error){
								*gns_error=gns::GNSError::Numerical;
							}				
							delete[] result_valid;
							return {};
						}
					}								
					if (xtol(xdim,v.begin(),1,tol)){
						delete[] result_valid;
						
						if (gns_error){
							*gns_error=gns::GNSError::Success;
						}
						return {xi};
					}
					gfloat beta=alpha;

					int step_iter=0;
					bool step_size_found=false;
					
					//maybe take U by reference
					auto eval=[](U r, typename T::iterator residual, int rdim, gfloat beta, typename T::iterator v, typename T::iterator xi, int xdim, gfloat alpha, gfloat& f0_val,bool& pars_changed){
						T _x(xdim); //new parameters
						
						for (int i=0;i<xdim;i++){
							_x[i]=gfloat(-1.0)*beta*v[i]+xi[i]; //get new parameters
						}
						
						auto check_if_different=[](typename T::iterator a,typename T::iterator b, int dim){
							for (int i=0;i<dim;i++){
								if (a[i]!=b[i]){
									return true;
								}
							}
							return false;
						};
				
						pars_changed=check_if_different(_x.begin(),xi,xdim); //if equal then due to rounding errors nothing changed. We should abort estimaton if this thread is selected later
							
						if (pars_changed){
							f0_val=0.0;
							r(_x.begin(),residual); //evaluate objective function
							gfloat temp=0.0;
							for (int i=0;i<rdim;i++){
								temp+=residual[i] * residual[i];
							}
							f0_val=temp;						
						}
						else{						
							f0_val=std::numeric_limits<gfloat>::infinity();
						}

						
					};
					do{

						for (int i=0;i<step_nthreads;i++){
							ts[i]=std::thread(eval,r,residuals.begin()+i*rdim,rdim,beta, v.begin(),xi.begin(), xdim, alpha,std::ref(f0_vals[i]),std::ref(result_valid[i]));	
							betas[i]=beta;
							beta*=lambda;
						}
						for (auto& t: ts){
							t.join();
						}
						
						//Find step size that minimizes f0 the most						
						curr_min=f0_vals[0];
						int min_index=0;
						for (int i=1;i<step_nthreads;i++){
							if (f0_vals[i]<curr_min){
								curr_min=f0_vals[i];
								min_index=i;
							}
						}
						if (result_valid[min_index]){
							//recover the stepsize factor that minimized f0 the most in this iteration, betaopt=alpha*lambda^{step_iter*step_nthreads+min_index}
							gfloat betaopt=alpha;
							for (int i=1;i<=step_iter*step_nthreads+min_index;i++){
								betaopt*=lambda;
							}
							
							//std::cout<<xi[0]<<"  "<<xi[1]<<"direction: "<<(-betaopt*v[0])<<"\t"<<(-betaopt*v[1])<<"\n";
							//std::cout<<v[0]<<"\t"<<v[1]<<"\n";
							//std::cout<<(xi[0]-betaopt*v[0])<<"   "<<(xi[1]-betaopt*v[1])<<"\n";
							if (bls(curr_min, fmin, betaopt, grad.begin(), v.begin(),xdim)){
								c_r=min_index*rdim;
								residual=residuals.begin()+c_r;
								fmin=curr_min;
								
								for (int i=0;i<xdim;i++){
									xi[i]=gfloat(-1.0)*betaopt*v[i]+xi[i]; //get new parameters
									//std::cout<<"x"<<xi[0]<<"  "<<xi[1]<<"\n";
								}
								if (buf){
									gns::stream_iteration_info(iter,xi.begin(), xdim, fmin, buf, parameter_names);
									os<<"\n";
								}
								step_size_found=true;
							}
							step_iter++;												
						}
						else{
							if(buf){
								os<<"Stepsize cannot be added to the current position. Possible causes:\n* The stepsize is so low that adding the descent vector to the position is below doubles' machine precision\n\n*The descent direction is wrong due to numerical errors in its calculation\n";
							}
							break;
						}
					}
					while(!step_size_found && step_iter<=max_step_iter);
													
					if (!step_size_found){
						if (buf){
							os<<"No step size found\n";
						}
						if (gns_error && step_iter>max_step_iter){
							*gns_error=gns::GNSError::Numerical;
						}
						else if (gns_error){
							*gns_error=gns::GNSError::NoStepSizeFound;
						}
						break;
					}
					
					if (send_numerical_info){
						if (!future.get()){
							if (buf){
								os<<"send_numerical_info did not finish successfully\n";
							}
						}
					}
			
					iter++;
				}
				while (iter<max_iter);
				delete[] result_valid;
				if (gns_error){
							*gns_error=gns::GNSError::NoConvergence;
				}
				return {};
				
			}


		
		};


		
	}
}

