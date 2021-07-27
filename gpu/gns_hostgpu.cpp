module;
#include <ostream>
#include <optional>
#include <vector>
#include <thread>
#include <future>
#include <string>
#include <iomanip>
#include <cmath>
#include <limits>
#include "hostgpu_bindings.h"
export module optimization.gpu.solvers.gaussnewton;

//import optimization.solvers;
import optimization.solvers.gaussnewton;

namespace opt{
	namespace solvers{
		namespace gpu{
		
			export template<class U,class F>
			class GNSGPU;

			export template<class U>
			class GNSGPU<U,float>{
				
				private:
				U r; //represents residual function
				U j; //represents function that calculates jacobian
				U j_inv; //represents function to calculate jacobian inverse
				bool (&xtol)(int, const float*, int, float)=opt::solvers::gns::abs_tol<const float*, float>;
				int rdim; //dimension of the residual vector
				int xdim; //dimension of the input vector
				int n_threads=std::thread::hardware_concurrency();
						
				std::streambuf* buf;
				std::ostream os;
				
				float f0(const float* params, float* residuals){
					float result=0.0;
					r(params,residuals);		
					for (int i=0;i<rdim;i++){
						result+=residuals[i]*residuals[i];
						
					}
					return result;	
				}
				
				bool check_if_real(float* x, int dim){
					for (int i=0;i<dim;i++){
						if (std::isnan(x[i])||std::isinf(x[i])){
							return false;
						}
					}
					return true;		
				}
				
				//backtracking line search
				bool bls(float fnew, float fcurrent, float beta, float* grad, float* direction){

					float mult=-c1*beta;		
					float sum=mult*direction[0]*grad[0];
					for (int i=1;i<xdim;i++){
						sum+=mult*direction[i]*grad[i];
					}
					sum+=fcurrent;
					//std::cout<<"fnew:"<<fnew<<"versus f(x)+c*gradF: "<<sum<<" fcurrent:"<<fcurrent <<"direction:"<<direction[0]<<" ,"<<direction[1]<<"\n";
					if (fnew<sum){
						return true;
					}
					else{
						return false;
					}
				}
				
				public:
				float lambda=0.2; //determines how much the step size is reduced in each iteratin
				int max_iter=300; //maximum number of iterations
				int max_step_iter=60/n_threads; //maxium number of iterations stepsize finding process
				int step_nthreads=(60>n_threads)?n_threads:60;
				float c1=0.0001;
				float alpha=0.8; //decrease factor in the step size finding iterations
				float tol=0.0001;
				bool (*send_numerical_info)(opt::solvers::gns::InfoType, int n, int m, float*, int stride_col, int stride_row)=nullptr; 
				std::string* parameter_names=nullptr; //Optionally, parameter names can be set for later display
							
				GNSGPU(U _r, U _j, int _xdim, int _rdim, std::streambuf* _buf): r(_r), j(_j), rdim(_rdim), xdim(_xdim),buf(_buf), os(_buf){
				}
				
				GNSGPU(U _r, U _j, int _xdim, int _rdim): r(_r), j(_j), rdim(_rdim), xdim(_xdim),buf(nullptr), os(nullptr){
				}
				
				//x0 is the initial value and xi the result that will be stored
				std::optional<float*> optimize(float* x0, float* xi,int _xdim, opt::solvers::gns::GNSError* gns_error=nullptr){
					bool run_finished=false;
					
					if (_xdim!=xdim){
						os<<"Size of the input container does not fit the dimension of the problem";
						if(gns_error){
							*gns_error=opt::solvers::gns::GNSError::InputError;
						}
						return {};
					}
					
					//Test if initial parameters already minimize f0 according to tol
					float* residuals=new float[rdim*n_threads];

					float fmin=f0(x0,residuals);

					float* residual=residuals;
				
					int c_r=0; //current best residual;
				
					float* J=new float[xdim*rdim];
					float* grad=new float[xdim];
					//float* xi=new float[xdim];
					float* direction= new float[xdim];
					
					for (int i=0;i<xdim;i++){
						xi[i]=x0[i];
					}
					int iter=0;
					
					std::vector<std::thread> ts(n_threads);
					std::future<bool> future;
					
					float curr_min;
					std::vector<float> f0_vals(step_nthreads);
					std::vector<float> betas(step_nthreads);
					bool* result_valid=new bool[step_nthreads]();		
					
					do{	
						j(xi,J);
						
						/*Calculate gradient from indices of jacobi matrix*/
						for (int i=0;i<xdim;i++){
							grad[i]=0.0;
						}
						float all_zeros=0.0;
						
						for (int i=0;i<rdim;i++){
							for (int j=0;j<xdim;j++){
								grad[j]+=J[i*xdim+j]*2.0*residual[i];
								all_zeros+=grad[j];
							}
						}
						
						if (all_zeros==0.0){
							if (gns_error){
								*gns_error=opt::solvers::gns::GNSError::Success;
							}
							delete[] result_valid;
							return{xi};
						}
						
						//If specified, send jacobian information for further processing
						if (send_numerical_info){
							future = std::async(std::launch::async, send_numerical_info,opt::solvers::gns::InfoType::Jacobian,rdim,xdim,J,1,xdim);
						}
						
						bool* lu_used=new bool();

						calc_stepdirection_f32(rdim,xdim,xi,residual,J,direction,lu_used);

						if (*lu_used){
							os<<"In the calculation of the descent direction LU decomposition was used instead of LDL^{T}. This was due a zero on the diagonal in the LDL^{T} decomposition\n";
							
							bool is_real=check_if_real(direction,xdim);
							if (!is_real){
								if(buf){
									os<<"The descent direction contains at least a NaN or infinity value. This is often caused by numerical errors (limited digit precision). Changing the solver to data type \"double\" might resolve this problem.\n";
								}
								if (gns_error){
								*gns_error=opt::solvers::gns::GNSError::Numerical;
							}	
								return {};
							}				
						}
						delete lu_used;
						
						if (xtol(xdim,direction,1,tol)){
							delete[] residuals;
							delete[] J;
							delete[] grad;
							delete[] direction;	
							delete[] result_valid;
							
							if (gns_error){
								*gns_error=gns::GNSError::Success;
							}
							return {xi};
						}
						
						float beta=alpha;
						
						int step_iter=0;
						bool step_size_found=false;
						
						auto eval=[](U r, float* residual, int rdim, float beta, float* v, float* xi, int xdim, float alpha, float& f0_val, bool& pars_changed){
							float* _x=new float[xdim];
						//	std::cout<<"beta:"<<beta<<"\n";
							for (int i=0;i<xdim;i++){
								_x[i]=-1.0*beta*v[i]+xi[i];
							}
							
							auto check_if_different=[](const float* a, const float* b, int dim){
								for (int i=0;i<dim;i++){
									if (a[i]!=b[i]){
										return true;
										}
								}
								return false;
							};
							
							pars_changed=check_if_different(_x,xi,xdim); //if equal then due to rounding errors nothing changed. We should abort estimaton if this thread is selected later

							if (pars_changed){
								f0_val=0.0;
								r(_x,residual);
								float temp=0.0;
								
								for (int i=0;i<rdim;i++){
									temp+=residual[i]*residual[i];
								}
								//std::cout<<"eval:new position:"<<_x[0]<<" and new f0:"<<temp<<"old position:"<<xi[0]<<"\n";
								f0_val=temp;
								delete[] _x;					
							}
							else{
								//If parameters have not changed then the result is not valid
								f0_val=std::numeric_limits<float>::infinity();
								delete[] _x;
							}				
						};
						
						do{
							for (int i=0;i<step_nthreads;i++){
								ts[i]=std::thread(eval,r,residuals+i*rdim,rdim,beta,direction,xi,xdim,alpha,std::ref(f0_vals[i]),std::ref(result_valid[i]));
								betas[i]=beta;
								beta*=lambda;
							}
										
							for (auto& t: ts){
								t.join();
							}			
							curr_min=f0_vals[0];
							int min_index=0;
							//std::cout<<f0_vals[0]<<" and "<<f0_vals[1]<<"\n";
							for (int i=1;i<step_nthreads;i++){
								if (f0_vals[i]<curr_min){
									curr_min=f0_vals[i];
									min_index=i;
								}
							}
							
							if (result_valid[min_index]){
								float betaopt=alpha;
							
								for (int i=1;i<=step_iter*step_nthreads+min_index;i++){
									betaopt*=lambda;
								}
								
								if (bls(curr_min,fmin,betaopt,grad,direction)){
									c_r=min_index*rdim;
									residual=residuals+c_r;
									fmin=curr_min;
									//std::cout<<"betaopt:"<<betaopt<<"\n";
									for (int i=0;i<xdim;i++){
										xi[i]=-1.0*betaopt*direction[i]+xi[i];
									}
									if (buf){
											opt::solvers::gns::stream_iteration_info(iter,xi, xdim, fmin, buf, parameter_names);
											os<<"\n";
									}
									step_size_found=true;	
								}
								//std::cout<<"step_iter:"<<step_iter<<"\n";
								step_iter++;				
							}
							else{
								if(buf){
									os<<"The stepsize times descent position cannot be added to the current position. This is probably due to numerical issues, i.e. the operation is below float's machine precision. Changing the solver to data type \"double\" might resolve this problem.\n";
								}								
								break;
							}

						}				
						while((step_size_found==false) && (step_iter<=max_step_iter));

						if (!step_size_found){
							if (gns_error && step_iter>max_step_iter){
								*gns_error=opt::solvers::gns::GNSError::Numerical;
							}
							else if (gns_error){
								*gns_error=opt::solvers::gns::GNSError::NoStepSizeFound;
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
					while(iter<max_iter);	
					delete[] J;
					delete[] grad;
					delete[] direction;						
					delete[] residuals;
					delete[] result_valid;
					if (gns_error){
						*gns_error=opt::solvers::gns::GNSError::NoConvergence;
					}
					return {};
				}
			};		
		}
	}
}
