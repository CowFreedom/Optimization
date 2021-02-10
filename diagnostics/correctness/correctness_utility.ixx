module;
#include <random>
#include<ostream>
export module tests.correctness:utility;


namespace opt{
	namespace test{
		namespace corr{
	
			template<class T, class F>
			bool verify_calculation(T A, T B, int n_elems, F tol){
			
				for (int i=0;i<n_elems;i++){
					F val=A[i]-B[i];
					val=(val<0)?-val:val;
					if (val>tol || (std::isinf(A[i])) || (std::isinf(B[i])) ||(std::isnan(A[i])) || (std::isnan(B[i]))){
					//std::cout<<"Error at:"<<i<<"\n";
						return false;
					}
				}
				return true;
			}
			

			template<class T, class F>
			bool verify_nonpacked_vs_packed(const T nonpacked_mat, const T packed_mat, int n, F tol){
				for (int i=0;i<n;i++){
					for (int j=0;j<=i;j++){
						int ix1=i*n+j;
						int ix2=i*0.5*(i+1)+i*0+j;
						
						F val=nonpacked_mat[ix1]-packed_mat[ix2];
						val=(val<0)?-val:val;
						if (val>tol){
							//	std::cout<<nonpacked_mat[ix1]<<" vs. "<<packed_mat[ix2]<<"at position "<<i*n+j<<"\n";
								return false;
							}
						}				
					}

				return true;
			}
			
			template<class T, class F>
			bool verify_upperlower(const T C1, const T C2, int n, const char selection, F tol){
				switch (selection){
					case 'U':{
							for (int i=0;i<n;i++){
								for (int j=i;j<n;j++){
									if (std::abs(C1[i*n+j]-C2[i*n+j])>tol){
									//std::cin.get();
									return false;
								}
								}
								
							}
							
							break;		
					}
					
					case 'L':{
							for (int i=0;i<n;i++){
								for (int j=0;j<i;j++){
									if (std::abs(C1[i*n+j]-C2[i*n+j])>tol){
									return false;
								}
								}	
							}
							
							break;		
					}	
				}

				return true;
			}
		

			export template<class C, class T>
			class Circle{
			
				public:
				Circle(int x, int y){
					offset[0]=x;
					offset[1]=y;
				}
				typename T::value_type offset[2]={0,0};
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
			
			
			export template<class C, class T,class F>
			class CircleB{
			
				public:
				CircleB(int x, int y){
					offset[0]=x;
					offset[1]=y;
				}
				F offset[2]={0,0};
				void residual(C params, T storage){
					F x0=*params;
					F x1=*(params+1);
					*storage=x0-offset[0];
					*(storage+1)=x1-offset[1];
				}

				void jacobian(C x, T storage){
					F x0=*x;
					F x1=*(x+1);
					*storage=1;
					*(storage+1)=0;
					*(storage+2)=0;
					*(storage+3)=1;	
				}

				const int rdim=2;
				const int xdim=2;
				
			};
		
			export template<class C, class T, class F>
			class Convex1{
				
				private:
				F a;
				F b;
				
				public:
				Convex1(F _a, F _b):a(_a),b(_b){
				}

				void residual(C params, T storage){
					F x=*params;
					F y=*(params+1);
					*storage=a*a*x*x+a*b*x*y+b*b*y*y;
				}
				
				void jacobian(C input, T storage){
					F x=*input;
					F y=*(input+1);
					//std::cout<<x<<"\t"<<y<<"\n\n";
					*storage=2*a*a*x+a*b*y;
					*(storage+1)=2*b*b*y+a*b*x;
				}
				
				//not needed
				void hessian(C input, T storage){
					F x=*input;
					F y=*(input+1);
					*storage=2*a*a;
					*(storage+1)=a*b;
					*(storage+2)=2*b*b;
					*(storage+3)=a*b;
				}
				
				const int rdim=1;
				const int xdim=2;
			};
/*
			export template<class C, class T,class F>
			class Transport1D{
				
				private:
				double c;
				
				public:
				Transport1D(F _a, F _b):a(_a),b(_b){
				}

				void residual(C params, T storage){
					F x=*params;
					F y=*(params+1);
					*storage=a*a*x*x+a*b*x*y+b*b*y*y;
				}
				
				void jacobian(C input, T storage){
					F x=*input;
					F y=*(input+1);
					//std::cout<<x<<"\t"<<y<<"\n\n";
					*storage=2*a*a*x+a*b*y;
					*(storage+1)=2*b*b*y+a*b*x;
				}
				
				//not needed
				void hessian(C input, T storage){
					F x=*input;
					F y=*(input+1);
					*storage=2*a*a;
					*(storage+1)=a*b;
					*(storage+2)=2*b*b;
					*(storage+3)=a*b;
				}
				
				const int rdim=1;
				const int xdim=2;
			};			
*/


		}
	}
}
