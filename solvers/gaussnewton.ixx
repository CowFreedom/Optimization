module;
#include <ostream>
#include <optional>
export module optimization.solvers:gaussnewton;

namespace opt{
	namespace solvers{
		
		namespace gns{
			
			enum class ResidualSpec{
				Pure,
				Data
			};
			
			template<ResidualSpec K,class T>
			class Residual{
			
				public:
		
			};
			
			export template<class T>
			class Residual<ResidualSpec::Pure,T>{
			
			
				typename T::value_type (&r)(T storage_begin, int n); //the container contains a vector of residuals.
				
				
				public:
				Residual(typename T::value_type (&_r)(T storage_begin, int n)):r(_r){
				
				}
			
			};
			
			template<class T>
			class Residual<ResidualSpec::Data,T>{
			
			
				typename T::value_type (&r)(T storage_begin, int n); //the container contains a vector of residuals.
				
				T v;
				
				public:
				Residual(typename T::value_type (&_r)(T storage_begin, int n),T _v):r(_r),v(_v){
				
				}
			
			};
			
			export template<class T>
			using ResidualPure=Residual<ResidualSpec::Pure,T>;

			export template<class T>
			using ResidualData=Residual<ResidualSpec::Data,T>;		
	
		}
	
	
	
		export template<class T,gns::ResidualSpec K>
		class GNSCPU {
		
			using gfloat=typename T::value_type;
			
			private:
			gns::Residual<K,T>& r;
		
			
			
			public:
			
			GNSCPU(gns::Residual<K,T>& _r): r(_r){
				//os<<"Hat alles geklappt, alter\n";
			}
			
			
			/*! Runs Gauss Newton's algorithm. Only this function has to be called to run the complete procedure.
			@param[in] initial_params Initial parameters containing starting values for the procedure.
			\return Code indicating success or failure of running the Newton procedure.
			*/
			std::optional<T> run(T x0, std::ostream& os){
			
			}
			
			

		};	
	}
}

