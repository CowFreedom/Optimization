@ECHO OFF
cls

set "USE_CUDA=1"

set curpath=%~dp0
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"

echo Building executable
g++ -c %curpath:~0,-1%\..\core\transformation.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 

g++ -c %curpath:~0,-1%\..\solvers\gaussnewton.cpp -pthread -O3 -std=c++20 -fmodules-ts
g++ -c %curpath:~0,-1%\..\solvers\solvers.cpp -pthread -O3 -std=c++20 -fmodules-ts

if defined USE_CUDA (
	::Nice
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\gemm.cu -c -o gpu_gemm.obj 
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\gmv.cu -c -o gpu_gmv.obj
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\ldl.cu -c -o gpu_ldl.obj
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\lu.cu -c -o gpu_lu.obj
	nvcc %curpath:~0,-1%\..\gpu\hostgpu_bindings.cu -c -o hostgpu_bindings.obj


	::Building correctness test modules
	g++ -c %curpath:~0,-1%\tests.cpp -O3 -std=c++20 -fmodules-ts
	g++ -c %curpath:~0,-1%\correctness/correctness_utility.cpp -O3 -std=c++20 -fmodules-ts
	g++ -c %curpath:~0,-1%\correctness/test_corr_gauss_newton_cpu.cpp -O3 -std=c++20 -fmodules-ts

	
	

)

	::	g++ -c %curpath:~0,-1%\..\gpu\solvers_gpu.cpp -O3 -std=c++20 -fmodules-ts

	::Building correctness test modules
::	g++ -c %curpath:~0,-1%\tests.cpp -O3 -std=c++20 -fmodules-ts
::	g++ -c %curpath:~0,-1%\correctness/correctness_utility.cpp -O3 -std=c++20 -fmodules-ts
::	g++ -c %curpath:~0,-1%\correctness/test_corr_gauss_newton_cpu.cpp -O3 -std=c++20 -fmodules-ts
::	g++ -c %curpath:~0,-1%\correctness/test_corr_transformation.cpp -O3 -std=c++20 -fmodules-ts


	::Building gpu correctness test submodules and entire module
::	g++ -c %curpath:~0,-1%\correctness/test_corr_gauss_newton_gpu.cpp -O3 -std=c++20 -fmodules-ts
::	g++ -c %curpath:~0,-1%\correctness/test_corr_transformation_gpu.cpp -O3 -std=c++20 -fmodules-ts	
::	g++ -c %curpath:~0,-1%\correctness/correctness.cpp -O3 -std=c++20 -fmodules-ts
	
	::Building performance test modules and entire module
::	g++ -c %curpath:~0,-1%\performance/test_perf_transformation.cpp  -O3 -std=c++20 -fmodules-ts
	
	::Building gpu performance test submodules and entire module
::	g++ -c %curpath:~0,-1%\performance/test_perf_transformation_gpu.cpp  -O3 -std=c++20 -fmodules-ts
::	g++ -c %curpath:~0,-1%\performance/performance.cpp  -O3 -std=c++20 -fmodules-ts -Dopt_use_cuda 
	
	
	::Building final executabble
::	g++ %curpath:~0,-1%\run_diagnostics.cpp %curpath:~0,-1%\tests.obj %curpath:~0,-1%\performance.obj %curpath:~0,-1%\test_perf_transformation.obj  %curpath:~0,-1%\test_perf_transformation_gpu.obj %curpath:~0,-1%\correctness.obj %curpath:~0,-1%\correctness_utility.obj  %curpath:~0,-1%\test_corr_gauss_newton_cpu.obj %curpath:~0,-1%\test_corr_transformation.obj %curpath:~0,-1%\test_corr_gauss_newton_gpu.obj %curpath:~0,-1%\test_corr_transformation_gpu.obj %curpath:~0,-1%\hostgpu_bindings.obj %curpath:~0,-1%\gns_hostgpu.obj %curpath:~0,-1%\solvers_gpu.obj  %curpath:~0,-1%\gpu_gemm.obj %curpath:~0,-1%\gpu_gmv.obj %curpath:~0,-1%\gpu_ldl.obj %curpath:~0,-1%\gpu_lu.obj -L /LIBPATH:%cudapath% cudart.lib  -O3 -std=c++20 -fmodules-ts

del *.o
rmdir /q /s gcm.cache