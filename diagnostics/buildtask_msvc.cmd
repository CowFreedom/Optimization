@ECHO OFF

:: This Windows CMD script builds the diagnostics executable
cls
set curpath=%~dp0
set "USE_CUDA=1"
echo Active path:
echo %curpath:~0,-1%
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"
echo %cudapath%
echo Building executable



::Building core program
cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\..\core\transformation.cpp /D USE_EXPLICIT_SIMD /interface
cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\..\solvers\gaussnewton.cpp /interface
cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\..\solvers\solvers.cpp /interface

if defined USE_CUDA (
	::Building CUDA
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\gemm.cu -c -o gpu_gemm.obj 
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\gmv.cu -c -o gpu_gmv.obj
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\ldl.cu -c -o gpu_ldl.obj
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\lu.cu -c -o gpu_lu.obj
	nvcc %curpath:~0,-1%\..\gpu\hostgpu_bindings.cu -c -o hostgpu_bindings.obj

	::Building gpu modules
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\..\gpu\gns_hostgpu.cpp /interface
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\..\gpu\solvers_gpu.cpp /interface
	
	
	::Building correctness test modules
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\tests.cpp /interface
	cl /Dopt_use_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/correctness_utility.cpp /interface
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/test_corr_gauss_newton_cpu.cpp /interface
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/test_corr_transformation.cpp /interface

	::Building gpu correctness test submodules and entire module
	cl /Dopt_use_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/test_corr_gauss_newton_gpu.cpp /interface
	cl /Dopt_use_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/test_corr_transformation_gpu.cpp /interface
	cl /Dopt_use_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/correctness.cpp /interface
	
	::Building performance test modules and entire module
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\performance/test_perf_transformation.cpp /interface
	
	::Building gpu performance test submodules and entire module
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\performance/test_perf_transformation_gpu.cpp /interface
	cl /Dopt_use_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\performance/performance.cpp /interface
	
	::Building final executabble
	cl /EHsc /Dopt_use_cuda /std:c++latest /O2  %curpath:~0,-1%\run_diagnostics.cpp %curpath:~0,-1%\tests.obj %curpath:~0,-1%\performance.obj %curpath:~0,-1%\test_perf_transformation.obj  %curpath:~0,-1%\test_perf_transformation_gpu.obj %curpath:~0,-1%\correctness.obj %curpath:~0,-1%\correctness_utility.obj  %curpath:~0,-1%\test_corr_gauss_newton_cpu.obj %curpath:~0,-1%\test_corr_transformation.obj %curpath:~0,-1%\test_corr_gauss_newton_gpu.obj %curpath:~0,-1%\test_corr_transformation_gpu.obj %curpath:~0,-1%\hostgpu_bindings.obj %curpath:~0,-1%\gns_hostgpu.obj %curpath:~0,-1%\solvers_gpu.obj  %curpath:~0,-1%\gpu_gemm.obj %curpath:~0,-1%\gpu_gmv.obj %curpath:~0,-1%\gpu_ldl.obj %curpath:~0,-1%\gpu_lu.obj /link /LIBPATH:%cudapath% cudart.lib
)

	
	

del /f *.ifc *.obj

