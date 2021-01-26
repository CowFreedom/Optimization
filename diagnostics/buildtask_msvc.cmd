@ECHO OFF

:: This Windows CMD script builds the diagnostics executable

set curpath=%~dp0
set "USE_CUDA=1"
echo Active path:
echo %curpath:~0,-1%
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"
echo %cudapath%
echo Building executable



::Building core program
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\core\transformation.ixx 
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\solvers\gaussnewton.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\solvers\solvers.ixx

if defined USE_CUDA (
	::Building CUDA
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\gemm.cu -c -o gpu_gemm.obj 
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\gmv.cu -c -o gpu_gmv.obj
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\ldl.cu -c -o gpu_ldl.obj
	nvcc %curpath:~0,-1%\..\gpu\linear_algebra\lu.cu -c -o gpu_lu.obj
	nvcc %curpath:~0,-1%\..\gpu\hostgpu_bindings.cu -c -o hostgpu_bindings.obj

	::Building gpu modules
	cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\gpu\gns_hostgpu.ixx
	cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\gpu\solvers_gpu.ixx
)

::Building test modules
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\tests.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\performance/test_perf_transform.ixx 
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\performance/performance.ixx 
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\correctness/correctness_utility.ixx 
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\correctness/test_corr_gauss_newton_cpu.ixx 
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\correctness/test_corr_transformation.ixx 

if defined USE_CUDA (
	cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\correctness/test_corr_gauss_newton_gpu.ixx
	cl /Dopt_use_cuda /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\correctness/correctness.ixx 
)



if defined USE_CUDA (
	cl /EHsc /experimental:module /Dopt_use_cuda /std:c++latest %curpath:~0,-1%\run_diagnostics.cpp %curpath:~0,-1%\tests.obj %curpath:~0,-1%\performance.obj %curpath:~0,-1%\test_perf_transform.obj  %curpath:~0,-1%\correctness.obj %curpath:~0,-1%\correctness_utility.obj  %curpath:~0,-1%\test_corr_gauss_newton_cpu.obj %curpath:~0,-1%\test_corr_transformation.obj %curpath:~0,-1%\test_corr_gauss_newton_gpu.obj %curpath:~0,-1%\hostgpu_bindings.obj %curpath:~0,-1%\gns_hostgpu.obj %curpath:~0,-1%\solvers_gpu.obj  %curpath:~0,-1%\gpu_gemm.obj %curpath:~0,-1%\gpu_gmv.obj %curpath:~0,-1%\gpu_ldl.obj %curpath:~0,-1%\gpu_lu.obj /link /LIBPATH:%cudapath% cudart.lib
)

del /f *.ifc *.obj

