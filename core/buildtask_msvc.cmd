@ECHO OFF

:: This Windows CMD script builds the diagnostics executable

SET curpath=%~dp0
echo Active path:
echo %curpath:~0,-1%

ECHO Building executable

::Building core program
::cl /EHsc /c %curpath:~0,-1%\main.cpp
nvcc %curpath:~0,-1%\..\gpu\gpu_blas.cu -c -o gpu_blas.obj
nvcc %curpath:~0,-1%\..\gpu\gpu_gaussnewton.cu -c -o gpu_gaussnewton.obj
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\gpu\gpu_bindings.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\transformation.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\solvers\gaussnewton.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\gpu\gpu_gaussnewton.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\solvers\solvers.ixx
cl /EHsc /experimental:module /std:c++latest  main.cpp gpu_blas.obj gpu_gaussnewton.obj gpu_bindings.obj transformation.obj solvers.obj gaussnewton.obj /link /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64\" cudart.lib

::cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\cpu\transformation.ixx 

::Building test modules
::cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\tests.ixx
::cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\performance/test_perf_transform_cpu.ixx 
::cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\performance/performance.ixx 
::cl /EHsc /experimental:module /std:c++latest %curpath:~0,-1%\run_diagnostics.cpp %curpath:~0,-1%\tests.obj %curpath:~0,-1%\performance.obj %curpath:~0,-1%\test_perf_transform_cpu.obj  
del /f *.ifc *.obj *.exp *.lib

