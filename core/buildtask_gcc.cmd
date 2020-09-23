@ECHO OFF

:: This Windows CMD script builds the diagnostics executable

SET curpath=%~dp0
echo Active path:
echo %curpath:~0,-1%

ECHO Building executable

::Building core program
::cl /EHsc /c %curpath:~0,-1%\main.cpp
nvcc %curpath:~0,-1%\..\gpu\gpu_blas.cu -c -o gpu_blas.obj
g++ -fmodules-ts -std=c++20  %curpath:~0,-1%\..\gpu\gpu_bindings.cc