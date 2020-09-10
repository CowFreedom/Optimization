@ECHO OFF

:: This Windows CMD script builds the diagnostics executable

SET curpath=%~dp0
echo Active path:
echo %curpath:~0,-1%

ECHO Building executable

::Building core program
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\..\cpu\transformation.ixx 

::Building test modules
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\tests.ixx
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\performance/test_perf_transform_cpu.ixx 
cl /EHsc /experimental:module /std:c++latest /c %curpath:~0,-1%\performance/performance.ixx 
cl /EHsc /experimental:module /std:c++latest %curpath:~0,-1%\run_diagnostics.cpp %curpath:~0,-1%\tests.obj %curpath:~0,-1%\performance.obj %curpath:~0,-1%\test_perf_transform_cpu.obj  
del /f *.ifc *.obj

