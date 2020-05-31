@echo off
setlocal
set PROJECT_ROOT=%cd%

REM The compiler environment is set.
cd "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build"
call vcvarsall.bat amd64

REM The current directory is now the directory containing the CMake contribution.
cd %PROJECT_ROOT%\tensorflow\contrib\cmake

REM `rd` removes a directory.
REM The option /S removes all files and subdirectories in
REM the directory to be removed.
IF EXIST build rd /S /Q build
mkdir build
cd build

REM The file "CMakeLists.txt" is located in the parent directory
REM of the current directory.
REM `-G` specifies a build generator system.
REM The CC unit tests will not be built as, in the file
REM "%PROJECT_ROOT%\tensorflow\contrib\cmake\CMakeLists.txt",
REM `tensorflow_BUILD_CC_TESTS` is set to OFF by default.
cmake .. -A x64 ^
-DCMAKE_BUILD_TYPE=Release ^
-Dtensorflow_BUILD_PYTHON_BINDINGS=OFF ^
-Dtensorflow_BUILD_CC_EXAMPLE=OFF ^
-Dtensorflow_DISABLE_EIGEN_FORCEINLINE=ON ^
-DCMAKE_CXX_FLAGS_DEBUG="/MTd /Zi /Ob0 /Od /RTC1" ^
-DCMAKE_CXX_FLAGS_MINSIZEREL="/MT /O1 /Ob1 /DNDEBUG" ^
-DCMAKE_CXX_FLAGS_RELEASE="/MT /O2 /Ob2 /DNDEBUG" ^
-DCMAKE_CXX_FLAGS_RELWITHDEBINFO="/MT /Zi /O2 /Ob1 /DNDEBUG" ^
-DCMAKE_C_FLAGS_DEBUG="/MTd /Zi /Ob0 /Od /RTC1" ^
-DCMAKE_C_FLAGS_MINSIZEREL="/MT /O1 /Ob1 /DNDEBUG" ^
-DCMAKE_C_FLAGS_RELEASE="/MT /O2 /Ob2 /DNDEBUG" ^
-DCMAKE_C_FLAGS_RELWITHDEBINFO="/MT /Zi /O2 /Ob1 /DNDEBUG"


