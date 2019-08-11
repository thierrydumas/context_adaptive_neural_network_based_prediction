@echo off
setlocal
set PROJECT_ROOT=%cd%

REM The current directory is now the directory containing "tensorflow.sln".
cd %PROJECT_ROOT%\tensorflow\contrib\cmake\build

REM The maximum number of simultaneous processes is 2.
msbuild.exe tensorflow.sln ^
-t:Build ^
-p:Configuration=Release;Platform=x64;PreferredToolArchitecture=x64 ^
-maxcpucount:2


