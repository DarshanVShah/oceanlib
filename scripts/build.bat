@echo off
REM Build helper: oceanlib needs the MSVC toolchain environment (INCLUDE, LIB)
REM that vcvars64 sets up. Configuring or building from a shell that has not
REM run it fails with "cannot open include file: 'cstddef'" - the compiler is
REM found via its absolute path in build.ninja, but the system headers are not.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cmake --build build %*
