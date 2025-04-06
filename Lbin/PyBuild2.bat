@echo off
setlocal

:: Ensure package name is passed as an argument
if "%1"=="" (
    echo Package name required as argument.
    exit /b 1
)

:: Remove old build artifacts
cd /d %~dp0
if exist dist (
    rmdir /S /Q dist
)
if exist %1.egg-info (
    rmdir /S /Q %1.egg-info
)

:: Build package
py -m build

:: Upload to TestPyPI
py -m twine upload --repository testpypi dist/*

:: Uninstall old version
pip uninstall -y %1

:: Install from TestPyPI
py -m pip install --index-url https://test.pypi.org/simple/ --no-deps %1
