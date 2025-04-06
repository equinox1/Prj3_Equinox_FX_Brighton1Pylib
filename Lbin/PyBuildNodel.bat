rem cd .\dist
rem del *.* 
rem cd ..
rem rmdir dist

rem cd .\%1.py.egg-info
rem del *.*
rem cd ..
rem rmdir %1.py.egg-info

REM py -m pip install --upgrade build
REM go to directory of package
py -m build

REM Upload to python PyPI test
REM py -m pip install --upgrade twine
py -m twine upload --repository testpypi dist/*

REM Install 

#pip uninstall %1.py

py -m pip install --index-url https://test.pypi.org/simple/ --no-deps --force --upgrade %1.py