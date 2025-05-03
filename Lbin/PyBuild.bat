cd .\dist
del *.* 
cd ..
rmdir dist

cd .\%1.egg-info
del *.*
cd ..
rmdir %1.egg-info

REM py -m pip install --upgrade build
REM go to directory of package
py -m build	

REM Upload to python PyPI test
REM py -m pip install --upgrade twine

py -m twine upload --repository testpypi dist/*
REM Install 

pip uninstall %1

py -m pip install --index-url https://test.pypi.org/simple/ --no-deps --upgrade %1