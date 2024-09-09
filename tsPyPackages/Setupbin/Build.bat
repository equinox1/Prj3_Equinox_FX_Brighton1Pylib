#py -m pip install --upgrade build
# go to directory of package
py -m build

# Upload to python PyPI test
py -m pip install --upgrade twine
py -m twine upload --repository testpypi dist/*

# Install 

py -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-shepa