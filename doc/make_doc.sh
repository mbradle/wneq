rm -f source/wneq.*.rst
mkdir -p source/_static source/_templates
sphinx-apidoc -M -f -n -o source ../wneq
make html
