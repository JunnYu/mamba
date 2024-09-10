cd csrc
python setup.py build

cd ..
rm -rf build dist src/*.egg-info
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall