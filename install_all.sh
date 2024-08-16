rm -rf build dist
python setup_causal_conv1d.py install
rm -rf build dist
python setup_selective_scan.py install
rm -rf build dist
python setup.py bdist_wheel
pip install dist/*.whl
