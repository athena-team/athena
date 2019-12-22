
rm -rf build/  athena.egg-info/ dist/
python setup.py bdist_wheel sdist
python -m pip install --ignore-installed dist/athena-0.1.0*.whl
