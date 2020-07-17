PYTHON_EXECUTABLE ?= /usr/bin/python3

.PHONY: doc

all: clean src wheel

clean:
	$(PYTHON_EXECUTABLE) setup.py clean
	rm -rf dist build *.egg-info

doc: notebooks flake8
	rm -rf build/sphinx doc/build .hypothesis
	$(PYTHON_EXECUTABLE) -m sphinx -j auto -b html doc build/sphinx/html

view_doc:
	x-www-browser build/sphinx/html/index.html

doc_autobuild:
	$(PYTHON_EXECUTABLE) -m sphinx_autobuild --open-browser --watch coax --ignore *.tfevents.* -b html doc build/sphinx/html

sync: intersphinx get_pylintrc

intersphinx:
	mkdir -p doc/_intersphinx
	wget -O doc/_intersphinx/python3.inv https://docs.python.org/3/objects.inv
	wget -O doc/_intersphinx/numpy.inv https://docs.scipy.org/doc/numpy/objects.inv
	wget -O doc/_intersphinx/sklearn.inv https://scikit-learn.org/stable/objects.inv
	wget -O doc/_intersphinx/jax.inv https://jax.readthedocs.io/en/latest/objects.inv
	wget -O doc/_intersphinx/haiku.inv https://dm-haiku.readthedocs.io/en/latest/objects.inv
	wget -O doc/_intersphinx/rllib.inv https://docs.ray.io/en/latest/objects.inv
	wget -O doc/_intersphinx/spinup.inv https://spinningup.openai.com/en/latest/objects.inv
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/python3.inv > doc/_intersphinx/python3.txt
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/numpy.inv > doc/_intersphinx/numpy.txt
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/sklearn.inv > doc/_intersphinx/sklearn.txt
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/jax.inv > doc/_intersphinx/jax.txt
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/haiku.inv > doc/_intersphinx/haiku.txt
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/rllib.inv > doc/_intersphinx/rllib.txt
	$(PYTHON_EXECUTABLE) -m sphinx.ext.intersphinx doc/_intersphinx/spinup.inv > doc/_intersphinx/spinup.txt

get_pylintrc:
	wget -O .pylintrc https://raw.githubusercontent.com/google/jax/master/pylintrc

src:
	$(PYTHON_EXECUTABLE) setup.py sdist

wheel:
	$(PYTHON_EXECUTABLE) setup.py bdist_wheel

install:
	$(PYTHON_EXECUTABLE) -m pip install .

upload: all
	$(PYTHON_EXECUTABLE) -m twine upload dist/*

pylint:
	$(PYTHON_EXECUTABLE) -m pylint --rcfile=.pylintrc coax

flake8:
	$(PYTHON_EXECUTABLE) -m flake8 coax

test: test_cpu test_gpu

test_cpu: flake8
	JAX_PLATFORM_NAME=cpu $(PYTHON_EXECUTABLE) -m pytest coax -v

test_gpu: flake8
	JAX_PLATFORM_NAME=gpu $(PYTHON_EXECUTABLE) -m pytest coax -v

notebooks:
	$(PYTHON_EXECUTABLE) ./doc/create_notebooks.py

install_dev: install_requirements
	$(PYTHON_EXECUTABLE) -m pip install -e .

install_requirements:
	for r in requirements.txt requirements.dev.txt doc/requirements.txt; do $(PYTHON_EXECUTABLE) -m pip install -r $$r; done

upgrade_requirements:
	for r in requirements.txt requirements.dev.txt doc/requirements.txt; do $(PYTHON_EXECUTABLE) -m pur -r $$r; $(PYTHON_EXECUTABLE) -m pip install -r $$r; done

rm_pycache:
	find -regex '.*__pycache__[^/]*' -type d -exec rm -rf '{}' \;
