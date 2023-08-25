.ONESHELL:
.PHONY: clean dev install test  build publish  docs

default: dev

BUILD			:= ${BUILD_NUMBER}
BRANCH			!= git rev-parse --abbrev-ref HEAD
VERSION			!= cat VERSION | sed -e 's/[-+].*//'

# if not in master branch, add 'rc' and build number to version
ifneq (${BRANCH},master)
POSTFIX         != echo ${BRANCH}${BUILD} | sed -e 's/[-+_:=\/]/./g'
VERSION			:= ${VERSION}-rc+${POSTFIX}
endif



check_variable:
ifeq (${BRANCH},master)
	@echo ${BUILD}
	@echo ${BRANCH}
	@echo ${VERSION}
endif

setup:
	python -m pip install --upgrade pip
	python -m pip install --upgrade twine setuptools==57.5.0 wheel
# Clean up interem/temporary build artifacts
clean:
	rm -rf **/*.egg-info
	rm -rf build
	rm -rf dist
	rm -rf docs/_build
	rm -rf .coverage
	rm -rf .ipynb_checkpoint
	rm -rf __pycache__
	rm -f testresults.xml
	rm -rf result deep_link pickles
	rm -rf chat_logs/*
	rm -rf gpt3_logs/*
	rm -rf notes/*


# Install python requirements (see setup.cfg)
install:
	pip install .

# Install python requirements in dev mode
dev:
	pip install --user -e .[dev]

# run tests (with code coverage)
test: 
	pip install --user .[test]
	mypy src/project_name_goes_here --ignore-missing-imports
	# comment for now 
	# python setup.py test
syntax: 
	mypy src/project_name_goes_here --ignore-missing-imports


version:
	@echo "__version__ = ${VERSION}" 
	@echo "__version__ = '${VERSION}' # type: ignore" >> src/project_name_goes_here/__init__.py  
	@echo ""
	@cat src/project_name_goes_here/__init__.py


# build project artifacts
build: clean setup 
	@echo ${BUILD}
	@echo ${BRANCH}
	@echo ${VERSION}
	python setup.py bdist_wheel

docs:
	pip install --user .[docs]
	sphinx-build -b html docs docs/_build
# publish artifacts
# need dynamically generated repository feed 
publish:
ifeq (${BRANCH},master)
	@echo $(ls -la dist)
	python -m twine upload --repository "GithubFeed" --config-file $(PYPIRC_PATH) dist/*.whl --verbose
else
	@echo "No publish from non master branch"
	@echo {BRANCH}
endif

# later will build every branch 
# ifeq (${BRANCH},master)
# 	python -m twine upload --repository "testfeed" --config-file $(PYPIRC_PATH) dist/*.whl
# else:
# 	@echo "No publish from non master branch"
# endif

activate:
	source ./venv/Scripts/activate
	pip install -r requirements.txt

venv:
	python -m venv venv

commit:
	git add .
	git commit -m "bump version to ${VERSION}, ${msg}"
	git push

jlist:
	jupyter kernelspec list

jremove:
	jupyter kernelspec remove -f ${venv}

start:
	source venv/Scripts/activate
	python app.py
start_debug:
	source venv/Scripts/activate
	flask --app app run --debug

