.PHONY: test
SHELL := /bin/bash

VENV ?= venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip
SQLFLUFF = $(VENV)/bin/sqlfluff
RUFF = $(VENV)/bin/ruff

# Install all needed requirements to run the project
# Well, that is actually not true - because this is only needed for linting
$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip 
	$(PIP) install -r requirements.txt

# alias for installing
$(VENV): $(VENV)/bin/activate
install: $(VENV)

# start jupyter notebook
jupyter: $(VENV)
	jupyter notebook main.ipynb

# same as above but for the python files
ruff: $(VENV)
	$(RUFF) check

# lint entire project
lint: $(VENV) ruff

# clean up the project
clean:
	rm -rf __pycache__
	rm -rf $(VENV)