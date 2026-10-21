PYTHON := .venv/bin/python

.PHONY: setup download-model run

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

server:
	$(PYTHON) lab_server.py

down:
	lsof -ti :8000 | xargs kill

jupyter:
	$(PYTHON) -m notebook training.ipynb