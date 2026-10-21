PYTHON := .venv/bin/python
export PYTHONPATH := src

.PHONY: setup server server-cache down clean test evaluate

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

server:
	$(PYTHON) src/lab_server.py --mode on-demand

server-cache:
	$(PYTHON) src/lab_server.py --mode precompute

down:
	@for pid in $$(lsof -tiTCP:8000 -sTCP:LISTEN); do \
		[ "$$(lsof -a -p "$$pid" -d cwd -Fn | sed -n 's/^n//p')" = "$(CURDIR)" ] || continue; \
		case "$$(ps -p "$$pid" -o args=)" in *" src/lab_server.py --mode "*) kill "$$pid" ;; esac; \
	done

clean:
	find dataset models -type f ! -name .gitkeep -delete
	rm -rf models/cache models/history reports .ruff_cache __pycache__ src/__pycache__ scripts/__pycache__ tests/__pycache__

test:
	$(PYTHON) -m unittest discover -s tests -v

evaluate:
	$(PYTHON) scripts/evaluate_model.py $(ARGS)
