.PHONY: install run format clean

install:
	pip install -e .
	pip install -r requirements.txt

run:
	python scripts/run_demo.py

format:
	black src scripts tests
	ruff check --fix src scripts tests

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
