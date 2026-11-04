.PHONY: install models test example clean

install:
	pip install -r requirements.txt

models:
	bash scripts/download_models.sh

test:
	python -m pytest -q tests/

example:
	python scripts/optimize.py --task protein_canya --start 1

clean:
	rm -rf designs results __pycache__ */__pycache__ .pytest_cache
