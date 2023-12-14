.PHONY: test
test:
	python -m unittest discover --pattern *_test.py --failfast


.PHONY: clean
clean:
	find . -type d -name  "__pycache__" -exec rm -r {} +
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .pytype
	rm -rf .nox


.PHONY: lint
lint:
	pre-commit run --all-files
