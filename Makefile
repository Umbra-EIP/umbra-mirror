.PHONY: help install lint format type-check test audit docker-build docker-run regression clean

PYTHON      := python3
IMAGE_NAME  := umbra
IMAGE_TAG   := latest
PORT        := 8501

help:
	@echo "Umbra — available targets:"
	@echo "  install       Install runtime + dev dependencies"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run ruff formatter (in-place)"
	@echo "  type-check    Run mypy type checker"
	@echo "  test          Run pytest with coverage"
	@echo "  audit         Run pip-audit security check"
	@echo "  docker-build  Build the Docker image"
	@echo "  docker-run    Run the Streamlit dashboard in Docker"
	@echo "  regression    Run model inference regression check"
	@echo "  clean         Remove build/cache artifacts"

install:
	pip3 install -r requirements.txt -r requirements-dev.txt
	pre-commit install

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

type-check:
	mypy src/

test:
	pytest

audit:
	pip-audit -r requirements.txt --format=columns

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .ma

docker-run:
	docker run --rm -p $(PORT):8501 \
		-v "$(PWD)/src/models:/app/src/models:ro" \
		-v "$(PWD)/data:/app/data:ro" \
		$(IMAGE_NAME):$(IMAGE_TAG)

regression:
	$(PYTHON) scripts/regression_check.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f coverage.xml .coverage
