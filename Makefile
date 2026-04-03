.PHONY: help install lint format type-check test audit docker-build docker-run regression benchmark clean train-eeg2emg-single

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
	@echo "  regression    Run model accuracy regression gate"
	@echo "  benchmark     Run full model benchmark and save report"
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

benchmark:
	$(PYTHON) scripts/run_comparison.py --dataset-id 1

# Single-subject / high-accuracy EEG→EMG (group-aware val split, cosine LR, AdamW)
train-eeg2emg-single:
	$(PYTHON) -m src.eeg_emg.eeg2emg_train_single_subject \
		--data_path data/eeg_emg/dataset_augmented.npz \
		--normalize \
		--save_path src/eeg_emg/eeg2emg_single_subject_best.pth

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f coverage.xml .coverage
