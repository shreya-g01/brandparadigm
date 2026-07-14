.PHONY: install install_dev test lint format clean run_api run_dashboard \
        download_datasets preprocess train_sentiment evaluate_sentiment \
        discover_topics train_topic_classifier docker_build docker_run

PYTHON ?= python3
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url $(TORCH_INDEX)
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

install_dev: install
	$(PYTHON) -m pip install -r requirements_dev.txt

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check brandparadigm api dashboard scripts tests
	black --check brandparadigm api dashboard scripts tests

format:
	ruff check --fix brandparadigm api dashboard scripts tests
	black brandparadigm api dashboard scripts tests

clean:
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist

run_api:
	uvicorn api.main:app --reload --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}

run_dashboard:
	streamlit run dashboard/Home.py

download_datasets:
	$(PYTHON) scripts/download_datasets.py --dataset all --sample-size 500

preprocess:
	$(PYTHON) scripts/run_preprocessing.py

train_sentiment:
	$(PYTHON) scripts/run_training_sentiment.py --profile smoke_test

evaluate_sentiment:
	$(PYTHON) scripts/run_evaluation.py

discover_topics:
	$(PYTHON) scripts/run_topic_discovery.py

train_topic_classifier:
	$(PYTHON) scripts/run_topic_classifier_training.py

docker_build:
	docker build -t brandparadigm .

docker_run:
	docker run --rm -p 8000:8000 -p 8501:8501 brandparadigm
