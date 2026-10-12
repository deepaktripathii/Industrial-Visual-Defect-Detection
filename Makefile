.PHONY: install install-dev dataset train evaluate predict benchmark simulate api dashboard test lint docker-build docker-up docker-down

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

dataset:
	python scripts/generate_demo_dataset.py

train: dataset
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

predict:
	python scripts/predict.py data/demo_dataset/test/scratch/scratch_000.jpg --explain

benchmark:
	python scripts/benchmark.py

simulate:
	python scripts/simulate_inspections.py --n 60

api:
	uvicorn defectiq.api.main:app --reload --port 8000

dashboard:
	streamlit run src/defectiq/dashboard/app.py

test:
	python -m pytest -q

lint:
	ruff check src tests scripts

docker-build:
	docker build -t defectiq-api:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
