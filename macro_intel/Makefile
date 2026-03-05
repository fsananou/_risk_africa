.PHONY: install run docker-build docker-run clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

docker-build:
	docker build -t macro-intel .

docker-run:
	docker run -p 8501:8501 macro-intel

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
