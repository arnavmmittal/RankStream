FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_pipeline.py .

CMD ["python", "-u", "feature_pipeline.py"]
