FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ranking_api.py .

# Copy model (will be mounted in production)
COPY --from=model /app/model/*.pkl /app/model/ 2>/dev/null || true
COPY --from=model /app/model/*.json /app/model/ 2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "ranking_api:app", "--host", "0.0.0.0", "--port", "8000"]
