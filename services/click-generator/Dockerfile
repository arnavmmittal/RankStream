FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY click_generator.py .

CMD ["python", "-u", "click_generator.py"]
