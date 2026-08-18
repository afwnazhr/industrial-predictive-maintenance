FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]