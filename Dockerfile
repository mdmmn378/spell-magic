FROM python:3.11.7-slim
WORKDIR /app
RUN apt update && apt install dumb-init

COPY poetry.lock pyproject.toml poetry.toml  ./
RUN pip install poetry && poetry install --no-root

COPY . .
RUN dvc pull

ENTRYPOINT ["/usr/bin/dumb-init", "--", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
