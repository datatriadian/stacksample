FROM python:3.10.2-slim-bullseye

RUN useradd --create-home appuser

WORKDIR /home/appuser

RUN apt-get update && apt-get install --no-install-recommends curl build-essential -y

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 - && \
  cd /usr/local/bin && \
  ln -s /etc/poetry/bin/poetry

COPY ./pyproject.toml ./

COPY ./models ./models

COPY ./app ./app

RUN chown -R appuser:appuser /home/appuser

USER appuser

RUN poetry run python -m pip install -U pip

RUN poetry install --no-root --no-dev

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
