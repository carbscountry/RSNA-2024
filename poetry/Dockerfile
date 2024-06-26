# 最初に構築したイメージと合わせる
FROM python:3.10-slim
WORKDIR /workspace

# お好みで好きなパッケージを追加
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN git config --global --add safe.directory /app

RUN pip install poetry \
  && poetry config virtualenvs.create false

# COPY ./pyproject.toml ./poetry.lock* ./
# RUN poetry install
