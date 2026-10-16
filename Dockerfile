FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data ./data
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["feedrec", "serve", "--host", "0.0.0.0"]
