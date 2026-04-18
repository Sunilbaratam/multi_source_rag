# Dockerfile — RAG Pipeline API
# Week 4, Day 3.
#
# Build:  docker build -t rag-pipeline .
# Run:    docker compose up

FROM python:3.11-slim

WORKDIR /app

# install system deps for pypdf and chromadb
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY . .

# pre-download the embedding model at build time
# so the first request doesn't take 30s
RUN python3 -c "
from langchain_huggingface import HuggingFaceEmbeddings
HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print('Embedding model cached')
"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]