#!/bin/bash
# test_api.sh — quick curl tests for all endpoints
# Run: bash api/test_api.sh
# Make sure uvicorn is running first: uvicorn api.main:app --reload --port 8000

BASE="http://localhost:8000"

echo "=== GET / ==="
curl -s $BASE/ | python3 -m json.tool

echo -e "\n=== POST /ingest/url ==="
curl -s -X POST $BASE/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"}' \
  | python3 -m json.tool

echo -e "\n=== GET /sources ==="
curl -s $BASE/sources | python3 -m json.tool

echo -e "\n=== POST /query (non-streaming) ==="
curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval augmented generation?", "stream": false}' \
  | python3 -m json.tool

echo -e "\n=== POST /query (streaming) ==="
curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the limitations of RAG?", "stream": true}'

echo -e "\n\n=== POST /query (filter: web only) ==="
curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "source_type": "web", "stream": false}' \
  | python3 -m json.tool