FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MOCK_LLM=1

RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn>=0.29.0 \
    pydantic>=2.6.0 \
    chromadb>=0.5.0 \
    sentence-transformers>=2.7.0 \
    langgraph>=0.2.0 \
    httpx>=0.27.0

COPY docs/ ./docs/
COPY main.py ./main.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
