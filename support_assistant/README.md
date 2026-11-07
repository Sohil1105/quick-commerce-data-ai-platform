# Module 3: GenAI Support Assistant (`/support_assistant`)

## Overview
Grounded Retrieval-Augmented Generation (RAG) customer support service built for Zepto's delivery, return, refund, membership, tracking, cancellation, gift card, and support hours policies.

- **Document Corpus**: 8 official Zepto policy documents (`support_assistant/docs/doc_01.txt` … `doc_08.txt`).
- **Embeddings & Vector Store**: Local `sentence-transformers` (`all-MiniLM-L6-v2`) embeddings stored in a persistent local `ChromaDB` collection (`zepto_policies`) using cosine similarity.
- **Orchestration**: `LangGraph` `StateGraph` with `TypedDict` state (`AssistantState`), 3 nodes (`classify_intent`, `retrieve_and_answer`, `direct_answer`), and a conditional intent router (`route_by_intent`).
- **Structured Validation & API**: `Pydantic` response validation (`AskResponse` with `answer`, `sources`, `confidence`) served via `FastAPI` (`POST /ask`) and containerized with `Docker`.

---

## Setup & Local Execution

### 1. Run Directly with Uvicorn (Graded Baseline: `MOCK_LLM` Default)
From the repository root:
```bash
pip install -r requirements.txt
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```
Or run the built-in self-check directly:
```bash
python support_assistant/main.py
```

### 2. Build & Run Locally with Docker (Graded Containerization Baseline)
```bash
cd support_assistant
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```
Once running, query `http://localhost:7860/ask` via `POST`.

---

## Example Call Transcripts (`MOCK_LLM` at Default)

### Example 1: Policy Question (Routes to `retrieve_and_answer`)
**Request (`POST /ask`)**:
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the delivery fee for orders below INR 149?"}'
```
**Raw JSON Response (`200 OK`)**:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_03"
  ],
  "confidence": 1.0
}
```

### Example 2: General Question (Routes to `direct_answer`)
**Request (`POST /ask`)**:
```bash
curl -X POST http://localhost:7860/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who won the Cricket World Cup in 2023?"}'
```
**Raw JSON Response (`200 OK`)**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Structured Prompt Template (Role–Context–Task–Format–Length Skeleton)
Defined in `support_assistant/main.py` as `STRUCTURED_RAG_PROMPT_TEMPLATE` (used when `MOCK_LLM=0`):

```text
[ROLE]
You are Zepto's official AI Customer Support Assistant specializing in company policies.

[CONTEXT]
Retrieved Zepto Policy Documents:
{context}

[TASK]
Answer the customer's question accurately using ONLY the facts stated in the [CONTEXT] section above.
NEGATIVE CONSTRAINT: Do NOT answer using information not present in the provided context, and never invent or extrapolate prices, timelines, or policy rules.

[FORMAT]
Respond with a valid JSON object matching this exact schema:
{{"answer": "<string>", "sources": ["<doc_id>", ...], "confidence": <float between 0.0 and 1.0>}}

[LENGTH]
Keep the "answer" field concise: 1 to 3 sentences (maximum 60 words).

[FEW-SHOT EXAMPLE]
Customer Question: "What is the delivery fee for orders below INR 149?"
Retrieved Context: "[doc_01] Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee."
Assistant JSON Output:
{{"answer": "Orders below INR 149 incur a flat INR 25 standard delivery fee, while orders over INR 149 qualify for free standard delivery.", "sources": ["doc_01"], "confidence": 0.98}}

[CURRENT REQUEST]
Customer Question: "{query}"
Assistant JSON Output:
```

---

## RAG Pipeline Architecture Description

```text
[support_assistant/docs/doc_01..08.txt]
                 │
                 ▼  (1. Ingestion & Chunking: get_vector_store() in main.py)
      [Per-Document Text Chunks]
                 │
                 ▼  (2. Embedding: SentenceTransformer("all-MiniLM-L6-v2"))
   [ChromaDB Collection: "zepto_policies" (cosine space)]
                 │
                 ▼
POST /ask ──► [classify_intent] (Keyword heuristic in MOCK_LLM=1; LLM classification in MOCK_LLM=0)
                 │
        ┌────────┴────────┐ (Conditional Edge: route_by_intent)
        ▼                 ▼
(policy_question)   (general_question)
        │                 │
        ▼                 ▼
[retrieve_and_answer]  [direct_answer]
  • Embeds query         • Returns fixed canned refusal (MOCK_LLM=1)
  • Retrieves top-3        or direct LLM JSON answer (MOCK_LLM=0)
    chunks from ChromaDB
  • Generates answer
        │                 │
        └────────┬────────┘
                 ▼
      [AskResponse Pydantic Validation]
      {"answer": str, "sources": list[str], "confidence": float}
```

### Stage-by-Stage Component Walkthrough
1. **Ingestion (`get_vector_store()` in `support_assistant/main.py`)**:
   Reads all 8 policy files (`support_assistant/docs/doc_01.txt` through `doc_08.txt`). Because each policy document is a self-contained single paragraph (~50–90 words), each document is ingested as one semantic chunk keyed by its document ID (`doc_01` … `doc_08`).
2. **Embedding (`get_vector_store()` in `support_assistant/main.py`)**:
   Uses the local open-source `SentenceTransformer("all-MiniLM-L6-v2")` model to encode the 8 chunks into normalized 384-dimensional dense vectors and upserts them into a persistent `ChromaDB` collection named `zepto_policies` (`metadata={"hnsw:space": "cosine"}`) stored under `support_assistant/chroma_db/`.
3. **Intent Classification & Conditional Routing (`classify_intent` & `route_by_intent` in `support_assistant/main.py`)**:
   Incoming requests hit `POST /ask` (`ask_endpoint`), which invokes the compiled LangGraph `support_graph`. The entry node `classify_intent` inspects the query and writes `intent` (`"policy_question"` or `"general_question"`) into `AssistantState`. The conditional edge `route_by_intent` then routes `"policy_question"` to `retrieve_and_answer` and `"general_question"` to `direct_answer`.
4. **Retrieval (`retrieve_and_answer` node in `support_assistant/main.py`)**:
   For every `"policy_question"`, `retrieve_and_answer` encodes the user's query with `all-MiniLM-L6-v2` and queries the `zepto_policies` ChromaDB collection (`collection.query(query_embeddings=..., n_results=3)`) to retrieve the top-3 most similar policy chunks and their document IDs (`retrieved_ids`). **Retrieval always executes for real in both mock and real-LLM modes.**
5. **Generation & Schema Validation (`retrieve_and_answer`, `direct_answer`, and `validate_with_retries` in `support_assistant/main.py`)**:
   Produces the final response and validates it against the `AskResponse` Pydantic model (`answer: str`, `sources: list[str]`, `confidence: float`).

### How `MOCK_LLM` Branches Across Stages
- **Stages that DO NOT branch on `MOCK_LLM`**: Document ingestion, `all-MiniLM-L6-v2` embedding, conditional graph routing (`route_by_intent`), and ChromaDB top-3 cosine retrieval all run identically regardless of `MOCK_LLM`.
- **Stages that DO branch on `MOCK_LLM`** (via `is_mock_llm()` checking `os.getenv("MOCK_LLM", "1") != "0"`):
  1. **`classify_intent` node**:
     - **Default (`MOCK_LLM` unset or `1`)**: Uses the deterministic keyword heuristic (`"delivery"`, `"return"`, `"refund"`, `"membership"`, `"tracking"`, `"cancel"`, `"gift card"`, `"support hours"`). Makes zero network or LLM calls.
     - **Optional Real-LLM (`MOCK_LLM=0`)**: Calls `call_real_llm()` (Groq API) to classify the query into `policy_question` or `general_question`.
  2. **`retrieve_and_answer` generation step**:
     - **Default (`MOCK_LLM` unset or `1`)**: Formats the top retrieved chunk deterministically as `f"Based on the retrieved context: {top_chunk_snippet}"` (first 200 characters of the top chunk), populates `sources` with the top-3 retrieved document IDs, and sets `confidence = 1.0`.
     - **Optional Real-LLM (`MOCK_LLM=0`)**: Formats `STRUCTURED_RAG_PROMPT_TEMPLATE` with the retrieved chunks and calls `validate_with_retries()`, which invokes `call_real_llm()` and retries up to 2 additional times with a corrective prompt if Pydantic `AskResponse` validation fails.
  3. **`direct_answer` generation step**:
     - **Default (`MOCK_LLM` unset or `1`)**: Returns the deterministic canned string `"I can only answer questions about Zepto policies right now."` with `sources = []` and `confidence = 1.0`.
     - **Optional Real-LLM (`MOCK_LLM=0`)**: Prompts the real LLM directly without retrieval and validates the JSON response via `validate_with_retries()`.
