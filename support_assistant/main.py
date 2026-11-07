import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
import httpx
from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)

# Task 2: Structured prompt template following Role-Context-Task-Format-Length skeleton,
# including an explicit negative constraint and a few-shot example.
STRUCTURED_RAG_PROMPT_TEMPLATE = """[ROLE]
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
"""


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Customer support query")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class AssistantState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved_ids: list[str]
    retrieved_chunks: list[str]
    response: dict


_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def is_mock_llm() -> bool:
    """Return True when MOCK_LLM is unset or not explicitly '0' (graded baseline)."""
    return os.getenv("MOCK_LLM", "1").strip() != "0"


def get_vector_store() -> tuple[SentenceTransformer, chromadb.Collection]:
    """Initialize local all-MiniLM-L6-v2 embedder and ChromaDB collection (ingesting docs if needed)."""
    global _embedder, _collection
    if _embedder is None:
        try:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        except Exception:
            _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        doc_files = sorted(DOCS_DIR.glob("doc_*.txt"))
        if _collection.count() != len(doc_files):
            ids = [p.stem for p in doc_files]
            documents = [p.read_text(encoding="utf-8").strip() for p in doc_files]
            metadatas = [{"source": p.name, "doc_id": p.stem} for p in doc_files]
            embeddings = _embedder.encode(documents, normalize_embeddings=True).tolist()
            _collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return _embedder, _collection


def call_real_llm(prompt: str) -> str:
    """Optional MOCK_LLM=0 helper calling Groq's free-tier OpenAI-compatible chat endpoint."""
    api_key = os.getenv("GROQ_API_KEY", "")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set for MOCK_LLM=0 mode.")
    resp = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
        timeout=20.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def validate_with_retries(base_prompt: str, fallback_sources: list[str]) -> AskResponse:
    """Task 5: Validate LLM JSON output against AskResponse schema with up to 2 corrective retries."""
    current_prompt = base_prompt
    for attempt in range(3):  # initial attempt + up to 2 retries
        try:
            raw_text = call_real_llm(current_prompt)
            start, end = raw_text.find("{"), raw_text.rfind("}")
            json_str = raw_text[start : end + 1] if (start != -1 and end != -1) else raw_text.strip()
            return AskResponse.model_validate(json.loads(json_str))
        except (ValidationError, json.JSONDecodeError, Exception) as exc:
            current_prompt = (
                f"{base_prompt}\n\n[CORRECTION RETRY {attempt + 1}/2]\n"
                f"Your previous output failed Pydantic validation with error: {exc}. "
                'Return ONLY raw valid JSON matching {"answer": str, "sources": list[str], "confidence": float}.'
            )
    return AskResponse(
        answer="Error: LLM output failed schema validation after 2 corrective retries.",
        sources=fallback_sources,
        confidence=0.0,
    )


# --- Task 3: LangGraph Nodes ---
def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"]
    q_lower = query.lower()
    heuristic_intent: Literal["policy_question", "general_question"] = (
        "policy_question" if any(kw in q_lower for kw in POLICY_KEYWORDS) else "general_question"
    )
    if is_mock_llm():
        return {"intent": heuristic_intent}

    # Optional MOCK_LLM=0 branch with graceful fallback to keyword heuristic on network error
    cls_prompt = (
        "Classify the following customer query into exactly one label: "
        "'policy_question' (if about Zepto delivery, returns, refunds, membership, tracking, cancellation, "
        "damaged items, gift cards, or support hours) or 'general_question'.\n"
        f"Query: {query}\nLabel:"
    )
    try:
        label = call_real_llm(cls_prompt).lower()
        return {"intent": "policy_question" if "policy_question" in label else "general_question"}
    except Exception:
        return {"intent": heuristic_intent}


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    query = state["query"]
    embedder, collection = get_vector_store()

    # Retrieval always runs for real in both mock and real-LLM modes
    query_vec = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=3)

    retrieved_ids: list[str] = results["ids"][0] if results["ids"] else []
    retrieved_chunks: list[str] = results["documents"][0] if results["documents"] else []

    if is_mock_llm():
        top_chunk_snippet = retrieved_chunks[0][:200] if retrieved_chunks else ""
        validated = AskResponse(
            answer=f"Based on the retrieved context: {top_chunk_snippet}",
            sources=retrieved_ids,
            confidence=1.0,
        )
        return {
            "retrieved_ids": retrieved_ids,
            "retrieved_chunks": retrieved_chunks,
            "response": validated.model_dump(),
        }

    # Optional MOCK_LLM=0 branch
    context_str = "\n".join(f"[{doc_id}] {chunk}" for doc_id, chunk in zip(retrieved_ids, retrieved_chunks))
    prompt = STRUCTURED_RAG_PROMPT_TEMPLATE.format(context=context_str, query=query)
    validated = validate_with_retries(prompt, fallback_sources=retrieved_ids)
    return {
        "retrieved_ids": retrieved_ids,
        "retrieved_chunks": retrieved_chunks,
        "response": validated.model_dump(),
    }


def direct_answer(state: AssistantState) -> AssistantState:
    query = state["query"]
    if is_mock_llm():
        validated = AskResponse(
            answer="I can only answer questions about Zepto policies right now.",
            sources=[],
            confidence=1.0,
        )
        return {"retrieved_ids": [], "retrieved_chunks": [], "response": validated.model_dump()}

    # Optional MOCK_LLM=0 branch
    direct_prompt = (
        "You are Zepto's assistant. Answer the general question briefly in JSON matching "
        '{"answer": "<string>", "sources": [], "confidence": <float 0-1>}.\n'
        f"Question: {query}"
    )
    validated = validate_with_retries(direct_prompt, fallback_sources=[])
    validated.sources = []
    return {"retrieved_ids": [], "retrieved_chunks": [], "response": validated.model_dump()}


# --- Task 4: Conditional Router & StateGraph Construction ---
def route_by_intent(state: AssistantState) -> Literal["retrieve_and_answer", "direct_answer"]:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_support_graph():
    builder = StateGraph(AssistantState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_and_answer", retrieve_and_answer)
    builder.add_node("direct_answer", direct_answer)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges("classify_intent", route_by_intent)
    builder.add_edge("retrieve_and_answer", END)
    builder.add_edge("direct_answer", END)
    return builder.compile()


support_graph = build_support_graph()


# --- Task 6: FastAPI Application ---
app = FastAPI(
    title="Zepto Policy Support Assistant",
    description="LangGraph + ChromaDB RAG service with deterministic MOCK_LLM baseline.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    _, col = get_vector_store()
    return {"status": "ok", "indexed_documents": col.count(), "mock_llm": is_mock_llm()}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
    final_state = support_graph.invoke({"query": payload.query})
    return AskResponse.model_validate(final_state["response"])


if __name__ == "__main__":
    # Self-check demonstrating both policy_question and general_question paths
    print("Testing policy_question routing & retrieval...")
    res_policy = ask_endpoint(AskRequest(query="What is the delivery fee for orders below INR 149?"))
    print("Policy Call Response:\n", json.dumps(res_policy.model_dump(), indent=2))

    print("\nTesting general_question routing...")
    res_general = ask_endpoint(AskRequest(query="Who won the Cricket World Cup in 2023?"))
    print("General Call Response:\n", json.dumps(res_general.model_dump(), indent=2))

    assert res_policy.sources[0] == "doc_01"
    assert res_policy.answer.startswith("Based on the retrieved context:")
    assert res_general.sources == []
    assert res_general.answer == "I can only answer questions about Zepto policies right now."
    print("\n[SUCCESS] All support_assistant self-checks passed!")
