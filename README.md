# 🍽️ Le Grand Bistro — Strict Closed-Domain RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers questions **exclusively** from the `grand_bistro_culinary_recipe_manual_50p.pdf`. This is a closed-domain system — it will **never** use external knowledge, web search, or the LLM's pretrained knowledge to supplement answers.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│          Guardrails Layer               │
│  • Prompt Injection Detection           │
│  • Query Sanitization                   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│       Hybrid Retrieval Engine           │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Dense Search │  │ BM25 Sparse  │    │
│  │ (ChromaDB +  │  │ Keyword      │    │
│  │  Gemini Emb) │  │ Search       │    │
│  └──────┬───────┘  └──────┬───────┘    │
│         └────────┬─────────┘            │
│                  ▼                      │
│    Reciprocal Rank Fusion (RRF)         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│        Scope Evaluation                 │
│  Lexical overlap check → abstain early  │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│     Constrained LLM Generation          │
│   (Gemini, temperature=0, strict        │
│    system prompt, PDF-only context)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    Grounding & Citation Validation      │
│  • Citation extraction [Page X]         │
│  • Fabricated citation detection        │
│  • Abstention if validation fails       │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Bistro/
├── app/
│   ├── __init__.py
│   ├── api.py              # FastAPI REST API endpoints
│   ├── config.py           # Settings, constants, abstention message
│   ├── guardrails.py       # Injection defense, citation validator, scope checker
│   ├── pdf_parser.py       # PDF → DocumentChunk extractor (page-level)
│   ├── rag_engine.py       # Orchestrates retrieval + LLM generation
│   └── retriever.py        # HybridRetriever: Dense + BM25 + RRF
├── tests/
│   ├── test_api.py         # FastAPI endpoint integration tests
│   └── test_rag_strictness.py  # Core RAG compliance & strictness tests
├── ui/
│   └── app.py              # Streamlit interactive UI
├── grand_bistro_culinary_recipe_manual_50p.pdf  # The ONLY knowledge source
├── requirements.txt
├── .env.example
└── README.md
```

---

## ✅ Key Features

| Feature | Implementation |
|---|---|
| **Dense Retrieval** | ChromaDB + `models/gemini-embedding-001` |
| **Sparse Retrieval** | BM25Okapi (rank-bm25) |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Abstention** | Lexical scope check + LLM-level + post-generation validation |
| **Citation Enforcement** | Every answer must cite `[Page X]` from retrieved pages only |
| **Prompt Injection Defense** | Regex pattern matching on 8+ injection templates |
| **No External Search** | Zero network calls for retrieval — PDF-only |
| **Numerical Fidelity** | Exact values preserved (e.g. `57.5°C`, `110 minutes`, `850g`) |

---

## 🚀 Setup & Usage

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run the FastAPI Server

```bash
uvicorn app.api:app --reload --port 8000
# API docs at: http://localhost:8000/docs
```

### 4. Run the Streamlit UI

```bash
streamlit run ui/app.py --server.port 8501
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🔌 API Reference

### `POST /query`

```json
{
  "query": "What are the ingredients for the Signature Truffle Ganache?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "...",
  "answer": "According to [Page 10], the Signature Truffle Ganache requires ...",
  "citations": [10],
  "is_abstained": false,
  "flagged_injection": false,
  "latency_seconds": 1.23,
  "retrieved_chunks": [...]
}
```

### `GET /page/{page_num}`
Retrieve raw PDF text for any page (audit trail).

### `GET /health`
Check system health, ChromaDB count, BM25 index status.

---

## 🧪 Test Coverage

| Test | Description |
|---|---|
| `test_in_scope_recipe_ganache_exact_numerical` | Verifies Page 10 retrieval with exact quantities (850g, 620ml, 90g, 15ml, 70%) |
| `test_in_scope_allergen_matrix_table` | Verifies Page 25 tree-nut allergen data |
| `test_in_scope_sous_vide_haccp_pasteurization` | Verifies exact 57.5°C / 110 min from Page 40 |
| `test_out_of_scope_general_knowledge_rejection` | Rejects capitals, presidents, astronomy |
| `test_out_of_scope_culinary_knowledge_rejection` | Rejects beef wellington, mayo, soft-boiled eggs |
| `test_prompt_injection_defense` | Neutralizes jailbreak attempts |
| `test_citation_validator_catches_fabrication` | Guardrails reject fabricated [Page 99] citations |
| `test_hybrid_rrf_scoring` | Validates RRF formula: `1/(k+dr) + 1/(k+sr)` |
| `test_no_external_search_network_calls` | Mocks `requests.get` → asserts 0 calls |

---

## 🛡️ Strict Abstention

If the PDF does not contain the requested information, the system **always** responds:

> *"I couldn't find this information in the provided PDF. I can only answer questions based on the contents of the Grand Bistro Culinary Recipe Manual."*

This applies even if the answer is common knowledge, the LLM knows it, or the question is about a famous topic not in the manual.

---

## 📊 Special Pages

| Page | Content |
|---|---|
| **Page 10** | Recipe Card 2.4 — Signature Truffle Ganache (850g Valrhona 70%, 620ml cream, 90g butter, 15ml truffle oil) |
| **Page 25** | Section 5.1 — Autumn Tasting Menu Allergen Matrix (tree nuts, sesame, shellfish) |
| **Page 40** | Section 7.2 — HACCP Food Safety & Sous-Vide Duck Breast (57.5°C core temp, 110 min hold) |

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key (required) |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for dense retrieval |
| `LLM_MODEL` | `gemini-2.0-flash` | LLM for constrained generation |
| `TOP_K` | `5` | Passages retrieved per query |
| `RRF_K` | `60` | RRF fusion constant |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence path |

---

## ⚠️ Important Notes

- The `chroma_db/` directory is auto-populated on first run (embeddings are generated via Gemini API).
- The `.env` file is excluded from version control — never commit your API key.
- The system uses `temperature=0.0` for fully deterministic LLM generation.
- Model fallback chain ensures availability even during API rate limits.
