# 🚀 Production-Grade Multimodal RAG System

An enterprise-ready, defensive Retrieval-Augmented Generation (RAG) architecture engineered to minimize hallucinations, provide complete observability, and verify contextual claims before presenting responses.

The system combines layout-aware document ingestion with a high-performance hybrid retrieval pipeline, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, Langfuse observability, and an automated LLM-as-a-Judge evaluation framework.

---

# 🏗️ Core Architectural Features

- **📄 Layout-Aware Ingestion**
  - Powered by **LlamaParse**
  - Preserves document structure, tables, headings, and layout information
  - Generates high-quality semantic chunks

- **🔍 Hybrid Retrieval**
  - Dense semantic search using **Qdrant**
  - Sparse lexical retrieval using **BM25**
  - Parallel retrieval for maximum recall

- **⚡ Rank Fusion & Reranking**
  - Reciprocal Rank Fusion (RRF)
  - Cross-Encoder reranking
  - Returns only the highest-confidence context

- **🛡️ Defensive Guardrails**
  - Dynamic confidence thresholding
  - Graceful degradation
  - Prevents hallucinations on low-confidence retrieval
  - Saves LLM API costs

- **✅ Citation Verification**
  - Embedding-based verification
  - Cosine similarity validation
  - Detects unsupported claims

- **📊 OpenTelemetry Observability**
  - Langfuse tracing
  - Prompt tracking
  - Token usage
  - Latency monitoring
  - End-to-end execution trees

- **🧪 Automated Evaluation**
  - Faithfulness
  - Answer Relevance
  - Context Recall
  - Offline regression testing

---

# 📂 Project Structure

```text
multimodal-rag-system/
│
├── data/
│   ├── eval_dataset.json
│   └── eval_report.md
│
├── src/
│   ├── ingestion/
│   │   ├── parser.py
│   │   └── chunker.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   └── fusion.py
│   │
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── utils/
│   │   └── telemetry.py
│   │
│   ├── config.py
│   └── main_pipeline.py
│
├── .env
├── .gitignore
├── app.py
├── main.py
├── requirements.txt
│
├── test_retrieval.py
├── test_generation.py
├── test_observability.py
└── test_evaluation.py
```

---

# ⚙️ Tech Stack

| Category | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Groq |
| Parsing | LlamaParse |
| Vector Database | Qdrant |
| Dense Embeddings | BAAI/bge-small-en-v1.5 |
| Sparse Retrieval | BM25 |
| Reranker | Cross-Encoder |
| Observability | Langfuse + OpenTelemetry |
| Evaluation | LLM-as-a-Judge |

---

# 🛠️ Local Setup

## 1. Clone Repository

```bash
git clone <your-repository-url>

cd multimodal-rag-system
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file inside the project root.

```env
##############################
# Groq
##############################

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx

##############################
# LlamaParse
##############################

LLAMA_CLOUD_API_KEY=llx_xxxxxxxxxxxxxxxxx

##############################
# Langfuse
##############################

LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx

LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx

LANGFUSE_HOST=https://cloud.langfuse.com

LANGFUSE_DEBUG=False

##############################
# Qdrant
##############################

# Local
QDRANT_HOST=http://localhost:6333
QDRANT_API_KEY=

# Cloud (Optional)
# QDRANT_HOST=https://xxxxx.cloud.qdrant.io
# QDRANT_API_KEY=xxxxxxxx
```

---

# 🐳 Start Qdrant

Run a local Qdrant container.

```bash
docker run -d \
-p 6333:6333 \
-p 6334:6334 \
qdrant/qdrant
```

Windows PowerShell

```powershell
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

# 🧪 Verification Workflow

Run every module independently before starting the application.

## Retrieval Test

```bash
python test_retrieval.py
```

---

## Generation Test

```bash
python test_generation.py
```

---

## Langfuse Observability

```bash
python test_observability.py
```

---

## Evaluation Harness

```bash
python test_evaluation.py
```

---

# 🚀 Running the Application

Open **two terminals**.

---

## Terminal 1 — FastAPI

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI

```
http://localhost:8000/docs
```

---

## Terminal 2 — Streamlit

```bash
streamlit run app.py
```

Application

```
http://localhost:8501
```

---

# 🔍 Retrieval Pipeline

```text
                User Query
                     │
                     ▼
        ┌──────────────────────┐
        │ Hybrid Retrieval     │
        └──────────────────────┘
          │                │
          ▼                ▼
     Dense Search     Sparse Search
      (Qdrant)           (BM25)
          │                │
          └──────┬─────────┘
                 ▼
     Reciprocal Rank Fusion
                 │
                 ▼
      Cross Encoder Reranker
                 │
                 ▼
      Top Context Documents
                 │
                 ▼
          Groq LLM Generation
                 │
                 ▼
        Citation Verification
                 │
                 ▼
            Final Response
```

---

# 📊 Evaluation Metrics

The evaluation pipeline automatically computes:

| Metric | Purpose |
|---------|----------|
| Faithfulness | Ensures generated claims exist inside retrieved context |
| Answer Relevance | Measures how well the answer addresses the user query |
| Context Recall | Measures retrieval effectiveness |

Evaluation reports are generated inside:

```text
data/eval_report.md
```

---

# 📈 Langfuse Observability

The project automatically records:

- Root traces
- Nested spans
- Prompt templates
- Retrieval latency
- Generation latency
- Token usage
- Metadata
- User sessions

All traces are available from the Langfuse dashboard.

---

# ☁️ Deployment

## FastAPI

Deploy on **Render**

Environment Variables:

- GROQ_API_KEY
- LLAMA_CLOUD_API_KEY
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_HOST
- QDRANT_HOST
- QDRANT_API_KEY

---

## Vector Database

Deploy a free managed cluster on **Qdrant Cloud**.

Replace:

```env
QDRANT_HOST=http://localhost:6333
```

with

```env
QDRANT_HOST=https://your-cluster.cloud.qdrant.io

QDRANT_API_KEY=xxxxxxxx
```

---

## Streamlit Frontend

Deploy on **Hugging Face Spaces**.

Update the backend endpoint inside `app.py` from

```text
http://localhost:8000
```

to your deployed Render URL.

---

# 📌 Features Summary

- ✅ Layout-aware document parsing
- ✅ Dense + Sparse Hybrid Retrieval
- ✅ Reciprocal Rank Fusion
- ✅ Cross-Encoder Reranking
- ✅ Hallucination Guardrails
- ✅ Citation Verification
- ✅ Langfuse Observability
- ✅ Automated Evaluation
- ✅ FastAPI Backend
- ✅ Streamlit Frontend
- ✅ Docker-ready
- ✅ Cloud Deployment Ready

---

# 📄 License

This project is released under the MIT License.

Feel free to use, modify, and distribute.
