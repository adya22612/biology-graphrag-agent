<div align="center">

# 🧬 Biology GraphRAG Agent

**A privacy-preserving, local AI agent combining semantic search with knowledge graph traversal**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agent-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge%20Graph-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)

*Data and inference stay fully local. Designed for data sovereignty and secure deployment.*

[Architecture](#architecture) · [Design Decisions](#design-decisions) · [Stack](#stack) · [Running It](#running-it) · [Troubleshooting](#troubleshooting) · [Limitations](#limitations)

</div>

---

## What It Does

Given a complex biological query, the system executes a multi-step Retrieval-Augmented Generation pipeline:

1. **Retrieves** dense semantic context from a local vector database
2. **Traverses** explicit entity relationships (e.g., *Mitochondria → PRODUCES → ATP*) via a biological knowledge graph
3. **Evaluates** context sufficiency using the LLM as a deterministic judge to prevent ungrounded generation
4. **Synthesises** a structured, factual response combining both data sources
5. **Audits** the final output through an automated guardrail node to catch hallucinations before returning the response

Traditional RAG systems rely solely on vector databases to find conceptually similar text. **Vector search finds documents that mean something similar. Graph traversal finds entities that are related in a specific way.** This GraphRAG architecture combines both — using graph traversal for structural queries like "what are all the byproducts of the Krebs cycle?" that dense embeddings alone would conflate.

> 🔒 **Data Sovereignty:** This system operates entirely offline. Ollama runs directly on host hardware and all databases are containerised locally — no data leaves the network environment.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit UI                      │  :8501
│           User Query → Biological Answer             │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│               FastAPI Orchestrator                   │  :8000
│       LangGraph Agent · Postgres Checkpointer        │
└──────┬──────────────────────┬─────────────────┬─────┘
       │ HTTP                 │ Bolt            │ Async
┌──────▼──────┐        ┌──────▼──────┐   ┌──────▼──────┐
│  ChromaDB   │        │    Neo4j    │   │ PostgreSQL  │
│   Vectors   │        │  Knowledge  │   │   Thread    │
│  Semantics  │        │    Graph    │   │   Memory    │
└─────────────┘        └─────────────┘   └─────────────┘
       │                      │
       └──────────┬───────────┘
                  │ REST API (host.docker.internal)
     ┌────────────▼────────────────────┐
     │   Ollama (Host — GPU Access)    │  :11434
     │   └── llama3.2 (Inference)      │
     └─────────────────────────────────┘
```

State management and databases run in isolated containers. The LLM orchestrator runs natively on the host to utilise hardware acceleration without duplicating model weights inside Docker.

---

## Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph (stateful workflow API) |
| LLM orchestration | llama3.2 via Ollama |
| REST API | FastAPI + uvicorn |
| UI | Streamlit |
| Vector store | ChromaDB (dense semantic retrieval) |
| Graph store | Neo4j (entity and relationship modelling) |
| Memory state | PostgreSQL (async checkpointer) |
| Containerisation | Docker Compose |
| Configuration | Pydantic BaseSettings (environment injection) |

---

## Design Decisions

### Hybrid RAG over pure semantic search

Vector databases excel at finding conceptually similar text but fail at structural logic. A question like "what are all the byproducts of the Krebs cycle?" requires traversing a chemical pathway, not finding similar sentences. By mapping biological entities into Neo4j nodes and their interactions into edges, the agent can traverse pathways explicitly.

The LangGraph router scores query-context sufficiency and selects between ChromaDB alone, Neo4j alone, or both in combination — spending the heavier graph query only when the vector result is insufficient.

### Host-bound LLM inference

Running LLMs inside Docker requires complex NVIDIA container toolkit configuration and balloons image sizes. By binding Ollama to `0.0.0.0` on the host and pointing the containerised agent to `host.docker.internal`, the pipeline achieves containerised portability with native GPU inference speeds. The GPU sees the model; Docker sees a network endpoint.

### Stateful checkpointing via PostgreSQL

`AsyncPostgresSaver` persists thread history across container restarts. A biological query session — which may involve several follow-up questions about the same pathway — survives a container restart without losing context. If the container restarts, the agent does not lose the conversational state of the biological query.

### Self-correcting retrieval with bounded retries

If the LLM sufficiency check scores the combined context below threshold, the agent triggers iterative refinement — up to three attempts — before returning the best available answer. This prevents both hallucination (answering with insufficient context) and infinite loops (retrying indefinitely). Programmatic guardrails audit the final output before it reaches the user.

### Pydantic secrets management

Environment variables are never hardcoded. `BaseSettings` enforces strict type-checking and dependency injection at startup, and validation fails loudly at boot rather than silently at query time.

---

## Running It

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the daemon running
- Python 3.10+
- [Ollama](https://ollama.com) (host install — not containerised)

### 1. Model and host configuration

```bash
ollama pull llama3.2
```

**Critical networking step:** Ollama must listen on all interfaces so the Docker bridge can reach it. Close Ollama from the Windows system tray, open Command Prompt, and run:

```cmd
setx OLLAMA_HOST "0.0.0.0"
```

Restart Ollama from the system tray. It is now bound to all network interfaces and ready to receive traffic from Docker.

### 2. Configuration

Create a `.env` file in the project root. The backend uses Pydantic `BaseSettings` to securely inject these variables at startup:

```env
# Database
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
CHROMA_PATH=/app/chroma_db_storage

# PostgreSQL checkpointer
POSTGRES_URI=postgresql://postgres:root@postgres:5432/langgraph_state

# LLM (bridge to host Ollama)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### 3. Boot the infrastructure

Build and start all containers. Leave this terminal running to monitor logs:

```bash
docker compose up --build
```

Wait for `Application startup complete` and `200 OK` in the logs — this confirms FastAPI has connected to both PostgreSQL and Neo4j successfully.

### Data

This project ingests a biology textbook PDF. Place your source file at:
biology_textbook/textbook.pdf
Any standard biology textbook will work, NCERT class 12 biology textbook is used here in this case . 

### 4. Data ingestion

Open a **second terminal** in the project root. The ingestion scripts run on the host against the running containers:

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Populate ChromaDB (vector embeddings)
python scripts/ingest_data.py

# Build the Neo4j knowledge graph (LLM-driven entity extraction)
python scripts/build_semantic_graph.py
```

### 5. Access the services

Once ingestion completes the full pipeline is operational:

| Service | URL | Purpose |
|---|---|---|
| Streamlit UI | http://localhost:8501 | Main query interface |
| FastAPI docs | http://localhost:8000/docs | API reference and testing |
| Neo4j browser | http://localhost:7474 | Inspect the knowledge graph |

---

## Troubleshooting

**Services not starting:** Check that Docker Desktop is running and ports 7474, 7687, 8000, 8501 are not in use by another process.

**Ollama unreachable from containers:** Confirm `OLLAMA_HOST` was set before restarting Ollama. Verify with:
```cmd
netstat -ano | findstr 11434
```
You want `0.0.0.0:11434`, not `127.0.0.1:11434`.

**Neo4j authentication error:** Ensure `NEO4J_PASSWORD` in `.env` matches what you set when Neo4j first initialised. If you changed it after first boot, the stored password in the volume may differ.

**Reset the environment** — stop all services:
```bash
docker compose down
```

**Deep clean** — destroy database volumes and start fresh (re-ingestion required):
```bash
docker compose down -v
```

Use `down -v` when you need to re-ingest from scratch or resolve state corruption. It wipes PostgreSQL thread history and Neo4j graph data permanently.

---

## Repository Layout

```
.
├── docker-compose.yml
├── .env                              # Secret configurations (gitignored)
├── requirements.txt
├── scripts/
│   ├── ingest_data.py                # Vector embedding pipeline → ChromaDB
│   └── build_semantic_graph.py       # LLM-driven entity extraction → Neo4j
└── src/
    ├── api.py                        # FastAPI entry point
    ├── config.py                     # Pydantic BaseSettings validation
    ├── db/
    │   ├── chroma.py                 # Vector client initialisation
    │   └── neo4j.py                  # Graph driver initialisation
    └── graph/
        ├── builder.py                # LangGraph edge and node compilation
        ├── nodes.py                  # Async reasoning and retrieval functions
        └── state.py                  # AgentState schema
```

---

## Limitations

| Area | Detail |
|---|---|
| **Retrieval** | Dense-only within the vector path; no reranking over ChromaDB results |
| **LLM reasoning** | llama3.2 (3B) occasionally drops structural JSON formatting during graph extraction; a larger model improves extraction fidelity |
| **Data preprocessing** | Basic text chunking at ingestion; no statistical filtering of noisy biological text |
| **Tool extensibility** | Agent currently restricted to internal databases; architecture is designed to extend to external sources via MCP |
| **Scale** | Single-node Docker Compose; not tested under concurrent thread load |

---

## What a Production Version Would Need

- **Larger inference model** — llama3.2 8B or above for reliable structured JSON extraction during knowledge graph construction
- **Reranking** — cross-encoder pass over dense retrieval results before synthesis
- **MCP integration** — extend the LangGraph orchestrator to call external biological databases (UniProt, PubChem) as standardised MCP tools, consistent with the architecture used in the [Clinical Drug Recommender](../clinical-recommender)
- **Kubernetes deployment** — transition from Docker Compose to managed clusters for concurrent thread handling at scale
- **Evaluation harness** — per-node scoring of the retrieval router, sufficiency grader, and guardrail steps, not just end-to-end output quality

---

<div align="center">

</div>
