# RCPE Architecture Deep Dive

This document provides a technical overview of the Research Cross-Pollination Engine (RCPE) internal components and data flow.

## 1. Orchestration Layer

We use **LangChain** for agentic orchestration. The `Orchestrator` is responsible for:
- **Intent Analysis**: Using LLMs to parse the user's research query.
- **Plan Generation**: Decomposing the query into sub-tasks for specialized agents.
- **Context Synthesis**: Aggregating findings from multiple agents into a coherent hypothesis.

## 2. Agent Archetypes

RCPE utilizes four distinct agent types:

| Agent | Responsibility | Primary Tools |
|-------|----------------|---------------|
| **Primary Domain Agent** | Maps the current knowledge in the user's field. | Vector Search, Citation Network Explorer |
| **Cross-Domain Explorer** | Identifies analogous problems in unrelated fields. | Multi-Field Search, Analogy Finder |
| **Methodology Translator** | Adapts techniques from source fields to the target. | Method Detail Retriever, implementation_analyzer |
| **Resource Locator** | Finds datasets, code repos, and protocols. | Dataset Search, GitHub Search |

## 3. Data & Retrieval Engine

### Vector Database
We use **Chroma DB** for local development and support **Pinecone** for production scaling. 
- **Embeddings**: We use `sentence-transformers/allenai-specter` for its superior performance on scientific document abstracts.
- **Retrieval Strategy**: Multi-Query Retrieval + Re-ranking (Cross-Encoder) for maximum precision.

### Ingestion Pipeline
- **arXiv**: Fetched via the `arxiv` Python package.
- **PubMed/PMC**: Fetched via BioPython (Entrez API).
- **Semantic Scholar**: Primary source for citation graph analysis.
- **OpenAlex**: Used for broad coverage across all academic fields.

## 4. Frontend-Backend Interaction

The system exposes a **FastAPI** backend that provides:
- `/generate`: Main endpoint for hypothesis creation.
- `/sources`: Endpoint to retrieve full metadata for cited papers.
- `/stats`: Provides real-time stats on document coverage and ingestion health.

The frontend is built with **Next.js**, emphasizing a clean, "scientific" aesthetic with support for Mermaid diagram rendering and LaTeX math notation.
