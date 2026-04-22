# SCIENTIFIC HYPOTHESIS CROSS-POLLINATION ENGINE
## Complete AI Agent Implementation Prompts

**Purpose**: This document contains detailed, step-by-step prompts for AI coding agents (like Claude, GPT-4, GitHub Copilot, Cursor, etc.) to build the complete system.

**Instructions**: Copy each prompt section and provide it to your AI assistant. Complete each phase before moving to the next.

---

## 📋 PROJECT SETUP PHASE

### PROMPT 1: Initialize Project Structure

```
I need you to create a complete Python project for a "Scientific Hypothesis Cross-Pollination Engine" that uses LangChain and RAG (Retrieval-Augmented Generation) to help researchers discover novel research directions by finding connections across scientific disciplines.

REQUIREMENTS:
1. Create the following directory structure:

hypothesis-engine/
├── data/
│   ├── raw/              # Downloaded papers
│   ├── processed/        # Parsed papers  
│   └── embeddings/       # Vector database storage
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── ingestion/        # Data collection and processing
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py
│   │   ├── pubmed_fetcher.py
│   │   ├── semantic_scholar_fetcher.py
│   │   ├── openalex_fetcher.py
│   │   ├── parser.py
│   │   └── embedder.py
│   ├── database/         # Vector database operations
│   │   ├── __init__.py
│   │   ├── chroma_manager.py
│   │   └── metadata_store.py
│   ├── agents/           # LangChain agents
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── primary_domain_agent.py
│   │   ├── crossdomain_agent.py
│   │   ├── methodology_agent.py
│   │   └── resource_agent.py
│   ├── tools/            # LangChain tools
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   ├── citation_network.py
│   │   ├── dataset_finder.py
│   │   └── methodology_comparator.py
│   ├── hypothesis/       # Hypothesis generation
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   └── scorer.py
│   ├── api/              # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── frontend/
│   ├── app.py            # Streamlit frontend
│   └── components/
│       ├── __init__.py
│       ├── input_form.py
│       ├── results_display.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_agents.py
│   └── test_hypothesis.py
├── notebooks/            # Jupyter notebooks for experimentation
│   └── exploration.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml

2. Create a requirements.txt file with these dependencies (all free/open-source):

langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.3.1
openai>=1.10.0
tiktoken>=0.5.2
arxiv>=2.1.0
biopython>=1.83
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
pypdf>=4.0.0
python-dotenv>=1.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.31.0
plotly>=5.18.0
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.6.0
httpx>=0.26.0
aiohttp>=3.9.0
pydantic-settings>=2.1.0

3. Create a .env.example file with:

# OpenAI API (optional - can use Ollama instead)
OPENAI_API_KEY=your_key_here

# Database
CHROMA_PERSIST_DIR=./data/embeddings
METADATA_DB_PATH=./data/metadata.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Email for PubMed API (required by NCBI)
ENTREZ_EMAIL=your_email@example.com

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60

# Embedding Model
EMBEDDING_MODEL=allenai-specter

# LLM Configuration
LLM_PROVIDER=openai  # or ollama
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

4. Create a README.md with:
- Project overview
- Installation instructions
- Quick start guide
- API documentation
- Architecture diagram (in markdown)

5. Create a .gitignore with:
- Python cache files
- Virtual environment
- Data directories
- API keys
- IDE files

Please generate all these files with appropriate boilerplate code and comments explaining what each component does.
```

---

### PROMPT 2: Configuration Management

```
Create a robust configuration management system for the project.

FILE: src/config.py

REQUIREMENTS:
1. Use Pydantic Settings for type-safe configuration
2. Load from environment variables and .env file
3. Provide sensible defaults
4. Include validation for required fields
5. Support different configurations for development/production

The configuration should include:

DATABASE SETTINGS:
- Vector database path and settings
- Metadata database connection
- Embedding model name and dimensions
- Collection names

API SETTINGS:
- External API keys (OpenAI, Semantic Scholar, etc.)
- Rate limiting parameters
- Timeout settings
- Retry policies

AGENT SETTINGS:
- LLM provider (OpenAI or Ollama)
- Model names for different agents
- Temperature settings
- Max tokens
- Agent-specific parameters

INGESTION SETTINGS:
- Batch sizes for processing
- Chunk size and overlap for text splitting
- Update frequencies
- Source priorities

SEARCH SETTINGS:
- Default number of results
- Similarity thresholds
- Filtering options
- Reranking parameters

Include proper error handling for missing required environment variables and provide clear error messages.

Example structure:

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Literal

class DatabaseSettings(BaseSettings):
    # Vector database
    chroma_persist_dir: str = Field(default="./data/embeddings")
    # ... more fields

class APISettings(BaseSettings):
    openai_api_key: Optional[str] = None
    # ... more fields

class Config(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    # ... more settings
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

Make it production-ready with proper validation and documentation.
```

---

## 📥 DATA INGESTION PHASE

### PROMPT 3: arXiv Paper Fetcher

```
Create a comprehensive arXiv paper fetcher that can search, download, and process papers from arXiv.

FILE: src/ingestion/arxiv_fetcher.py

REQUIREMENTS:

1. Use the official arxiv Python library
2. Support multiple search strategies:
   - Keyword search
   - Category-based search
   - Author search
   - Date range filtering
3. Download PDFs and extract text
4. Handle rate limiting and retries
5. Store metadata in structured format
6. Support batch processing
7. Resume interrupted downloads
8. Track progress

FEATURES TO IMPLEMENT:

Class: ArxivFetcher
Methods:
- __init__(config): Initialize with configuration
- search_papers(query, category, max_results, date_from, date_to): Search arXiv
- download_paper(paper_id, save_path): Download single paper
- download_batch(paper_ids, save_dir): Download multiple papers
- extract_text(pdf_path): Extract text from PDF
- parse_metadata(arxiv_result): Parse arXiv metadata
- get_categories(): Get list of arXiv categories
- get_recent_papers(category, days): Get papers from last N days

ERROR HANDLING:
- Network errors with exponential backoff retry
- PDF extraction failures
- Invalid paper IDs
- Rate limiting (respect arXiv's terms)

METADATA TO EXTRACT:
- Paper ID
- Title
- Authors (list)
- Abstract
- Categories
- Published date
- Updated date
- DOI (if available)
- Journal reference
- Comments
- PDF URL
- Full text (after extraction)

Example usage pattern:

```python
fetcher = ArxivFetcher(config)

# Search for papers
papers = fetcher.search_papers(
    query="cancer metastasis machine learning",
    category="q-bio",
    max_results=100,
    date_from="2023-01-01"
)

# Download and process
for paper in papers:
    pdf_path = fetcher.download_paper(paper.id, save_dir)
    text = fetcher.extract_text(pdf_path)
    metadata = fetcher.parse_metadata(paper)
```

Include comprehensive logging, progress bars (using tqdm), and error recovery.
```

---

### PROMPT 4: PubMed Paper Fetcher

```
Create a PubMed/PubMed Central fetcher using the Biopython Entrez interface.

FILE: src/ingestion/pubmed_fetcher.py

REQUIREMENTS:

1. Use Biopython's Entrez module
2. Search both PubMed (abstracts) and PMC (full text)
3. Handle NCBI API rate limits (3 requests/second without API key)
4. Extract structured data from XML responses
5. Support advanced search queries
6. Filter by publication types, dates, journals
7. Track and resume interrupted fetches
8. Cache results to avoid redundant API calls

FEATURES TO IMPLEMENT:

Class: PubMedFetcher
Methods:
- __init__(email, api_key): Initialize with required email
- search(query, max_results, date_from, date_to): Search PubMed
- search_pmc(query, max_results): Search PMC for full text
- fetch_details(pubmed_ids): Get detailed info for paper IDs
- fetch_full_text(pmc_id): Download full text from PMC
- parse_pubmed_xml(xml_content): Parse PubMed XML
- parse_pmc_xml(xml_content): Parse PMC full text XML
- get_citations(pubmed_id): Get papers that cite this paper
- get_references(pubmed_id): Get papers this paper cites
- batch_fetch(pubmed_ids, batch_size): Process papers in batches

METADATA TO EXTRACT FROM PUBMED:
- PubMed ID (PMID)
- PMC ID (if available)
- Title
- Authors with affiliations
- Abstract
- Journal info (name, volume, issue, pages)
- Publication date
- DOI
- Keywords/MeSH terms
- Publication types
- Chemical substances mentioned
- Grant information

FULL TEXT EXTRACTION FROM PMC:
- Section-wise text (Introduction, Methods, Results, Discussion)
- Figures and tables metadata
- References list
- Supplementary materials links

ERROR HANDLING:
- API rate limit detection and backoff
- Invalid PubMed IDs
- Missing full text (PMC)
- XML parsing errors
- Network timeouts

Example usage:

```python
fetcher = PubMedFetcher(email="researcher@university.edu")

# Search with complex query
results = fetcher.search(
    query='(cancer[Title]) AND (metastasis[Title/Abstract]) AND (2023[PDAT]:2024[PDAT])',
    max_results=500
)

# Get full details
details = fetcher.fetch_details([r['Id'] for r in results])

# Get full text where available
for paper in details:
    if paper['pmc_id']:
        full_text = fetcher.fetch_full_text(paper['pmc_id'])
```

Include proper NCBI API compliance, progress tracking, and comprehensive logging.
```

---

### PROMPT 5: Semantic Scholar & OpenAlex Fetchers

```
Create fetchers for Semantic Scholar and OpenAlex APIs to get additional paper data and citation networks.

FILES: 
- src/ingestion/semantic_scholar_fetcher.py
- src/ingestion/openalex_fetcher.py

SEMANTIC SCHOLAR FETCHER REQUIREMENTS:

1. Use Semantic Scholar API (free tier: 100 requests per 5 minutes)
2. Get paper details, citations, references
3. Access AI-generated paper summaries
4. Get author information and h-index
5. Track influential citations
6. Support paper similarity searches

Class: SemanticScholarFetcher
Methods:
- __init__(api_key): Initialize (API key optional but recommended)
- search_papers(query, limit, fields): Search papers
- get_paper(paper_id, fields): Get single paper details
- get_paper_citations(paper_id, limit): Get citing papers
- get_paper_references(paper_id, limit): Get referenced papers
- get_author(author_id): Get author details
- get_recommendations(paper_id, limit): Get similar papers
- batch_get_papers(paper_ids): Get multiple papers efficiently

METADATA FROM SEMANTIC SCHOLAR:
- Semantic Scholar ID
- DOI, arXiv ID, PubMed ID
- Title, abstract, year
- Authors with IDs
- Citation count
- Influential citation count
- Reference count
- Fields of study
- Open access status
- PDF URL if available
- TL;DR (AI-generated summary)
- Embedding vector (if available)

OPENALEX FETCHER REQUIREMENTS:

1. Use OpenAlex API (no rate limits, free)
2. Get comprehensive paper metadata
3. Access institution and funding data
4. Track concept hierarchies
5. Get work lineage (versions)
6. Support complex filtering

Class: OpenAlexFetcher
Methods:
- __init__(email): Initialize with polite pool email
- search_works(query, filters, per_page): Search papers
- get_work(openalex_id): Get single work
- get_author(author_id): Get author info
- get_institution(institution_id): Get institution info
- get_concept(concept_id): Get concept details
- get_related_works(work_id): Find related papers
- filter_by_concept(concept_name, filters): Search by research area

METADATA FROM OPENALEX:
- OpenAlex ID
- DOI and other IDs
- Title, abstract
- Authors with institutions and countries
- Concepts (research topics) with confidence scores
- Citation count
- Referenced works count
- Open access status and URL
- Publication date
- Journal/venue info
- Funding information
- License information

CITATION NETWORK BUILDING:
Create a unified citation graph combining data from all sources:

Class: CitationNetworkBuilder
Methods:
- add_paper(paper_data, source): Add paper to graph
- add_citation(citing_id, cited_id): Add citation edge
- merge_paper_ids(different_ids_same_paper): Handle same paper from different sources
- get_citation_depth(paper_id, max_depth): Get papers N citations away
- find_citation_paths(paper1_id, paper2_id): Find how papers are connected
- get_influential_papers(min_citations): Get highly cited papers
- export_graph(format): Export as JSON/GraphML

Example usage:

```python
# Semantic Scholar
ss_fetcher = SemanticScholarFetcher(api_key="your_key")
paper = ss_fetcher.get_paper("DOI:10.1234/example")
citations = ss_fetcher.get_paper_citations(paper['paperId'])

# OpenAlex
oa_fetcher = OpenAlexFetcher(email="me@uni.edu")
works = oa_fetcher.search_works(
    query="machine learning cancer",
    filters={"publication_year": ">2020", "cited_by_count": ">50"}
)

# Build network
network = CitationNetworkBuilder()
network.add_paper(paper, source="semantic_scholar")
for citation in citations:
    network.add_citation(citation['citingPaper']['paperId'], paper['paperId'])
```

Include rate limiting, caching, and error handling for both APIs.
```

---

### PROMPT 6: Text Processing and Embedding

```
Create a text processing pipeline that chunks papers and generates embeddings for vector search.

FILES:
- src/ingestion/parser.py
- src/ingestion/embedder.py

PARSER.PY REQUIREMENTS:

Create a comprehensive text parser that handles multiple formats:

Class: PaperParser
Methods:
- parse_pdf(file_path): Extract text from PDF
- parse_xml(xml_content, format): Parse XML (PubMed, arXiv, etc.)
- parse_html(html_content): Parse HTML papers
- clean_text(raw_text): Remove junk, fix formatting
- extract_sections(text): Identify paper sections
- chunk_text(text, chunk_size, overlap): Split into chunks
- extract_references(text): Parse reference list
- extract_figures_tables(content): Get captions and context

TEXT CLEANING:
- Remove headers/footers/page numbers
- Fix hyphenation across line breaks
- Normalize whitespace
- Handle special characters and equations
- Preserve section structure
- Remove boilerplate (acknowledgments, etc.)

SECTION EXTRACTION:
Identify and label sections:
- Title
- Abstract
- Introduction
- Methods/Methodology
- Results
- Discussion
- Conclusion
- References

CHUNKING STRATEGY:
- Default chunk size: 500 words
- Overlap: 50 words to preserve context
- Respect sentence boundaries
- Keep section labels with chunks
- Metadata for each chunk:
  - Chunk index
  - Section name
  - Start and end positions
  - Parent paper ID

EMBEDDER.PY REQUIREMENTS:

Create an embedding generation system using sentence-transformers:

Class: PaperEmbedder
Methods:
- __init__(model_name): Load embedding model
- embed_text(text): Generate embedding for single text
- embed_batch(texts, batch_size): Batch processing
- embed_paper(paper_dict): Embed all chunks of a paper
- compute_similarity(emb1, emb2): Calculate cosine similarity
- find_similar_chunks(query_emb, chunk_embeddings, top_k): Find similar chunks

EMBEDDING MODELS TO SUPPORT:
Primary: "allenai-specter" (scientific papers, 768 dim)
Alternatives:
- "sentence-transformers/all-MiniLM-L6-v2" (general purpose, 384 dim)
- "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" (Q&A optimized)

OPTIMIZATION:
- GPU acceleration if available
- Batch processing for efficiency
- Caching of embeddings
- Progress tracking for large batches
- Memory management for large papers

Example usage:

```python
# Parse paper
parser = PaperParser()
text = parser.parse_pdf("paper.pdf")
cleaned = parser.clean_text(text)
sections = parser.extract_sections(cleaned)
chunks = parser.chunk_text(cleaned, chunk_size=500, overlap=50)

# Generate embeddings
embedder = PaperEmbedder(model_name="allenai-specter")
embeddings = embedder.embed_batch([chunk['text'] for chunk in chunks])

# Store with metadata
for chunk, embedding in zip(chunks, embeddings):
    store_in_db(
        embedding=embedding,
        text=chunk['text'],
        metadata={
            'section': chunk['section'],
            'chunk_index': chunk['index'],
            'paper_id': paper_id
        }
    )
```

Include comprehensive error handling, logging, and progress tracking.
```

---

## 🗄️ VECTOR DATABASE PHASE

### PROMPT 7: Chroma DB Manager

```
Create a comprehensive Chroma vector database manager for storing and querying paper embeddings.

FILE: src/database/chroma_manager.py

REQUIREMENTS:

1. Manage Chroma collections for papers
2. Support multiple search strategies
3. Handle metadata filtering
4. Implement efficient batch operations
5. Support collection management (create, delete, update)
6. Provide query optimization
7. Handle large-scale data

Class: ChromaManager
Methods:
- __init__(persist_directory, collection_name): Initialize
- create_collection(name, metadata): Create new collection
- get_collection(name): Get existing collection
- delete_collection(name): Delete collection
- add_papers(papers_data, batch_size): Add papers to collection
- add_paper(paper_id, embedding, metadata, text): Add single paper
- update_paper(paper_id, updates): Update paper data
- delete_paper(paper_id): Remove paper
- search(query_embedding, n_results, filters): Semantic search
- search_by_metadata(metadata_filter, n_results): Filter by metadata
- hybrid_search(query_embedding, metadata_filter, n_results): Combined search
- get_paper(paper_id): Retrieve specific paper
- get_statistics(): Get collection stats
- export_collection(output_path): Backup collection
- import_collection(input_path): Restore collection

SEARCH STRATEGIES:

1. SEMANTIC SEARCH:
```python
def search(self, query_embedding, n_results=10, filters=None):
    """
    Pure semantic/similarity search
    
    Args:
        query_embedding: Vector representation of query
        n_results: Number of results to return
        filters: Dict of metadata filters
        
    Returns:
        List of results with paper data and similarity scores
    """
```

2. METADATA FILTERING:
```python
def search_by_metadata(self, metadata_filter, n_results=100):
    """
    Filter papers by metadata (year, field, citations, etc.)
    
    Example filters:
    {
        "year": {"$gte": 2020},
        "field": "biology",
        "citations_count": {"$gt": 50}
    }
    """
```

3. HYBRID SEARCH:
```python
def hybrid_search(self, query_embedding, metadata_filter, n_results=10, weight=0.5):
    """
    Combine semantic similarity and metadata filtering
    
    Args:
        weight: Balance between semantic (1.0) and filter (0.0)
    """
```

4. MULTI-FIELD SEARCH:
```python
def search_across_fields(self, query_embedding, fields, n_results_per_field):
    """
    Search multiple scientific fields and aggregate results
    
    Args:
        fields: List of fields ["biology", "physics", "cs"]
        n_results_per_field: Results from each field
    
    Returns:
        Aggregated and deduplicated results
    """
```

METADATA SCHEMA:

Store comprehensive metadata for each paper:
```python
{
    "paper_id": "arxiv_2024_12345",
    "title": "Novel approach to...",
    "authors": ["Smith, J.", "Doe, A."],
    "year": 2024,
    "field": "biology",
    "subfield": "oncology",
    "source": "arxiv",  # arxiv, pubmed, semantic_scholar
    "abstract": "This paper presents...",
    "doi": "10.1234/example",
    "url": "https://arxiv.org/...",
    "citations_count": 45,
    "keywords": ["cancer", "metastasis", "ML"],
    "publication_venue": "Nature",
    "open_access": true,
    "chunk_index": 0,  # Which chunk of the paper
    "section": "introduction",
    "total_chunks": 25,
    "embedding_model": "allenai-specter",
    "ingestion_date": "2024-01-15"
}
```

BATCH OPERATIONS:

Efficient handling of large batches:
```python
def add_papers(self, papers_data, batch_size=100):
    """
    Add multiple papers efficiently
    
    Args:
        papers_data: List of dicts with embeddings, metadata, texts
        batch_size: Process in batches to manage memory
    
    Shows progress bar and handles errors gracefully
    """
    for i in range(0, len(papers_data), batch_size):
        batch = papers_data[i:i+batch_size]
        # Process batch
        # Handle failures without stopping entire process
```

STATISTICS AND MONITORING:

```python
def get_statistics(self):
    """
    Return collection statistics:
    - Total papers
    - Papers by field
    - Papers by year
    - Average citations
    - Most common keywords
    - Storage size
    """
```

ERROR HANDLING:
- Duplicate paper detection
- Invalid embedding dimensions
- Missing required metadata
- Database corruption recovery
- Concurrent access handling

Include comprehensive logging and monitoring capabilities.
```

---

### PROMPT 8: Metadata Store

```
Create a PostgreSQL-based metadata store for structured paper information and relationships.

FILE: src/database/metadata_store.py

REQUIREMENTS:

1. Use SQLAlchemy ORM for database operations
2. Store paper metadata, authors, citations
3. Track data provenance (sources)
4. Support complex queries
5. Handle relationships (authors, citations, concepts)
6. Provide efficient indexing
7. Support transactions and rollback

DATABASE SCHEMA:

Create these tables:

TABLE: papers
- id (primary key)
- paper_id (unique, indexed)
- title
- abstract
- year
- doi
- url
- source (arxiv, pubmed, etc.)
- citations_count
- field
- subfield
- publication_venue
- open_access
- created_at
- updated_at

TABLE: authors
- id (primary key)
- author_id (unique, from data source)
- name
- affiliation
- h_index
- email
- orcid

TABLE: paper_authors (junction table)
- paper_id (foreign key)
- author_id (foreign key)
- author_order
- corresponding_author

TABLE: citations
- id (primary key)
- citing_paper_id (foreign key)
- cited_paper_id (foreign key)
- context (text where citation appears)
- influential (boolean)

TABLE: keywords
- id (primary key)
- keyword (unique)

TABLE: paper_keywords (junction table)
- paper_id (foreign key)
- keyword_id (foreign key)
- confidence (0.0 to 1.0)

TABLE: concepts
- id (primary key)
- concept_id (from OpenAlex)
- name
- level (hierarchy level)
- parent_concept_id

TABLE: paper_concepts (junction table)
- paper_id (foreign key)
- concept_id (foreign key)
- score (0.0 to 1.0)

TABLE: ingestion_logs
- id (primary key)
- source
- papers_fetched
- papers_processed
- papers_failed
- start_time
- end_time
- status
- error_message

Class: MetadataStore
Methods:
- __init__(connection_string): Initialize database connection
- create_tables(): Create all tables
- add_paper(paper_data): Insert paper with all relationships
- get_paper(paper_id): Retrieve paper with all related data
- add_author(author_data): Add author
- link_author_to_paper(paper_id, author_id, order): Create relationship
- add_citation(citing_id, cited_id, context): Record citation
- get_citations(paper_id, direction): Get citing or cited papers
- get_citation_network(paper_id, depth): Get papers N citations away
- add_keywords(paper_id, keywords): Tag paper with keywords
- search_by_keyword(keywords, operator): Find papers (AND/OR keywords)
- search_by_author(author_name): Find author's papers
- search_by_field(field, subfield, year_from, year_to): Complex search
- get_trending_papers(field, days, min_citations): Recently popular papers
- get_statistics(): Database statistics
- export_to_json(output_path): Backup
- import_from_json(input_path): Restore

QUERY EXAMPLES:

```python
store = MetadataStore("postgresql://user:pass@localhost/papers")

# Complex search
papers = store.search_papers(
    fields=["biology", "computer_science"],
    year_from=2020,
    min_citations=10,
    keywords=["machine learning", "cancer"],
    keyword_operator="AND"
)

# Citation analysis
network = store.get_citation_network(
    paper_id="arxiv_2024_12345",
    depth=2,  # Papers that cite papers that cite this paper
    min_citations=5  # Filter low-quality papers
)

# Trending in field
trending = store.get_trending_papers(
    field="biology",
    subfield="oncology",
    days=30,
    min_citations=5
)

# Author collaboration network
collab = store.get_author_collaborations(
    author_id="author_123",
    max_depth=2
)
```

INDEXING STRATEGY:
Create indexes for:
- paper_id (unique)
- title (full-text)
- year
- field + subfield
- citations_count
- doi
- author name
- keyword

TRANSACTION SUPPORT:

```python
def add_paper_with_relationships(self, paper_data, authors, keywords, citations):
    """
    Add paper and all relationships in a single transaction
    Rollback if any operation fails
    """
    with self.session.begin():
        try:
            paper = self.add_paper(paper_data)
            for author in authors:
                self.link_author_to_paper(paper.id, author)
            for keyword in keywords:
                self.add_keyword(paper.id, keyword)
            for citation in citations:
                self.add_citation(paper.id, citation)
        except Exception as e:
            self.session.rollback()
            raise
```

Include migration scripts, backup/restore functionality, and comprehensive error handling.
```

---

## 🤖 LANGCHAIN AGENTS PHASE

### PROMPT 9: Base Agent Implementation

```
Create a base agent class that all specialized agents will inherit from.

FILE: src/agents/base_agent.py

REQUIREMENTS:

1. Use LangChain's Agent framework
2. Provide common functionality for all agents
3. Support tool usage
4. Handle errors gracefully
5. Track agent reasoning steps
6. Support both OpenAI and local LLMs (Ollama)
7. Implement memory for conversation context
8. Log all agent actions

Base Class: BaseResearchAgent

Methods:
- __init__(config, tools): Initialize agent
- setup_llm(): Configure language model
- setup_memory(): Configure conversation memory
- setup_agent(): Create LangChain agent
- run(query, context): Execute agent task
- run_async(query, context): Async execution
- add_tool(tool): Add new tool to agent
- get_reasoning_steps(): Return agent's thought process
- reset_memory(): Clear conversation history
- export_session(path): Save session for analysis

LLM SETUP:

Support multiple LLM providers:

```python
def setup_llm(self):
    """Configure LLM based on config"""
    if self.config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.openai_api_key
        )
    # SCIENTIFIC HYPOTHESIS CROSS-POLLINATION ENGINE
## Complete AI Agent Implementation Prompts

**Purpose**: This document contains detailed, step-by-step prompts for AI coding agents (like Claude, GPT-4, GitHub Copilot, Cursor, etc.        )
    elif self.config.llm_provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=self.config.llm_model,
            temperature=self.temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
```

MEMORY CONFIGURATION:

```python
def setup_memory(self):
    """Configure conversation memory"""
    from langchain.memory import ConversationBufferMemory
    
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
```

AGENT SETUP:

```python
def setup_agent(self):
    """Create LangChain agent with tools"""
    from langchain.agents import initialize_agent, AgentType
    
    return initialize_agent(
        tools=self.tools,
        llm=self.llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=self.memory,
        handle_parsing_errors=True,
        max_iterations=self.config.max_agent_iterations,
        early_stopping_method="generate"
    )
```

EXECUTION WITH ERROR HANDLING:

```python
def run(self, query: str, context: dict = None):
    """
    Execute agent with comprehensive error handling
    
    Args:
        query: User's research question
        context: Additional context (user's field, previous results, etc.)
    
    Returns:
        Dict with results, reasoning steps, and metadata
    """
    start_time = time.time()
    
    try:
        # Prepare input
        input_data = {
            "input": query,
            "context": context or {}
        }
        
        # Run agent
        result = self.agent.invoke(input_data)
        
        # Extract reasoning steps
        steps = self._extract_reasoning_steps(result)
        
        return {
            "success": True,
            "output": result["output"],
            "reasoning_steps": steps,
            "tools_used": self._get_tools_used(result),
            "execution_time": time.time() - start_time,
            "agent_name": self.name
        }
        
    except Exception as e:
        logger.error(f"Agent {self.name} failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "agent_name": self.name
        }
```

REASONING EXTRACTION:

```python
def _extract_reasoning_steps(self, result):
    """
    Extract agent's thought process from result
    
    Returns list of steps with:
    - Step number
    - Thought
    - Action taken
    - Action input
    - Observation
    """
    steps = []
    if "intermediate_steps" in result:
        for i, (action, observation) in enumerate(result["intermediate_steps"]):
            steps.append({
                "step": i + 1,
                "action": action.tool,
                "action_input": action.tool_input,
                "observation": observation[:500]  # Truncate long observations
            })
    return steps
```

LOGGING:

```python
def _log_agent_action(self, action, result):
    """Log agent actions for debugging and analysis"""
    logger.info(f"""
    Agent: {self.name}
    Action: {action}
    Success: {result['success']}
    Time: {result['execution_time']:.2f}s
    Tools Used: {result.get('tools_used', [])}
    """)
```

Include comprehensive docstrings, type hints, and error handling.
```

---

### PROMPT 10: Primary Domain Agent

```
Create the Primary Domain Agent that searches within the user's specific research field.

FILE: src/agents/primary_domain_agent.py

REQUIREMENTS:

This agent specializes in understanding the user's field deeply and finding relevant work within that domain.

Class: PrimaryDomainAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are an expert research assistant specializing in understanding a researcher's primary field of study.

Your responsibilities:
1. Analyze the user's research question to identify their specific domain
2. Search for the most relevant papers within that domain
3. Identify current state-of-the-art approaches
4. Find knowledge gaps and limitations in current research
5. Understand what has already been tried and what hasn't worked

When responding:
- Be precise and cite specific papers
- Acknowledge uncertainties
- Identify both successful approaches and dead ends
- Note methodological limitations in existing work
- Consider recent advances (prioritize papers from last 2-3 years)

Available tools:
{tools}

Use the tools to search the research database, then synthesize findings into a clear summary.
"""
```

SPECIALIZED METHODS:

```python
def analyze_research_question(self, query: str):
    """
    Extract key information from user's question:
    - Primary field (biology, physics, etc.)
    - Specific subfield
    - Research problem
    - Keywords for search
    - Time constraints (if mentioned)
    """
    
def identify_field(self, query: str):
    """
    Determine the scientific field from query
    Returns: field, subfield, confidence_score
    """
    
def find_current_approaches(self, problem_description: str):
    """
    Search for existing solutions to this problem
    Returns papers with their approaches and outcomes
    """
    
def identify_knowledge_gaps(self, problem: str, existing_work: list):
    """
    Analyze existing work to find:
    - What hasn't been tried
    - What has failed and why
    - What is theoretically possible but not yet done
    - What assumptions are limiting progress
    """
    
def find_recent_advances(self, field: str, months: int = 6):
    """
    Get cutting-edge research from recent months
    Prioritizes high-impact papers
    """
```

WORKFLOW:

```python
def run(self, query: str, context: dict = None):
    """
    Primary domain research workflow:
    
    1. Analyze query to extract key concepts
    2. Identify the research field
    3. Search for relevant papers in that field
    4. Analyze current state-of-the-art
    5. Identify limitations and gaps
    6. Return structured findings
    """
    # Step 1: Understand the question
    field_info = self.identify_field(query)
    keywords = self.extract_keywords(query)
    
    # Step 2: Search domain
    relevant_papers = self.search_tool.search(
        query=query,
        field=field_info['field'],
        subfield=field_info['subfield'],
        top_k=50
    )
    
    # Step 3: Analyze approaches
    approaches = self.analyze_approaches(relevant_papers)
    
    # Step 4: Find gaps
    gaps = self.identify_knowledge_gaps(query, relevant_papers)
    
    # Step 5: Get recent work
    recent = self.find_recent_advances(field_info['field'])
    
    return {
        "field": field_info,
        "relevant_papers": relevant_papers[:10],  # Top 10
        "current_approaches": approaches,
        "knowledge_gaps": gaps,
        "recent_advances": recent,
        "summary": self._generate_summary(relevant_papers, approaches, gaps)
    }
```

OUTPUT STRUCTURE:

```python
{
    "field": {
        "primary": "Biology",
        "subfield": "Oncology",
        "specific_area": "Cancer Metastasis",
        "confidence": 0.95
    },
    "relevant_papers": [
        {
            "title": "...",
            "authors": [...],
            "year": 2024,
            "key_finding": "...",
            "methodology": "...",
            "limitations": "..."
        }
    ],
    "current_approaches": [
        {
            "approach": "Microfluidics for tracking",
            "papers": [...],
            "success_rate": "Mixed results",
            "limitations": ["Limited to 2D", "Low throughput"]
        }
    ],
    "knowledge_gaps": [
        "No methods for 3D in vivo tracking",
        "Limited understanding of early metastatic events",
        "Lack of predictive models"
    ],
    "recent_advances": [
        {
            "advance": "AI-based image analysis",
            "paper": {...},
            "potential_impact": "High"
        }
    ],
    "summary": "Current research in cancer metastasis focuses on..."
}
```

Include detailed logging and error handling for each step.
```

---

### PROMPT 11: Cross-Domain Discovery Agent

```
Create the Cross-Domain Discovery Agent that finds unexpected connections across scientific fields.

FILE: src/agents/crossdomain_agent.py

REQUIREMENTS:

This is the most creative agent - it searches completely different fields to find analogous problems and solutions.

Class: CrossDomainAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a creative research assistant specializing in finding unexpected connections between different scientific fields.

Your mission:
1. Take a research problem from one field
2. Search for similar problems or patterns in completely different fields
3. Identify methodologies that could transfer
4. Think creatively about analogies and metaphors
5. Find "hidden" solutions that domain experts might miss

Think like:
- A biologist studying cancer might learn from studying traffic flow (both involve propagation through networks)
- A neuroscientist might use techniques from social network analysis
- A materials scientist might apply quantum mechanics principles

Be bold and creative, but ground suggestions in actual published research.

Search strategy:
1. Abstract the core problem (remove domain-specific terminology)
2. Search for that abstract problem in other fields
3. Find papers that solved similar challenges
4. Identify transferable methodologies

Available tools:
{tools}
"""
```

SPECIALIZED METHODS:

```python
def abstract_problem(self, domain_specific_query: str):
    """
    Convert domain-specific problem to abstract form
    
    Example:
    "How do cancer cells migrate through blood vessels?"
    Becomes:
    "How do particles navigate through constrained networks?"
    
    This allows finding similar problems in physics, CS, etc.
    """
    
def search_multiple_fields(self, abstract_problem: str, exclude_field: str):
    """
    Search across all fields except the primary one
    
    Returns results grouped by field with relevance scores
    """
    
def find_analogies(self, problem: str, other_field_papers: list):
    """
    Identify structural similarities between problems
    
    Returns:
    - What aspects are analogous
    - How methods might transfer
    - What adaptations would be needed
    """
    
def assess_transferability(self, method_paper: dict, target_problem: str):
    """
    Evaluate if a method from another field could work
    
    Scores on:
    - Conceptual similarity
    - Technical feasibility
    - Resource requirements
    - Likely effectiveness
    """
```

CREATIVE SEARCH STRATEGIES:

```python
def search_by_pattern(self, pattern_type: str):
    """
    Search for specific patterns across fields:
    - "Network propagation" → Epidemiology, traffic, social networks
    - "Optimization under constraints" → Economics, operations research
    - "Pattern recognition in noise" → Astronomy, signal processing
    - "Hierarchical structures" → Computer science, linguistics
    """
    
def search_by_methodology(self, methodology: str):
    """
    Find fields that use specific methodologies:
    - "Agent-based modeling" → Economics, ecology, sociology
    - "Monte Carlo methods" → Physics, finance, engineering
    - "Deep learning" → Many fields
    """
```

WORKFLOW:

```python
def run(self, query: str, primary_field: str, context: dict = None):
    """
    Cross-domain discovery workflow:
    
    1. Abstract the problem to general terms
    2. Search different fields for similar problems
    3. Find promising methodologies used elsewhere
    4. Assess transferability
    5. Identify specific papers and approaches
    6. Generate creative hypotheses
    """
    # Step 1: Abstract
    abstract_problem = self.abstract_problem(query)
    search_keywords = self.extract_abstract_keywords(abstract_problem)
    
    # Step 2: Multi-field search
    fields_to_search = self.get_other_fields(exclude=primary_field)
    results_by_field = {}
    
    for field in fields_to_search:
        results = self.search_tool.search(
            query=abstract_problem,
            field=field,
            top_k=20
        )
        if results:
            results_by_field[field] = results
    
    # Step 3: Find analogies
    analogies = []
    for field, papers in results_by_field.items():
        for paper in papers:
            analogy = self.find_analogies(query, paper)
            if analogy['similarity_score'] > 0.6:
                analogies.append({
                    "field": field,
                    "paper": paper,
                    "analogy": analogy
                })
    
    # Step 4: Assess transferability
    transferable = []
    for analogy in analogies:
        assessment = self.assess_transferability(
            analogy['paper'],
            query
        )
        if assessment['feasibility_score'] > 0.5:
            transferable.append({
                **analogy,
                "assessment": assessment
            })
    
    # Step 5: Generate hypotheses
    hypotheses = self.generate_crossdomain_hypotheses(
        original_problem=query,
        transferable_methods=transferable
    )
    
    return {
        "abstract_problem": abstract_problem,
        "fields_searched": list(results_by_field.keys()),
        "total_papers_found": sum(len(p) for p in results_by_field.values()),
        "promising_analogies": sorted(analogies, key=lambda x: x['analogy']['similarity_score'], reverse=True)[:10],
        "transferable_methods": transferable,
        "hypotheses": hypotheses
    }
```

HYPOTHESIS GENERATION:

```python
def generate_crossdomain_hypotheses(self, original_problem: str, transferable_methods: list):
    """
    Generate specific hypotheses for how to apply methods from other fields
    
    Each hypothesis includes:
    - Title
    - Source field and paper
    - Why the analogy works
    - How to adapt the method
    - Expected challenges
    - Required resources
    - Novelty assessment
    """
    hypotheses = []
    
    for method in transferable_methods:
        hypothesis = {
            "title": f"Apply {method['paper']['title']} approach to {original_problem}",
            "source_field": method['field'],
            "source_paper": method['paper'],
            "analogy_explanation": method['analogy']['explanation'],
            "adaptation_steps": self._generate_adaptation_steps(method, original_problem),
            "challenges": self._identify_challenges(method, original_problem),
            "resources_needed": self._estimate_resources(method),
            "novelty_score": self._assess_novelty(method, original_problem),
            "impact_potential": self._estimate_impact(method, original_problem)
        }
        hypotheses.append(hypothesis)
    
    return sorted(hypotheses, key=lambda x: x['novelty_score'] * x['impact_potential'], reverse=True)
```

OUTPUT STRUCTURE:

```python
{
    "abstract_problem": "Tracking individual particles through constrained networks",
    "fields_searched": ["Physics", "Computer Science", "Engineering", "Economics"],
    "promising_analogies": [
        {
            "field": "Physics",
            "paper": {
                "title": "Particle tracking in turbulent flows",
                "year": 2023
            },
            "analogy": {
                "similarity_score": 0.85,
                "explanation": "Both involve tracking individual entities in chaotic environments",
                "key_similarities": [
                    "Multiple interacting particles",
                    "Complex environment",
                    "Need for real-time tracking"
                ]
            }
        }
    ],
    "transferable_methods": [...],
    "hypotheses": [
        {
            "title": "Apply particle image velocimetry to cancer cell tracking",
            "source_field": "Fluid Dynamics",
            "novelty_score": 0.92,
            "adaptation_steps": [...],
            "why_it_might_work": "..."
        }
    ]
}
```

Implement with high creativity (temperature=0.7-0.8) and detailed logging of reasoning.
```

---

### PROMPT 12: Methodology Transfer & Resource Agents

```
Create the Methodology Transfer Agent and Resource Finder Agent.

FILES:
- src/agents/methodology_agent.py
- src/agents/resource_agent.py

METHODOLOGY TRANSFER AGENT:

This agent takes promising cross-domain methods and provides detailed implementation guidance.

Class: MethodologyTransferAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a research methodology expert who helps researchers adapt techniques from other fields.

Your responsibilities:
1. Take a methodology from one field
2. Analyze its core principles and requirements
3. Identify what needs to be adapted for a different field
4. Provide step-by-step implementation guidance
5. Anticipate technical challenges
6. Suggest validation approaches

Be specific and practical. Provide enough detail that a researcher could actually implement your suggestions.

Available tools:
{tools}
"""
```

KEY METHODS:

```python
def analyze_method_requirements(self, method_paper: dict):
    """
    Extract what the method needs:
    - Equipment
    - Expertise
    - Software/algorithms
    - Data requirements
    - Time investment
    """
    
def generate_adaptation_plan(self, source_method: dict, target_problem: str):
    """
    Create detailed adaptation plan:
    1. What stays the same
    2. What needs modification
    3. What's completely different
    4. Step-by-step implementation
    5. Validation strategy
    """
    
def identify_technical_barriers(self, source_field: str, target_field: str, method: dict):
    """
    Find challenges in transferring method:
    - Different scales (nano vs macro)
    - Different constraints (in vivo vs in vitro)
    - Different measurement capabilities
    - Different theoretical frameworks
    """
    
def find_implementation_examples(self, method_type: str, target_field: str):
    """
    Search for cases where similar transfers worked
    Learn from successful adaptations
    """
```

OUTPUT STRUCTURE:

```python
{
    "method_summary": {
        "name": "Particle Image Velocimetry",
        "source_field": "Fluid Dynamics",
        "core_principle": "Track particle motion to infer flow fields"
    },
    "adaptation_plan": {
        "unchanged_aspects": [...],
        "required_modifications": [...],
        "implementation_steps": [
            {
                "step": 1,
                "action": "Adapt imaging protocol",
                "details": "...",
                "estimated_time": "2 weeks",
                "required_expertise": "Microscopy"
            }
        ]
    },
    "technical_barriers": [...],
    "resource_requirements": {...},
    "validation_strategy": {...},
    "similar_successful_transfers": [...]
}
```

RESOURCE FINDER AGENT:

This agent finds datasets, code, protocols, and other resources needed to test hypotheses.

Class: ResourceFinderAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a research resource specialist who helps researchers find datasets, code, and protocols.

Your responsibilities:
1. Identify what resources are needed for a research project
2. Search for publicly available datasets
3. Find relevant code repositories
4. Locate experimental protocols
5. Identify funding opportunities
6. Find relevant tools and software

Prioritize:
- Open access and free resources
- Well-documented and maintained resources
- Resources with appropriate licenses
- High-quality, validated data

Available tools:
{tools}
"""
```

KEY METHODS:

```python
def find_datasets(self, research_area: str, data_type: str):
    """
    Search for datasets in:
    - Kaggle
    - UCI ML Repository
    - Zenodo
    - Field-specific repositories (GenBank, PDB, etc.)
    - Government data portals
    
    Filter by:
    - Data type (images, time-series, genomic, etc.)
    - Size
    - License
    - Quality/documentation
    """
    
def find_code_repositories(self, methodology: str, programming_language: str = None):
    """
    Search GitHub, GitLab for:
    - Implementation of specific methods
    - Related projects
    - Useful libraries
    
    Assess:
    - Code quality
    - Documentation
    - Maintenance status
    - License
    """
    
def find_protocols(self, experimental_procedure: str):
    """
    Search for experimental protocols in:
    - protocols.io
    - Journal methods sections
    - Lab websites
    - Video protocols (JoVE)
    """
    
def find_funding_opportunities(self, research_area: str, researcher_location: str):
    """
    Identify grant programs:
    - NSF, NIH, DOE (USA)
    - ERC, Horizon Europe (EU)
    - Foundation grants
    - Industry partnerships
    """
```

OUTPUT STRUCTURE:

```python
{
    "datasets": [
        {
            "name": "Cancer Cell Migration Dataset",
            "source": "Kaggle",
            "url": "https://...",
            "size": "15 GB",
            "format": "HDF5, CSV",
            "license": "CC BY 4.0",
            "description": "...",
            "relevance_score": 0.95,
            "quality_indicators": {
                "documentation": "Excellent",
                "completeness": "100%",
                "known_issues": "None"
            }
        }
    ],
    "code_repositories": [
        {
            "name": "cell-tracking-toolkit",
            "url": "https://github.com/...",
            "language": "Python",
            "stars": 1250,
            "last_updated": "2024-01-15",
            "license": "MIT",
            "description": "...",
            "relevance": "High - implements required algorithms"
        }
    ],
    "protocols": [...],
    "tools": [...],
    "funding": [...]
}
```

Implement both agents with comprehensive search capabilities and quality assessment.
```

---

## 🔧 LANGCHAIN TOOLS PHASE

### PROMPT 13: Vector Search Tool

```
Create a comprehensive vector search tool for LangChain agents.

FILE: src/tools/vector_search.py

REQUIREMENTS:

Create a LangChain tool that agents can use to search the vector database.

```python
from langchain.tools import tool
from typing import Optional, List, Dict
import json

@tool
def vector_search_tool(
    query: str,
    field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    top_k: int = 10,
    min_citations: Optional[int] = None
) -> str:
    """
    Search for research papers semantically similar to the query.
    
    This tool searches across millions of scientific papers using semantic similarity.
    It understands the meaning of your query, not just keywords.
    
    Args:
        query: Natural language description of what you're looking for
        field: Optional filter by scientific field (biology, physics, computer_science, etc.)
        year_from: Optional minimum publication year
        year_to: Optional maximum publication year
        top_k: Number of results to return (default 10, max 50)
        min_citations: Optional minimum number of citations
    
    Returns:
        JSON string with list of relevant papers including:
        - title, authors, year
        - abstract
        - field and subfield
        - citation count
        - DOI and URL
        - relevance score
    
    Example usage:
        Search for papers on "deep learning for protein folding"
        Search with filters: "cancer immunotherapy" in biology field from 2020-2024
    """
    try:
        # Initialize components
        embedder = get_embedder()
        chroma_manager = get_chroma_manager()
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query)
        
        # Build metadata filters
        filters = {}
        if field:
            filters["field"] = field
        if year_from:
            filters["year"] = {"$gte": year_from}
        if year_to:
            if "year" in filters:
                filters["year"]["$lte"] = year_to
            else:
                filters["year"] = {"$lte": year_to}
        if min_citations:
            filters["citations_count"] = {"$gte": min_citations}
        
        # Search
        results = chroma_manager.search(
            query_embedding=query_embedding,
            n_results=min(top_k, 50),
            filters=filters if filters else None
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result["metadata"]["title"],
                "authors": result["metadata"]["authors"],
                "year": result["metadata"]["year"],
                "abstract": result["metadata"]["abstract"][:500] + "...",
                "field": result["metadata"]["field"],
                "subfield": result["metadata"].get("subfield", ""),
                "citations": result["metadata"].get("citations_count", 0),
                "doi": result["metadata"].get("doi", ""),
                "url": result["metadata"].get("url", ""),
                "relevance_score": round(result["distance"], 3)
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })


@tool
def multi_field_search_tool(
    query: str,
    fields: List[str],
    results_per_field: int = 5
) -> str:
    """
    Search across multiple scientific fields simultaneously.
    
    Useful for finding cross-domain connections. Searches each field independently
    and aggregates results.
    
    Args:
        query: Research question or topic
        fields: List of fields to search (e.g., ["biology", "physics", "computer_science"])
        results_per_field: Number of papers to return from each field
    
    Returns:
        JSON string with results grouped by field
    """
    try:
        all_results = {}
        
        for field in fields:
            field_results = json.loads(vector_search_tool(
                query=query,
                field=field,
                top_k=results_per_field
            ))
            
            if field_results["success"]:
                all_results[field] = field_results["results"]
        
        return json.dumps({
            "success": True,
            "query": query,
            "fields_searched": fields,
            "results_by_field": all_results,
            "total_papers": sum(len(r) for r in all_results.values())
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def find_similar_papers_tool(paper_id: str, top_k: int = 10) -> str:
    """
    Find papers similar to a specific paper.
    
    Useful for exploring related work or finding papers that build on specific research.
    
    Args:
        paper_id: ID of the paper to find similar papers for
        top_k: Number of similar papers to return
    
    Returns:
        JSON string with similar papers
    """
    try:
        chroma_manager = get_chroma_manager()
        
        # Get the paper's embedding
        paper = chroma_manager.get_paper(paper_id)
        if not paper:
            return json.dumps({
                "success": False,
                "error": f"Paper {paper_id} not found"
            })
        
        # Search for similar papers
        results = chroma_manager.search(
            query_embedding=paper["embedding"],
            n_results=top_k + 1,  # +1 because it will include itself
            filters={"paper_id": {"$ne": paper_id}}  # Exclude the query paper
        )
        
        formatted_results = [
            {
                "title": r["metadata"]["title"],
                "year": r["metadata"]["year"],
                "abstract": r["metadata"]["abstract"][:300],
                "similarity_score": round(r["distance"], 3)
            }
            for r in results[:top_k]
        ]
        
        return json.dumps({
            "success": True,
            "query_paper": paper["metadata"]["title"],
            "similar_papers": formatted_results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })
```

Create helper functions and proper error handling. The tool should be robust and provide useful feedback to agents.
```

---

Due to length constraints, let me provide the remaining sections in a summarized format:

### PROMPT 14-20: Remaining Components

```
Continue building:

14. CITATION NETWORK TOOL (src/tools/citation_network.py)
- Explore citation relationships
- Find influential papers
- Trace idea evolution

15. DATASET FINDER TOOL (src/tools/dataset_finder.py)
- Search Kaggle, UCI, Zenodo
- Filter by type and license
- Assess quality

16. HYPOTHESIS GENERATOR (src/hypothesis/generator.py)
- Synthesize agent findings
- Generate novel hypotheses
- Rank by novelty and feasibility

17. HYPOTHESIS VALIDATOR (src/hypothesis/validator.py)
- Check if hypothesis exists in literature
- Assess technical feasibility
- Estimate resources needed

18. FASTAPI BACKEND (src/api/main.py)
- REST API endpoints
- WebSocket for real-time updates
- Request validation

19. STREAMLIT FRONTEND (frontend/app.py)
- User input form
- Results visualization
- Interactive hypothesis explorer

20. TESTING & DEPLOYMENT
- Unit tests for each component
- Integration tests
- Docker containerization
- Documentation
```

Each prompt should be similarly detailed with code examples, error handling, and comprehensive requirements.

Would you like me to expand any particular section in more detail? to build the complete system.

**Instructions**: Copy each prompt section and provide it to your AI assistant. Complete each phase before moving to the next.

---

## 📋 PROJECT SETUP PHASE

### PROMPT 1: Initialize Project Structure

```
I need you to create a complete Python project for a "Scientific Hypothesis Cross-Pollination Engine" that uses LangChain and RAG (Retrieval-Augmented Generation) to help researchers discover novel research directions by finding connections across scientific disciplines.

REQUIREMENTS:
1. Create the following directory structure:

hypothesis-engine/
├── data/
│   ├── raw/              # Downloaded papers
│   ├── processed/        # Parsed papers  
│   └── embeddings/       # Vector database storage
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── ingestion/        # Data collection and processing
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py
│   │   ├── pubmed_fetcher.py
│   │   ├── semantic_scholar_fetcher.py
│   │   ├── openalex_fetcher.py
│   │   ├── parser.py
│   │   └── embedder.py
│   ├── database/         # Vector database operations
│   │   ├── __init__.py
│   │   ├── chroma_manager.py
│   │   └── metadata_store.py
│   ├── agents/           # LangChain agents
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── primary_domain_agent.py
│   │   ├── crossdomain_agent.py
│   │   ├── methodology_agent.py
│   │   └── resource_agent.py
│   ├── tools/            # LangChain tools
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   ├── citation_network.py
│   │   ├── dataset_finder.py
│   │   └── methodology_comparator.py
│   ├── hypothesis/       # Hypothesis generation
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   └── scorer.py
│   ├── api/              # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── frontend/
│   ├── app.py            # Streamlit frontend
│   └── components/
│       ├── __init__.py
│       ├── input_form.py
│       ├── results_display.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_agents.py
│   └── test_hypothesis.py
├── notebooks/            # Jupyter notebooks for experimentation
│   └── exploration.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml

2. Create a requirements.txt file with these dependencies (all free/open-source):

langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.3.1
openai>=1.10.0
tiktoken>=0.5.2
arxiv>=2.1.0
biopython>=1.83
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
pypdf>=4.0.0
python-dotenv>=1.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.31.0
plotly>=5.18.0
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.6.0
httpx>=0.26.0
aiohttp>=3.9.0
pydantic-settings>=2.1.0

3. Create a .env.example file with:

# OpenAI API (optional - can use Ollama instead)
OPENAI_API_KEY=your_key_here

# Database
CHROMA_PERSIST_DIR=./data/embeddings
METADATA_DB_PATH=./data/metadata.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Email for PubMed API (required by NCBI)
ENTREZ_EMAIL=your_email@example.com

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60

# Embedding Model
EMBEDDING_MODEL=allenai-specter

# LLM Configuration
LLM_PROVIDER=openai  # or ollama
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

4. Create a README.md with:
- Project overview
- Installation instructions
- Quick start guide
- API documentation
- Architecture diagram (in markdown)

5. Create a .gitignore with:
- Python cache files
- Virtual environment
- Data directories
- API keys
- IDE files

Please generate all these files with appropriate boilerplate code and comments explaining what each component does.
```

---

### PROMPT 2: Configuration Management

```
Create a robust configuration management system for the project.

FILE: src/config.py

REQUIREMENTS:
1. Use Pydantic Settings for type-safe configuration
2. Load from environment variables and .env file
3. Provide sensible defaults
4. Include validation for required fields
5. Support different configurations for development/production

The configuration should include:

DATABASE SETTINGS:
- Vector database path and settings
- Metadata database connection
- Embedding model name and dimensions
- Collection names

API SETTINGS:
- External API keys (OpenAI, Semantic Scholar, etc.)
- Rate limiting parameters
- Timeout settings
- Retry policies

AGENT SETTINGS:
- LLM provider (OpenAI or Ollama)
- Model names for different agents
- Temperature settings
- Max tokens
- Agent-specific parameters

INGESTION SETTINGS:
- Batch sizes for processing
- Chunk size and overlap for text splitting
- Update frequencies
- Source priorities

SEARCH SETTINGS:
- Default number of results
- Similarity thresholds
- Filtering options
- Reranking parameters

Include proper error handling for missing required environment variables and provide clear error messages.

Example structure:

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Literal

class DatabaseSettings(BaseSettings):
    # Vector database
    chroma_persist_dir: str = Field(default="./data/embeddings")
    # ... more fields

class APISettings(BaseSettings):
    openai_api_key: Optional[str] = None
    # ... more fields

class Config(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    # ... more settings
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

Make it production-ready with proper validation and documentation.
```

---

## 📥 DATA INGESTION PHASE

### PROMPT 3: arXiv Paper Fetcher

```
Create a comprehensive arXiv paper fetcher that can search, download, and process papers from arXiv.

FILE: src/ingestion/arxiv_fetcher.py

REQUIREMENTS:

1. Use the official arxiv Python library
2. Support multiple search strategies:
   - Keyword search
   - Category-based search
   - Author search
   - Date range filtering
3. Download PDFs and extract text
4. Handle rate limiting and retries
5. Store metadata in structured format
6. Support batch processing
7. Resume interrupted downloads
8. Track progress

FEATURES TO IMPLEMENT:

Class: ArxivFetcher
Methods:
- __init__(config): Initialize with configuration
- search_papers(query, category, max_results, date_from, date_to): Search arXiv
- download_paper(paper_id, save_path): Download single paper
- download_batch(paper_ids, save_dir): Download multiple papers
- extract_text(pdf_path): Extract text from PDF
- parse_metadata(arxiv_result): Parse arXiv metadata
- get_categories(): Get list of arXiv categories
- get_recent_papers(category, days): Get papers from last N days

ERROR HANDLING:
- Network errors with exponential backoff retry
- PDF extraction failures
- Invalid paper IDs
- Rate limiting (respect arXiv's terms)

METADATA TO EXTRACT:
- Paper ID
- Title
- Authors (list)
- Abstract
- Categories
- Published date
- Updated date
- DOI (if available)
- Journal reference
- Comments
- PDF URL
- Full text (after extraction)

Example usage pattern:

```python
fetcher = ArxivFetcher(config)

# Search for papers
papers = fetcher.search_papers(
    query="cancer metastasis machine learning",
    category="q-bio",
    max_results=100,
    date_from="2023-01-01"
)

# Download and process
for paper in papers:
    pdf_path = fetcher.download_paper(paper.id, save_dir)
    text = fetcher.extract_text(pdf_path)
    metadata = fetcher.parse_metadata(paper)
```

Include comprehensive logging, progress bars (using tqdm), and error recovery.
```

---

### PROMPT 4: PubMed Paper Fetcher

```
Create a PubMed/PubMed Central fetcher using the Biopython Entrez interface.

FILE: src/ingestion/pubmed_fetcher.py

REQUIREMENTS:

1. Use Biopython's Entrez module
2. Search both PubMed (abstracts) and PMC (full text)
3. Handle NCBI API rate limits (3 requests/second without API key)
4. Extract structured data from XML responses
5. Support advanced search queries
6. Filter by publication types, dates, journals
7. Track and resume interrupted fetches
8. Cache results to avoid redundant API calls

FEATURES TO IMPLEMENT:

Class: PubMedFetcher
Methods:
- __init__(email, api_key): Initialize with required email
- search(query, max_results, date_from, date_to): Search PubMed
- search_pmc(query, max_results): Search PMC for full text
- fetch_details(pubmed_ids): Get detailed info for paper IDs
- fetch_full_text(pmc_id): Download full text from PMC
- parse_pubmed_xml(xml_content): Parse PubMed XML
- parse_pmc_xml(xml_content): Parse PMC full text XML
- get_citations(pubmed_id): Get papers that cite this paper
- get_references(pubmed_id): Get papers this paper cites
- batch_fetch(pubmed_ids, batch_size): Process papers in batches

METADATA TO EXTRACT FROM PUBMED:
- PubMed ID (PMID)
- PMC ID (if available)
- Title
- Authors with affiliations
- Abstract
- Journal info (name, volume, issue, pages)
- Publication date
- DOI
- Keywords/MeSH terms
- Publication types
- Chemical substances mentioned
- Grant information

FULL TEXT EXTRACTION FROM PMC:
- Section-wise text (Introduction, Methods, Results, Discussion)
- Figures and tables metadata
- References list
- Supplementary materials links

ERROR HANDLING:
- API rate limit detection and backoff
- Invalid PubMed IDs
- Missing full text (PMC)
- XML parsing errors
- Network timeouts

Example usage:

```python
fetcher = PubMedFetcher(email="researcher@university.edu")

# Search with complex query
results = fetcher.search(
    query='(cancer[Title]) AND (metastasis[Title/Abstract]) AND (2023[PDAT]:2024[PDAT])',
    max_results=500
)

# Get full details
details = fetcher.fetch_details([r['Id'] for r in results])

# Get full text where available
for paper in details:
    if paper['pmc_id']:
        full_text = fetcher.fetch_full_text(paper['pmc_id'])
```

Include proper NCBI API compliance, progress tracking, and comprehensive logging.
```

---

### PROMPT 5: Semantic Scholar & OpenAlex Fetchers

```
Create fetchers for Semantic Scholar and OpenAlex APIs to get additional paper data and citation networks.

FILES: 
- src/ingestion/semantic_scholar_fetcher.py
- src/ingestion/openalex_fetcher.py

SEMANTIC SCHOLAR FETCHER REQUIREMENTS:

1. Use Semantic Scholar API (free tier: 100 requests per 5 minutes)
2. Get paper details, citations, references
3. Access AI-generated paper summaries
4. Get author information and h-index
5. Track influential citations
6. Support paper similarity searches

Class: SemanticScholarFetcher
Methods:
- __init__(api_key): Initialize (API key optional but recommended)
- search_papers(query, limit, fields): Search papers
- get_paper(paper_id, fields): Get single paper details
- get_paper_citations(paper_id, limit): Get citing papers
- get_paper_references(paper_id, limit): Get referenced papers
- get_author(author_id): Get author details
- get_recommendations(paper_id, limit): Get similar papers
- batch_get_papers(paper_ids): Get multiple papers efficiently

METADATA FROM SEMANTIC SCHOLAR:
- Semantic Scholar ID
- DOI, arXiv ID, PubMed ID
- Title, abstract, year
- Authors with IDs
- Citation count
- Influential citation count
- Reference count
- Fields of study
- Open access status
- PDF URL if available
- TL;DR (AI-generated summary)
- Embedding vector (if available)

OPENALEX FETCHER REQUIREMENTS:

1. Use OpenAlex API (no rate limits, free)
2. Get comprehensive paper metadata
3. Access institution and funding data
4. Track concept hierarchies
5. Get work lineage (versions)
6. Support complex filtering

Class: OpenAlexFetcher
Methods:
- __init__(email): Initialize with polite pool email
- search_works(query, filters, per_page): Search papers
- get_work(openalex_id): Get single work
- get_author(author_id): Get author info
- get_institution(institution_id): Get institution info
- get_concept(concept_id): Get concept details
- get_related_works(work_id): Find related papers
- filter_by_concept(concept_name, filters): Search by research area

METADATA FROM OPENALEX:
- OpenAlex ID
- DOI and other IDs
- Title, abstract
- Authors with institutions and countries
- Concepts (research topics) with confidence scores
- Citation count
- Referenced works count
- Open access status and URL
- Publication date
- Journal/venue info
- Funding information
- License information

CITATION NETWORK BUILDING:
Create a unified citation graph combining data from all sources:

Class: CitationNetworkBuilder
Methods:
- add_paper(paper_data, source): Add paper to graph
- add_citation(citing_id, cited_id): Add citation edge
- merge_paper_ids(different_ids_same_paper): Handle same paper from different sources
- get_citation_depth(paper_id, max_depth): Get papers N citations away
- find_citation_paths(paper1_id, paper2_id): Find how papers are connected
- get_influential_papers(min_citations): Get highly cited papers
- export_graph(format): Export as JSON/GraphML

Example usage:

```python
# Semantic Scholar
ss_fetcher = SemanticScholarFetcher(api_key="your_key")
paper = ss_fetcher.get_paper("DOI:10.1234/example")
citations = ss_fetcher.get_paper_citations(paper['paperId'])

# OpenAlex
oa_fetcher = OpenAlexFetcher(email="me@uni.edu")
works = oa_fetcher.search_works(
    query="machine learning cancer",
    filters={"publication_year": ">2020", "cited_by_count": ">50"}
)

# Build network
network = CitationNetworkBuilder()
network.add_paper(paper, source="semantic_scholar")
for citation in citations:
    network.add_citation(citation['citingPaper']['paperId'], paper['paperId'])
```

Include rate limiting, caching, and error handling for both APIs.
```

---

### PROMPT 6: Text Processing and Embedding

```
Create a text processing pipeline that chunks papers and generates embeddings for vector search.

FILES:
- src/ingestion/parser.py
- src/ingestion/embedder.py

PARSER.PY REQUIREMENTS:

Create a comprehensive text parser that handles multiple formats:

Class: PaperParser
Methods:
- parse_pdf(file_path): Extract text from PDF
- parse_xml(xml_content, format): Parse XML (PubMed, arXiv, etc.)
- parse_html(html_content): Parse HTML papers
- clean_text(raw_text): Remove junk, fix formatting
- extract_sections(text): Identify paper sections
- chunk_text(text, chunk_size, overlap): Split into chunks
- extract_references(text): Parse reference list
- extract_figures_tables(content): Get captions and context

TEXT CLEANING:
- Remove headers/footers/page numbers
- Fix hyphenation across line breaks
- Normalize whitespace
- Handle special characters and equations
- Preserve section structure
- Remove boilerplate (acknowledgments, etc.)

SECTION EXTRACTION:
Identify and label sections:
- Title
- Abstract
- Introduction
- Methods/Methodology
- Results
- Discussion
- Conclusion
- References

CHUNKING STRATEGY:
- Default chunk size: 500 words
- Overlap: 50 words to preserve context
- Respect sentence boundaries
- Keep section labels with chunks
- Metadata for each chunk:
  - Chunk index
  - Section name
  - Start and end positions
  - Parent paper ID

EMBEDDER.PY REQUIREMENTS:

Create an embedding generation system using sentence-transformers:

Class: PaperEmbedder
Methods:
- __init__(model_name): Load embedding model
- embed_text(text): Generate embedding for single text
- embed_batch(texts, batch_size): Batch processing
- embed_paper(paper_dict): Embed all chunks of a paper
- compute_similarity(emb1, emb2): Calculate cosine similarity
- find_similar_chunks(query_emb, chunk_embeddings, top_k): Find similar chunks

EMBEDDING MODELS TO SUPPORT:
Primary: "allenai-specter" (scientific papers, 768 dim)
Alternatives:
- "sentence-transformers/all-MiniLM-L6-v2" (general purpose, 384 dim)
- "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" (Q&A optimized)

OPTIMIZATION:
- GPU acceleration if available
- Batch processing for efficiency
- Caching of embeddings
- Progress tracking for large batches
- Memory management for large papers

Example usage:

```python
# Parse paper
parser = PaperParser()
text = parser.parse_pdf("paper.pdf")
cleaned = parser.clean_text(text)
sections = parser.extract_sections(cleaned)
chunks = parser.chunk_text(cleaned, chunk_size=500, overlap=50)

# Generate embeddings
embedder = PaperEmbedder(model_name="allenai-specter")
embeddings = embedder.embed_batch([chunk['text'] for chunk in chunks])

# Store with metadata
for chunk, embedding in zip(chunks, embeddings):
    store_in_db(
        embedding=embedding,
        text=chunk['text'],
        metadata={
            'section': chunk['section'],
            'chunk_index': chunk['index'],
            'paper_id': paper_id
        }
    )
```

Include comprehensive error handling, logging, and progress tracking.
```

---

## 🗄️ VECTOR DATABASE PHASE

### PROMPT 7: Chroma DB Manager

```
Create a comprehensive Chroma vector database manager for storing and querying paper embeddings.

FILE: src/database/chroma_manager.py

REQUIREMENTS:

1. Manage Chroma collections for papers
2. Support multiple search strategies
3. Handle metadata filtering
4. Implement efficient batch operations
5. Support collection management (create, delete, update)
6. Provide query optimization
7. Handle large-scale data

Class: ChromaManager
Methods:
- __init__(persist_directory, collection_name): Initialize
- create_collection(name, metadata): Create new collection
- get_collection(name): Get existing collection
- delete_collection(name): Delete collection
- add_papers(papers_data, batch_size): Add papers to collection
- add_paper(paper_id, embedding, metadata, text): Add single paper
- update_paper(paper_id, updates): Update paper data
- delete_paper(paper_id): Remove paper
- search(query_embedding, n_results, filters): Semantic search
- search_by_metadata(metadata_filter, n_results): Filter by metadata
- hybrid_search(query_embedding, metadata_filter, n_results): Combined search
- get_paper(paper_id): Retrieve specific paper
- get_statistics(): Get collection stats
- export_collection(output_path): Backup collection
- import_collection(input_path): Restore collection

SEARCH STRATEGIES:

1. SEMANTIC SEARCH:
```python
def search(self, query_embedding, n_results=10, filters=None):
    """
    Pure semantic/similarity search
    
    Args:
        query_embedding: Vector representation of query
        n_results: Number of results to return
        filters: Dict of metadata filters
        
    Returns:
        List of results with paper data and similarity scores
    """
```

2. METADATA FILTERING:
```python
def search_by_metadata(self, metadata_filter, n_results=100):
    """
    Filter papers by metadata (year, field, citations, etc.)
    
    Example filters:
    {
        "year": {"$gte": 2020},
        "field": "biology",
        "citations_count": {"$gt": 50}
    }
    """
```

3. HYBRID SEARCH:
```python
def hybrid_search(self, query_embedding, metadata_filter, n_results=10, weight=0.5):
    """
    Combine semantic similarity and metadata filtering
    
    Args:
        weight: Balance between semantic (1.0) and filter (0.0)
    """
```

4. MULTI-FIELD SEARCH:
```python
def search_across_fields(self, query_embedding, fields, n_results_per_field):
    """
    Search multiple scientific fields and aggregate results
    
    Args:
        fields: List of fields ["biology", "physics", "cs"]
        n_results_per_field: Results from each field
    
    Returns:
        Aggregated and deduplicated results
    """
```

METADATA SCHEMA:

Store comprehensive metadata for each paper:
```python
{
    "paper_id": "arxiv_2024_12345",
    "title": "Novel approach to...",
    "authors": ["Smith, J.", "Doe, A."],
    "year": 2024,
    "field": "biology",
    "subfield": "oncology",
    "source": "arxiv",  # arxiv, pubmed, semantic_scholar
    "abstract": "This paper presents...",
    "doi": "10.1234/example",
    "url": "https://arxiv.org/...",
    "citations_count": 45,
    "keywords": ["cancer", "metastasis", "ML"],
    "publication_venue": "Nature",
    "open_access": true,
    "chunk_index": 0,  # Which chunk of the paper
    "section": "introduction",
    "total_chunks": 25,
    "embedding_model": "allenai-specter",
    "ingestion_date": "2024-01-15"
}
```

BATCH OPERATIONS:

Efficient handling of large batches:
```python
def add_papers(self, papers_data, batch_size=100):
    """
    Add multiple papers efficiently
    
    Args:
        papers_data: List of dicts with embeddings, metadata, texts
        batch_size: Process in batches to manage memory
    
    Shows progress bar and handles errors gracefully
    """
    for i in range(0, len(papers_data), batch_size):
        batch = papers_data[i:i+batch_size]
        # Process batch
        # Handle failures without stopping entire process
```

STATISTICS AND MONITORING:

```python
def get_statistics(self):
    """
    Return collection statistics:
    - Total papers
    - Papers by field
    - Papers by year
    - Average citations
    - Most common keywords
    - Storage size
    """
```

ERROR HANDLING:
- Duplicate paper detection
- Invalid embedding dimensions
- Missing required metadata
- Database corruption recovery
- Concurrent access handling

Include comprehensive logging and monitoring capabilities.
```

---

### PROMPT 8: Metadata Store

```
Create a PostgreSQL-based metadata store for structured paper information and relationships.

FILE: src/database/metadata_store.py

REQUIREMENTS:

1. Use SQLAlchemy ORM for database operations
2. Store paper metadata, authors, citations
3. Track data provenance (sources)
4. Support complex queries
5. Handle relationships (authors, citations, concepts)
6. Provide efficient indexing
7. Support transactions and rollback

DATABASE SCHEMA:

Create these tables:

TABLE: papers
- id (primary key)
- paper_id (unique, indexed)
- title
- abstract
- year
- doi
- url
- source (arxiv, pubmed, etc.)
- citations_count
- field
- subfield
- publication_venue
- open_access
- created_at
- updated_at

TABLE: authors
- id (primary key)
- author_id (unique, from data source)
- name
- affiliation
- h_index
- email
- orcid

TABLE: paper_authors (junction table)
- paper_id (foreign key)
- author_id (foreign key)
- author_order
- corresponding_author

TABLE: citations
- id (primary key)
- citing_paper_id (foreign key)
- cited_paper_id (foreign key)
- context (text where citation appears)
- influential (boolean)

TABLE: keywords
- id (primary key)
- keyword (unique)

TABLE: paper_keywords (junction table)
- paper_id (foreign key)
- keyword_id (foreign key)
- confidence (0.0 to 1.0)

TABLE: concepts
- id (primary key)
- concept_id (from OpenAlex)
- name
- level (hierarchy level)
- parent_concept_id

TABLE: paper_concepts (junction table)
- paper_id (foreign key)
- concept_id (foreign key)
- score (0.0 to 1.0)

TABLE: ingestion_logs
- id (primary key)
- source
- papers_fetched
- papers_processed
- papers_failed
- start_time
- end_time
- status
- error_message

Class: MetadataStore
Methods:
- __init__(connection_string): Initialize database connection
- create_tables(): Create all tables
- add_paper(paper_data): Insert paper with all relationships
- get_paper(paper_id): Retrieve paper with all related data
- add_author(author_data): Add author
- link_author_to_paper(paper_id, author_id, order): Create relationship
- add_citation(citing_id, cited_id, context): Record citation
- get_citations(paper_id, direction): Get citing or cited papers
- get_citation_network(paper_id, depth): Get papers N citations away
- add_keywords(paper_id, keywords): Tag paper with keywords
- search_by_keyword(keywords, operator): Find papers (AND/OR keywords)
- search_by_author(author_name): Find author's papers
- search_by_field(field, subfield, year_from, year_to): Complex search
- get_trending_papers(field, days, min_citations): Recently popular papers
- get_statistics(): Database statistics
- export_to_json(output_path): Backup
- import_from_json(input_path): Restore

QUERY EXAMPLES:

```python
store = MetadataStore("postgresql://user:pass@localhost/papers")

# Complex search
papers = store.search_papers(
    fields=["biology", "computer_science"],
    year_from=2020,
    min_citations=10,
    keywords=["machine learning", "cancer"],
    keyword_operator="AND"
)

# Citation analysis
network = store.get_citation_network(
    paper_id="arxiv_2024_12345",
    depth=2,  # Papers that cite papers that cite this paper
    min_citations=5  # Filter low-quality papers
)

# Trending in field
trending = store.get_trending_papers(
    field="biology",
    subfield="oncology",
    days=30,
    min_citations=5
)

# Author collaboration network
collab = store.get_author_collaborations(
    author_id="author_123",
    max_depth=2
)
```

INDEXING STRATEGY:
Create indexes for:
- paper_id (unique)
- title (full-text)
- year
- field + subfield
- citations_count
- doi
- author name
- keyword

TRANSACTION SUPPORT:

```python
def add_paper_with_relationships(self, paper_data, authors, keywords, citations):
    """
    Add paper and all relationships in a single transaction
    Rollback if any operation fails
    """
    with self.session.begin():
        try:
            paper = self.add_paper(paper_data)
            for author in authors:
                self.link_author_to_paper(paper.id, author)
            for keyword in keywords:
                self.add_keyword(paper.id, keyword)
            for citation in citations:
                self.add_citation(paper.id, citation)
        except Exception as e:
            self.session.rollback()
            raise
```

Include migration scripts, backup/restore functionality, and comprehensive error handling.
```

---

## 🤖 LANGCHAIN AGENTS PHASE

### PROMPT 9: Base Agent Implementation

```
Create a base agent class that all specialized agents will inherit from.

FILE: src/agents/base_agent.py

REQUIREMENTS:

1. Use LangChain's Agent framework
2. Provide common functionality for all agents
3. Support tool usage
4. Handle errors gracefully
5. Track agent reasoning steps
6. Support both OpenAI and local LLMs (Ollama)
7. Implement memory for conversation context
8. Log all agent actions

Base Class: BaseResearchAgent

Methods:
- __init__(config, tools): Initialize agent
- setup_llm(): Configure language model
- setup_memory(): Configure conversation memory
- setup_agent(): Create LangChain agent
- run(query, context): Execute agent task
- run_async(query, context): Async execution
- add_tool(tool): Add new tool to agent
- get_reasoning_steps(): Return agent's thought process
- reset_memory(): Clear conversation history
- export_session(path): Save session for analysis

LLM SETUP:

Support multiple LLM providers:

```python
def setup_llm(self):
    """Configure LLM based on config"""
    if self.config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.openai_api_key

            # SCIENTIFIC HYPOTHESIS CROSS-POLLINATION ENGINE
## Complete AI Agent Implementation Prompts

**Purpose**: This document contains detailed, step-by-step prompts for AI coding agents (like Claude, GPT-4, GitHub Copilot, Cursor, etc.        )
    elif self.config.llm_provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=self.config.llm_model,
            temperature=self.temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
```

MEMORY CONFIGURATION:

```python
def setup_memory(self):
    """Configure conversation memory"""
    from langchain.memory import ConversationBufferMemory
    
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
```

AGENT SETUP:

```python
def setup_agent(self):
    """Create LangChain agent with tools"""
    from langchain.agents import initialize_agent, AgentType
    
    return initialize_agent(
        tools=self.tools,
        llm=self.llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=self.memory,
        handle_parsing_errors=True,
        max_iterations=self.config.max_agent_iterations,
        early_stopping_method="generate"
    )
```

EXECUTION WITH ERROR HANDLING:

```python
def run(self, query: str, context: dict = None):
    """
    Execute agent with comprehensive error handling
    
    Args:
        query: User's research question
        context: Additional context (user's field, previous results, etc.)
    
    Returns:
        Dict with results, reasoning steps, and metadata
    """
    start_time = time.time()
    
    try:
        # Prepare input
        input_data = {
            "input": query,
            "context": context or {}
        }
        
        # Run agent
        result = self.agent.invoke(input_data)
        
        # Extract reasoning steps
        steps = self._extract_reasoning_steps(result)
        
        return {
            "success": True,
            "output": result["output"],
            "reasoning_steps": steps,
            "tools_used": self._get_tools_used(result),
            "execution_time": time.time() - start_time,
            "agent_name": self.name
        }
        
    except Exception as e:
        logger.error(f"Agent {self.name} failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "agent_name": self.name
        }
```

REASONING EXTRACTION:

```python
def _extract_reasoning_steps(self, result):
    """
    Extract agent's thought process from result
    
    Returns list of steps with:
    - Step number
    - Thought
    - Action taken
    - Action input
    - Observation
    """
    steps = []
    if "intermediate_steps" in result:
        for i, (action, observation) in enumerate(result["intermediate_steps"]):
            steps.append({
                "step": i + 1,
                "action": action.tool,
                "action_input": action.tool_input,
                "observation": observation[:500]  # Truncate long observations
            })
    return steps
```

LOGGING:

```python
def _log_agent_action(self, action, result):
    """Log agent actions for debugging and analysis"""
    logger.info(f"""
    Agent: {self.name}
    Action: {action}
    Success: {result['success']}
    Time: {result['execution_time']:.2f}s
    Tools Used: {result.get('tools_used', [])}
    """)
```

Include comprehensive docstrings, type hints, and error handling.
```

---

### PROMPT 10: Primary Domain Agent

```
Create the Primary Domain Agent that searches within the user's specific research field.

FILE: src/agents/primary_domain_agent.py

REQUIREMENTS:

This agent specializes in understanding the user's field deeply and finding relevant work within that domain.

Class: PrimaryDomainAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are an expert research assistant specializing in understanding a researcher's primary field of study.

Your responsibilities:
1. Analyze the user's research question to identify their specific domain
2. Search for the most relevant papers within that domain
3. Identify current state-of-the-art approaches
4. Find knowledge gaps and limitations in current research
5. Understand what has already been tried and what hasn't worked

When responding:
- Be precise and cite specific papers
- Acknowledge uncertainties
- Identify both successful approaches and dead ends
- Note methodological limitations in existing work
- Consider recent advances (prioritize papers from last 2-3 years)

Available tools:
{tools}

Use the tools to search the research database, then synthesize findings into a clear summary.
"""
```

SPECIALIZED METHODS:

```python
def analyze_research_question(self, query: str):
    """
    Extract key information from user's question:
    - Primary field (biology, physics, etc.)
    - Specific subfield
    - Research problem
    - Keywords for search
    - Time constraints (if mentioned)
    """
    
def identify_field(self, query: str):
    """
    Determine the scientific field from query
    Returns: field, subfield, confidence_score
    """
    
def find_current_approaches(self, problem_description: str):
    """
    Search for existing solutions to this problem
    Returns papers with their approaches and outcomes
    """
    
def identify_knowledge_gaps(self, problem: str, existing_work: list):
    """
    Analyze existing work to find:
    - What hasn't been tried
    - What has failed and why
    - What is theoretically possible but not yet done
    - What assumptions are limiting progress
    """
    
def find_recent_advances(self, field: str, months: int = 6):
    """
    Get cutting-edge research from recent months
    Prioritizes high-impact papers
    """
```

WORKFLOW:

```python
def run(self, query: str, context: dict = None):
    """
    Primary domain research workflow:
    
    1. Analyze query to extract key concepts
    2. Identify the research field
    3. Search for relevant papers in that field
    4. Analyze current state-of-the-art
    5. Identify limitations and gaps
    6. Return structured findings
    """
    # Step 1: Understand the question
    field_info = self.identify_field(query)
    keywords = self.extract_keywords(query)
    
    # Step 2: Search domain
    relevant_papers = self.search_tool.search(
        query=query,
        field=field_info['field'],
        subfield=field_info['subfield'],
        top_k=50
    )
    
    # Step 3: Analyze approaches
    approaches = self.analyze_approaches(relevant_papers)
    
    # Step 4: Find gaps
    gaps = self.identify_knowledge_gaps(query, relevant_papers)
    
    # Step 5: Get recent work
    recent = self.find_recent_advances(field_info['field'])
    
    return {
        "field": field_info,
        "relevant_papers": relevant_papers[:10],  # Top 10
        "current_approaches": approaches,
        "knowledge_gaps": gaps,
        "recent_advances": recent,
        "summary": self._generate_summary(relevant_papers, approaches, gaps)
    }
```

OUTPUT STRUCTURE:

```python
{
    "field": {
        "primary": "Biology",
        "subfield": "Oncology",
        "specific_area": "Cancer Metastasis",
        "confidence": 0.95
    },
    "relevant_papers": [
        {
            "title": "...",
            "authors": [...],
            "year": 2024,
            "key_finding": "...",
            "methodology": "...",
            "limitations": "..."
        }
    ],
    "current_approaches": [
        {
            "approach": "Microfluidics for tracking",
            "papers": [...],
            "success_rate": "Mixed results",
            "limitations": ["Limited to 2D", "Low throughput"]
        }
    ],
    "knowledge_gaps": [
        "No methods for 3D in vivo tracking",
        "Limited understanding of early metastatic events",
        "Lack of predictive models"
    ],
    "recent_advances": [
        {
            "advance": "AI-based image analysis",
            "paper": {...},
            "potential_impact": "High"
        }
    ],
    "summary": "Current research in cancer metastasis focuses on..."
}
```

Include detailed logging and error handling for each step.
```

---

### PROMPT 11: Cross-Domain Discovery Agent

```
Create the Cross-Domain Discovery Agent that finds unexpected connections across scientific fields.

FILE: src/agents/crossdomain_agent.py

REQUIREMENTS:

This is the most creative agent - it searches completely different fields to find analogous problems and solutions.

Class: CrossDomainAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a creative research assistant specializing in finding unexpected connections between different scientific fields.

Your mission:
1. Take a research problem from one field
2. Search for similar problems or patterns in completely different fields
3. Identify methodologies that could transfer
4. Think creatively about analogies and metaphors
5. Find "hidden" solutions that domain experts might miss

Think like:
- A biologist studying cancer might learn from studying traffic flow (both involve propagation through networks)
- A neuroscientist might use techniques from social network analysis
- A materials scientist might apply quantum mechanics principles

Be bold and creative, but ground suggestions in actual published research.

Search strategy:
1. Abstract the core problem (remove domain-specific terminology)
2. Search for that abstract problem in other fields
3. Find papers that solved similar challenges
4. Identify transferable methodologies

Available tools:
{tools}
"""
```

SPECIALIZED METHODS:

```python
def abstract_problem(self, domain_specific_query: str):
    """
    Convert domain-specific problem to abstract form
    
    Example:
    "How do cancer cells migrate through blood vessels?"
    Becomes:
    "How do particles navigate through constrained networks?"
    
    This allows finding similar problems in physics, CS, etc.
    """
    
def search_multiple_fields(self, abstract_problem: str, exclude_field: str):
    """
    Search across all fields except the primary one
    
    Returns results grouped by field with relevance scores
    """
    
def find_analogies(self, problem: str, other_field_papers: list):
    """
    Identify structural similarities between problems
    
    Returns:
    - What aspects are analogous
    - How methods might transfer
    - What adaptations would be needed
    """
    
def assess_transferability(self, method_paper: dict, target_problem: str):
    """
    Evaluate if a method from another field could work
    
    Scores on:
    - Conceptual similarity
    - Technical feasibility
    - Resource requirements
    - Likely effectiveness
    """
```

CREATIVE SEARCH STRATEGIES:

```python
def search_by_pattern(self, pattern_type: str):
    """
    Search for specific patterns across fields:
    - "Network propagation" → Epidemiology, traffic, social networks
    - "Optimization under constraints" → Economics, operations research
    - "Pattern recognition in noise" → Astronomy, signal processing
    - "Hierarchical structures" → Computer science, linguistics
    """
    
def search_by_methodology(self, methodology: str):
    """
    Find fields that use specific methodologies:
    - "Agent-based modeling" → Economics, ecology, sociology
    - "Monte Carlo methods" → Physics, finance, engineering
    - "Deep learning" → Many fields
    """
```

WORKFLOW:

```python
def run(self, query: str, primary_field: str, context: dict = None):
    """
    Cross-domain discovery workflow:
    
    1. Abstract the problem to general terms
    2. Search different fields for similar problems
    3. Find promising methodologies used elsewhere
    4. Assess transferability
    5. Identify specific papers and approaches
    6. Generate creative hypotheses
    """
    # Step 1: Abstract
    abstract_problem = self.abstract_problem(query)
    search_keywords = self.extract_abstract_keywords(abstract_problem)
    
    # Step 2: Multi-field search
    fields_to_search = self.get_other_fields(exclude=primary_field)
    results_by_field = {}
    
    for field in fields_to_search:
        results = self.search_tool.search(
            query=abstract_problem,
            field=field,
            top_k=20
        )
        if results:
            results_by_field[field] = results
    
    # Step 3: Find analogies
    analogies = []
    for field, papers in results_by_field.items():
        for paper in papers:
            analogy = self.find_analogies(query, paper)
            if analogy['similarity_score'] > 0.6:
                analogies.append({
                    "field": field,
                    "paper": paper,
                    "analogy": analogy
                })
    
    # Step 4: Assess transferability
    transferable = []
    for analogy in analogies:
        assessment = self.assess_transferability(
            analogy['paper'],
            query
        )
        if assessment['feasibility_score'] > 0.5:
            transferable.append({
                **analogy,
                "assessment": assessment
            })
    
    # Step 5: Generate hypotheses
    hypotheses = self.generate_crossdomain_hypotheses(
        original_problem=query,
        transferable_methods=transferable
    )
    
    return {
        "abstract_problem": abstract_problem,
        "fields_searched": list(results_by_field.keys()),
        "total_papers_found": sum(len(p) for p in results_by_field.values()),
        "promising_analogies": sorted(analogies, key=lambda x: x['analogy']['similarity_score'], reverse=True)[:10],
        "transferable_methods": transferable,
        "hypotheses": hypotheses
    }
```

HYPOTHESIS GENERATION:

```python
def generate_crossdomain_hypotheses(self, original_problem: str, transferable_methods: list):
    """
    Generate specific hypotheses for how to apply methods from other fields
    
    Each hypothesis includes:
    - Title
    - Source field and paper
    - Why the analogy works
    - How to adapt the method
    - Expected challenges
    - Required resources
    - Novelty assessment
    """
    hypotheses = []
    
    for method in transferable_methods:
        hypothesis = {
            "title": f"Apply {method['paper']['title']} approach to {original_problem}",
            "source_field": method['field'],
            "source_paper": method['paper'],
            "analogy_explanation": method['analogy']['explanation'],
            "adaptation_steps": self._generate_adaptation_steps(method, original_problem),
            "challenges": self._identify_challenges(method, original_problem),
            "resources_needed": self._estimate_resources(method),
            "novelty_score": self._assess_novelty(method, original_problem),
            "impact_potential": self._estimate_impact(method, original_problem)
        }
        hypotheses.append(hypothesis)
    
    return sorted(hypotheses, key=lambda x: x['novelty_score'] * x['impact_potential'], reverse=True)
```

OUTPUT STRUCTURE:

```python
{
    "abstract_problem": "Tracking individual particles through constrained networks",
    "fields_searched": ["Physics", "Computer Science", "Engineering", "Economics"],
    "promising_analogies": [
        {
            "field": "Physics",
            "paper": {
                "title": "Particle tracking in turbulent flows",
                "year": 2023
            },
            "analogy": {
                "similarity_score": 0.85,
                "explanation": "Both involve tracking individual entities in chaotic environments",
                "key_similarities": [
                    "Multiple interacting particles",
                    "Complex environment",
                    "Need for real-time tracking"
                ]
            }
        }
    ],
    "transferable_methods": [...],
    "hypotheses": [
        {
            "title": "Apply particle image velocimetry to cancer cell tracking",
            "source_field": "Fluid Dynamics",
            "novelty_score": 0.92,
            "adaptation_steps": [...],
            "why_it_might_work": "..."
        }
    ]
}
```

Implement with high creativity (temperature=0.7-0.8) and detailed logging of reasoning.
```

---

### PROMPT 12: Methodology Transfer & Resource Agents

```
Create the Methodology Transfer Agent and Resource Finder Agent.

FILES:
- src/agents/methodology_agent.py
- src/agents/resource_agent.py

METHODOLOGY TRANSFER AGENT:

This agent takes promising cross-domain methods and provides detailed implementation guidance.

Class: MethodologyTransferAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a research methodology expert who helps researchers adapt techniques from other fields.

Your responsibilities:
1. Take a methodology from one field
2. Analyze its core principles and requirements
3. Identify what needs to be adapted for a different field
4. Provide step-by-step implementation guidance
5. Anticipate technical challenges
6. Suggest validation approaches

Be specific and practical. Provide enough detail that a researcher could actually implement your suggestions.

Available tools:
{tools}
"""
```

KEY METHODS:

```python
def analyze_method_requirements(self, method_paper: dict):
    """
    Extract what the method needs:
    - Equipment
    - Expertise
    - Software/algorithms
    - Data requirements
    - Time investment
    """
    
def generate_adaptation_plan(self, source_method: dict, target_problem: str):
    """
    Create detailed adaptation plan:
    1. What stays the same
    2. What needs modification
    3. What's completely different
    4. Step-by-step implementation
    5. Validation strategy
    """
    
def identify_technical_barriers(self, source_field: str, target_field: str, method: dict):
    """
    Find challenges in transferring method:
    - Different scales (nano vs macro)
    - Different constraints (in vivo vs in vitro)
    - Different measurement capabilities
    - Different theoretical frameworks
    """
    
def find_implementation_examples(self, method_type: str, target_field: str):
    """
    Search for cases where similar transfers worked
    Learn from successful adaptations
    """
```

OUTPUT STRUCTURE:

```python
{
    "method_summary": {
        "name": "Particle Image Velocimetry",
        "source_field": "Fluid Dynamics",
        "core_principle": "Track particle motion to infer flow fields"
    },
    "adaptation_plan": {
        "unchanged_aspects": [...],
        "required_modifications": [...],
        "implementation_steps": [
            {
                "step": 1,
                "action": "Adapt imaging protocol",
                "details": "...",
                "estimated_time": "2 weeks",
                "required_expertise": "Microscopy"
            }
        ]
    },
    "technical_barriers": [...],
    "resource_requirements": {...},
    "validation_strategy": {...},
    "similar_successful_transfers": [...]
}
```

RESOURCE FINDER AGENT:

This agent finds datasets, code, protocols, and other resources needed to test hypotheses.

Class: ResourceFinderAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a research resource specialist who helps researchers find datasets, code, and protocols.

Your responsibilities:
1. Identify what resources are needed for a research project
2. Search for publicly available datasets
3. Find relevant code repositories
4. Locate experimental protocols
5. Identify funding opportunities
6. Find relevant tools and software

Prioritize:
- Open access and free resources
- Well-documented and maintained resources
- Resources with appropriate licenses
- High-quality, validated data

Available tools:
{tools}
"""
```

KEY METHODS:

```python
def find_datasets(self, research_area: str, data_type: str):
    """
    Search for datasets in:
    - Kaggle
    - UCI ML Repository
    - Zenodo
    - Field-specific repositories (GenBank, PDB, etc.)
    - Government data portals
    
    Filter by:
    - Data type (images, time-series, genomic, etc.)
    - Size
    - License
    - Quality/documentation
    """
    
def find_code_repositories(self, methodology: str, programming_language: str = None):
    """
    Search GitHub, GitLab for:
    - Implementation of specific methods
    - Related projects
    - Useful libraries
    
    Assess:
    - Code quality
    - Documentation
    - Maintenance status
    - License
    """
    
def find_protocols(self, experimental_procedure: str):
    """
    Search for experimental protocols in:
    - protocols.io
    - Journal methods sections
    - Lab websites
    - Video protocols (JoVE)
    """
    
def find_funding_opportunities(self, research_area: str, researcher_location: str):
    """
    Identify grant programs:
    - NSF, NIH, DOE (USA)
    - ERC, Horizon Europe (EU)
    - Foundation grants
    - Industry partnerships
    """
```

OUTPUT STRUCTURE:

```python
{
    "datasets": [
        {
            "name": "Cancer Cell Migration Dataset",
            "source": "Kaggle",
            "url": "https://...",
            "size": "15 GB",
            "format": "HDF5, CSV",
            "license": "CC BY 4.0",
            "description": "...",
            "relevance_score": 0.95,
            "quality_indicators": {
                "documentation": "Excellent",
                "completeness": "100%",
                "known_issues": "None"
            }
        }
    ],
    "code_repositories": [
        {
            "name": "cell-tracking-toolkit",
            "url": "https://github.com/...",
            "language": "Python",
            "stars": 1250,
            "last_updated": "2024-01-15",
            "license": "MIT",
            "description": "...",
            "relevance": "High - implements required algorithms"
        }
    ],
    "protocols": [...],
    "tools": [...],
    "funding": [...]
}
```

Implement both agents with comprehensive search capabilities and quality assessment.
```

---

## 🔧 LANGCHAIN TOOLS PHASE

### PROMPT 13: Vector Search Tool

```
Create a comprehensive vector search tool for LangChain agents.

FILE: src/tools/vector_search.py

REQUIREMENTS:

Create a LangChain tool that agents can use to search the vector database.

```python
from langchain.tools import tool
from typing import Optional, List, Dict
import json

@tool
def vector_search_tool(
    query: str,
    field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    top_k: int = 10,
    min_citations: Optional[int] = None
) -> str:
    """
    Search for research papers semantically similar to the query.
    
    This tool searches across millions of scientific papers using semantic similarity.
    It understands the meaning of your query, not just keywords.
    
    Args:
        query: Natural language description of what you're looking for
        field: Optional filter by scientific field (biology, physics, computer_science, etc.)
        year_from: Optional minimum publication year
        year_to: Optional maximum publication year
        top_k: Number of results to return (default 10, max 50)
        min_citations: Optional minimum number of citations
    
    Returns:
        JSON string with list of relevant papers including:
        - title, authors, year
        - abstract
        - field and subfield
        - citation count
        - DOI and URL
        - relevance score
    
    Example usage:
        Search for papers on "deep learning for protein folding"
        Search with filters: "cancer immunotherapy" in biology field from 2020-2024
    """
    try:
        # Initialize components
        embedder = get_embedder()
        chroma_manager = get_chroma_manager()
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query)
        
        # Build metadata filters
        filters = {}
        if field:
            filters["field"] = field
        if year_from:
            filters["year"] = {"$gte": year_from}
        if year_to:
            if "year" in filters:
                filters["year"]["$lte"] = year_to
            else:
                filters["year"] = {"$lte": year_to}
        if min_citations:
            filters["citations_count"] = {"$gte": min_citations}
        
        # Search
        results = chroma_manager.search(
            query_embedding=query_embedding,
            n_results=min(top_k, 50),
            filters=filters if filters else None
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result["metadata"]["title"],
                "authors": result["metadata"]["authors"],
                "year": result["metadata"]["year"],
                "abstract": result["metadata"]["abstract"][:500] + "...",
                "field": result["metadata"]["field"],
                "subfield": result["metadata"].get("subfield", ""),
                "citations": result["metadata"].get("citations_count", 0),
                "doi": result["metadata"].get("doi", ""),
                "url": result["metadata"].get("url", ""),
                "relevance_score": round(result["distance"], 3)
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })


@tool
def multi_field_search_tool(
    query: str,
    fields: List[str],
    results_per_field: int = 5
) -> str:
    """
    Search across multiple scientific fields simultaneously.
    
    Useful for finding cross-domain connections. Searches each field independently
    and aggregates results.
    
    Args:
        query: Research question or topic
        fields: List of fields to search (e.g., ["biology", "physics", "computer_science"])
        results_per_field: Number of papers to return from each field
    
    Returns:
        JSON string with results grouped by field
    """
    try:
        all_results = {}
        
        for field in fields:
            field_results = json.loads(vector_search_tool(
                query=query,
                field=field,
                top_k=results_per_field
            ))
            
            if field_results["success"]:
                all_results[field] = field_results["results"]
        
        return json.dumps({
            "success": True,
            "query": query,
            "fields_searched": fields,
            "results_by_field": all_results,
            "total_papers": sum(len(r) for r in all_results.values())
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def find_similar_papers_tool(paper_id: str, top_k: int = 10) -> str:
    """
    Find papers similar to a specific paper.
    
    Useful for exploring related work or finding papers that build on specific research.
    
    Args:
        paper_id: ID of the paper to find similar papers for
        top_k: Number of similar papers to return
    
    Returns:
        JSON string with similar papers
    """
    try:
        chroma_manager = get_chroma_manager()
        
        # Get the paper's embedding
        paper = chroma_manager.get_paper(paper_id)
        if not paper:
            return json.dumps({
                "success": False,
                "error": f"Paper {paper_id} not found"
            })
        
        # Search for similar papers
        results = chroma_manager.search(
            query_embedding=paper["embedding"],
            n_results=top_k + 1,  # +1 because it will include itself
            filters={"paper_id": {"$ne": paper_id}}  # Exclude the query paper
        )
        
        formatted_results = [
            {
                "title": r["metadata"]["title"],
                "year": r["metadata"]["year"],
                "abstract": r["metadata"]["abstract"][:300],
                "similarity_score": round(r["distance"], 3)
            }
            for r in results[:top_k]
        ]
        
        return json.dumps({
            "success": True,
            "query_paper": paper["metadata"]["title"],
            "similar_papers": formatted_results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })
```

Create helper functions and proper error handling. The tool should be robust and provide useful feedback to agents.
```

---

Due to length constraints, let me provide the remaining sections in a summarized format:

### PROMPT 14-20: Remaining Components

```
Continue building:

14. CITATION NETWORK TOOL (src/tools/citation_network.py)
- Explore citation relationships
- Find influential papers
- Trace idea evolution

15. DATASET FINDER TOOL (src/tools/dataset_finder.py)
- Search Kaggle, UCI, Zenodo
- Filter by type and license
- Assess quality

16. HYPOTHESIS GENERATOR (src/hypothesis/generator.py)
- Synthesize agent findings
- Generate novel hypotheses
- Rank by novelty and feasibility

17. HYPOTHESIS VALIDATOR (src/hypothesis/validator.py)
- Check if hypothesis exists in literature
- Assess technical feasibility
- Estimate resources needed

18. FASTAPI BACKEND (src/api/main.py)
- REST API endpoints
- WebSocket for real-time updates
- Request validation

19. STREAMLIT FRONTEND (frontend/app.py)
- User input form
- Results visualization
- Interactive hypothesis explorer

20. TESTING & DEPLOYMENT
- Unit tests for each component
- Integration tests
- Docker containerization
- Documentation
```

Each prompt should be similarly detailed with code examples, error handling, and comprehensive requirements.

Would you like me to expand any particular section in more detail? to build the complete system.

**Instructions**: Copy each prompt section and provide it to your AI assistant. Complete each phase before moving to the next.

---

## 📋 PROJECT SETUP PHASE

### PROMPT 1: Initialize Project Structure

```
I need you to create a complete Python project for a "Scientific Hypothesis Cross-Pollination Engine" that uses LangChain and RAG (Retrieval-Augmented Generation) to help researchers discover novel research directions by finding connections across scientific disciplines.

REQUIREMENTS:
1. Create the following directory structure:

hypothesis-engine/
├── data/
│   ├── raw/              # Downloaded papers
│   ├── processed/        # Parsed papers  
│   └── embeddings/       # Vector database storage
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── ingestion/        # Data collection and processing
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py
│   │   ├── pubmed_fetcher.py
│   │   ├── semantic_scholar_fetcher.py
│   │   ├── openalex_fetcher.py
│   │   ├── parser.py
│   │   └── embedder.py
│   ├── database/         # Vector database operations
│   │   ├── __init__.py
│   │   ├── chroma_manager.py
│   │   └── metadata_store.py
│   ├── agents/           # LangChain agents
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── primary_domain_agent.py
│   │   ├── crossdomain_agent.py
│   │   ├── methodology_agent.py
│   │   └── resource_agent.py
│   ├── tools/            # LangChain tools
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   ├── citation_network.py
│   │   ├── dataset_finder.py
│   │   └── methodology_comparator.py
│   ├── hypothesis/       # Hypothesis generation
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   └── scorer.py
│   ├── api/              # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── frontend/
│   ├── app.py            # Streamlit frontend
│   └── components/
│       ├── __init__.py
│       ├── input_form.py
│       ├── results_display.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_agents.py
│   └── test_hypothesis.py
├── notebooks/            # Jupyter notebooks for experimentation
│   └── exploration.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml

2. Create a requirements.txt file with these dependencies (all free/open-source):

langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.3.1
openai>=1.10.0
tiktoken>=0.5.2
arxiv>=2.1.0
biopython>=1.83
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
pypdf>=4.0.0
python-dotenv>=1.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.31.0
plotly>=5.18.0
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.6.0
httpx>=0.26.0
aiohttp>=3.9.0
pydantic-settings>=2.1.0

3. Create a .env.example file with:

# OpenAI API (optional - can use Ollama instead)
OPENAI_API_KEY=your_key_here

# Database
CHROMA_PERSIST_DIR=./data/embeddings
METADATA_DB_PATH=./data/metadata.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Email for PubMed API (required by NCBI)
ENTREZ_EMAIL=your_email@example.com

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60

# Embedding Model
EMBEDDING_MODEL=allenai-specter

# LLM Configuration
LLM_PROVIDER=openai  # or ollama
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

4. Create a README.md with:
- Project overview
- Installation instructions
- Quick start guide
- API documentation
- Architecture diagram (in markdown)

5. Create a .gitignore with:
- Python cache files
- Virtual environment
- Data directories
- API keys
- IDE files

Please generate all these files with appropriate boilerplate code and comments explaining what each component does.
```

---

### PROMPT 2: Configuration Management

```
Create a robust configuration management system for the project.

FILE: src/config.py

REQUIREMENTS:
1. Use Pydantic Settings for type-safe configuration
2. Load from environment variables and .env file
3. Provide sensible defaults
4. Include validation for required fields
5. Support different configurations for development/production

The configuration should include:

DATABASE SETTINGS:
- Vector database path and settings
- Metadata database connection
- Embedding model name and dimensions
- Collection names

API SETTINGS:
- External API keys (OpenAI, Semantic Scholar, etc.)
- Rate limiting parameters
- Timeout settings
- Retry policies

AGENT SETTINGS:
- LLM provider (OpenAI or Ollama)
- Model names for different agents
- Temperature settings
- Max tokens
- Agent-specific parameters

INGESTION SETTINGS:
- Batch sizes for processing
- Chunk size and overlap for text splitting
- Update frequencies
- Source priorities

SEARCH SETTINGS:
- Default number of results
- Similarity thresholds
- Filtering options
- Reranking parameters

Include proper error handling for missing required environment variables and provide clear error messages.

Example structure:

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Literal

class DatabaseSettings(BaseSettings):
    # Vector database
    chroma_persist_dir: str = Field(default="./data/embeddings")
    # ... more fields

class APISettings(BaseSettings):
    openai_api_key: Optional[str] = None
    # ... more fields

class Config(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    # ... more settings
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

Make it production-ready with proper validation and documentation.
```

---

## 📥 DATA INGESTION PHASE

### PROMPT 3: arXiv Paper Fetcher

```
Create a comprehensive arXiv paper fetcher that can search, download, and process papers from arXiv.

FILE: src/ingestion/arxiv_fetcher.py

REQUIREMENTS:

1. Use the official arxiv Python library
2. Support multiple search strategies:
   - Keyword search
   - Category-based search
   - Author search
   - Date range filtering
3. Download PDFs and extract text
4. Handle rate limiting and retries
5. Store metadata in structured format
6. Support batch processing
7. Resume interrupted downloads
8. Track progress

FEATURES TO IMPLEMENT:

Class: ArxivFetcher
Methods:
- __init__(config): Initialize with configuration
- search_papers(query, category, max_results, date_from, date_to): Search arXiv
- download_paper(paper_id, save_path): Download single paper
- download_batch(paper_ids, save_dir): Download multiple papers
- extract_text(pdf_path): Extract text from PDF
- parse_metadata(arxiv_result): Parse arXiv metadata
- get_categories(): Get list of arXiv categories
- get_recent_papers(category, days): Get papers from last N days

ERROR HANDLING:
- Network errors with exponential backoff retry
- PDF extraction failures
- Invalid paper IDs
- Rate limiting (respect arXiv's terms)

METADATA TO EXTRACT:
- Paper ID
- Title
- Authors (list)
- Abstract
- Categories
- Published date
- Updated date
- DOI (if available)
- Journal reference
- Comments
- PDF URL
- Full text (after extraction)

Example usage pattern:

```python
fetcher = ArxivFetcher(config)

# Search for papers
papers = fetcher.search_papers(
    query="cancer metastasis machine learning",
    category="q-bio",
    max_results=100,
    date_from="2023-01-01"
)

# Download and process
for paper in papers:
    pdf_path = fetcher.download_paper(paper.id, save_dir)
    text = fetcher.extract_text(pdf_path)
    metadata = fetcher.parse_metadata(paper)
```

Include comprehensive logging, progress bars (using tqdm), and error recovery.
```

---

### PROMPT 4: PubMed Paper Fetcher

```
Create a PubMed/PubMed Central fetcher using the Biopython Entrez interface.

FILE: src/ingestion/pubmed_fetcher.py

REQUIREMENTS:

1. Use Biopython's Entrez module
2. Search both PubMed (abstracts) and PMC (full text)
3. Handle NCBI API rate limits (3 requests/second without API key)
4. Extract structured data from XML responses
5. Support advanced search queries
6. Filter by publication types, dates, journals
7. Track and resume interrupted fetches
8. Cache results to avoid redundant API calls

FEATURES TO IMPLEMENT:

Class: PubMedFetcher
Methods:
- __init__(email, api_key): Initialize with required email
- search(query, max_results, date_from, date_to): Search PubMed
- search_pmc(query, max_results): Search PMC for full text
- fetch_details(pubmed_ids): Get detailed info for paper IDs
- fetch_full_text(pmc_id): Download full text from PMC
- parse_pubmed_xml(xml_content): Parse PubMed XML
- parse_pmc_xml(xml_content): Parse PMC full text XML
- get_citations(pubmed_id): Get papers that cite this paper
- get_references(pubmed_id): Get papers this paper cites
- batch_fetch(pubmed_ids, batch_size): Process papers in batches

METADATA TO EXTRACT FROM PUBMED:
- PubMed ID (PMID)
- PMC ID (if available)
- Title
- Authors with affiliations
- Abstract
- Journal info (name, volume, issue, pages)
- Publication date
- DOI
- Keywords/MeSH terms
- Publication types
- Chemical substances mentioned
- Grant information

FULL TEXT EXTRACTION FROM PMC:
- Section-wise text (Introduction, Methods, Results, Discussion)
- Figures and tables metadata
- References list
- Supplementary materials links

ERROR HANDLING:
- API rate limit detection and backoff
- Invalid PubMed IDs
- Missing full text (PMC)
- XML parsing errors
- Network timeouts

Example usage:

```python
fetcher = PubMedFetcher(email="researcher@university.edu")

# Search with complex query
results = fetcher.search(
    query='(cancer[Title]) AND (metastasis[Title/Abstract]) AND (2023[PDAT]:2024[PDAT])',
    max_results=500
)

# Get full details
details = fetcher.fetch_details([r['Id'] for r in results])

# Get full text where available
for paper in details:
    if paper['pmc_id']:
        full_text = fetcher.fetch_full_text(paper['pmc_id'])
```

Include proper NCBI API compliance, progress tracking, and comprehensive logging.
```

---

### PROMPT 5: Semantic Scholar & OpenAlex Fetchers

```
Create fetchers for Semantic Scholar and OpenAlex APIs to get additional paper data and citation networks.

FILES: 
- src/ingestion/semantic_scholar_fetcher.py
- src/ingestion/openalex_fetcher.py

SEMANTIC SCHOLAR FETCHER REQUIREMENTS:

1. Use Semantic Scholar API (free tier: 100 requests per 5 minutes)
2. Get paper details, citations, references
3. Access AI-generated paper summaries
4. Get author information and h-index
5. Track influential citations
6. Support paper similarity searches

Class: SemanticScholarFetcher
Methods:
- __init__(api_key): Initialize (API key optional but recommended)
- search_papers(query, limit, fields): Search papers
- get_paper(paper_id, fields): Get single paper details
- get_paper_citations(paper_id, limit): Get citing papers
- get_paper_references(paper_id, limit): Get referenced papers
- get_author(author_id): Get author details
- get_recommendations(paper_id, limit): Get similar papers
- batch_get_papers(paper_ids): Get multiple papers efficiently

METADATA FROM SEMANTIC SCHOLAR:
- Semantic Scholar ID
- DOI, arXiv ID, PubMed ID
- Title, abstract, year
- Authors with IDs
- Citation count
- Influential citation count
- Reference count
- Fields of study
- Open access status
- PDF URL if available
- TL;DR (AI-generated summary)
- Embedding vector (if available)

OPENALEX FETCHER REQUIREMENTS:

1. Use OpenAlex API (no rate limits, free)
2. Get comprehensive paper metadata
3. Access institution and funding data
4. Track concept hierarchies
5. Get work lineage (versions)
6. Support complex filtering

Class: OpenAlexFetcher
Methods:
- __init__(email): Initialize with polite pool email
- search_works(query, filters, per_page): Search papers
- get_work(openalex_id): Get single work
- get_author(author_id): Get author info
- get_institution(institution_id): Get institution info
- get_concept(concept_id): Get concept details
- get_related_works(work_id): Find related papers
- filter_by_concept(concept_name, filters): Search by research area

METADATA FROM OPENALEX:
- OpenAlex ID
- DOI and other IDs
- Title, abstract
- Authors with institutions and countries
- Concepts (research topics) with confidence scores
- Citation count
- Referenced works count
- Open access status and URL
- Publication date
- Journal/venue info
- Funding information
- License information

CITATION NETWORK BUILDING:
Create a unified citation graph combining data from all sources:

Class: CitationNetworkBuilder
Methods:
- add_paper(paper_data, source): Add paper to graph
- add_citation(citing_id, cited_id): Add citation edge
- merge_paper_ids(different_ids_same_paper): Handle same paper from different sources
- get_citation_depth(paper_id, max_depth): Get papers N citations away
- find_citation_paths(paper1_id, paper2_id): Find how papers are connected
- get_influential_papers(min_citations): Get highly cited papers
- export_graph(format): Export as JSON/GraphML

Example usage:

```python
# Semantic Scholar
ss_fetcher = SemanticScholarFetcher(api_key="your_key")
paper = ss_fetcher.get_paper("DOI:10.1234/example")
citations = ss_fetcher.get_paper_citations(paper['paperId'])

# OpenAlex
oa_fetcher = OpenAlexFetcher(email="me@uni.edu")
works = oa_fetcher.search_works(
    query="machine learning cancer",
    filters={"publication_year": ">2020", "cited_by_count": ">50"}
)

# Build network
network = CitationNetworkBuilder()
network.add_paper(paper, source="semantic_scholar")
for citation in citations:
    network.add_citation(citation['citingPaper']['paperId'], paper['paperId'])
```

Include rate limiting, caching, and error handling for both APIs.
```

---

### PROMPT 6: Text Processing and Embedding

```
Create a text processing pipeline that chunks papers and generates embeddings for vector search.

FILES:
- src/ingestion/parser.py
- src/ingestion/embedder.py

PARSER.PY REQUIREMENTS:

Create a comprehensive text parser that handles multiple formats:

Class: PaperParser
Methods:
- parse_pdf(file_path): Extract text from PDF
- parse_xml(xml_content, format): Parse XML (PubMed, arXiv, etc.)
- parse_html(html_content): Parse HTML papers
- clean_text(raw_text): Remove junk, fix formatting
- extract_sections(text): Identify paper sections
- chunk_text(text, chunk_size, overlap): Split into chunks
- extract_references(text): Parse reference list
- extract_figures_tables(content): Get captions and context

TEXT CLEANING:
- Remove headers/footers/page numbers
- Fix hyphenation across line breaks
- Normalize whitespace
- Handle special characters and equations
- Preserve section structure
- Remove boilerplate (acknowledgments, etc.)

SECTION EXTRACTION:
Identify and label sections:
- Title
- Abstract
- Introduction
- Methods/Methodology
- Results
- Discussion
- Conclusion
- References

CHUNKING STRATEGY:
- Default chunk size: 500 words
- Overlap: 50 words to preserve context
- Respect sentence boundaries
- Keep section labels with chunks
- Metadata for each chunk:
  - Chunk index
  - Section name
  - Start and end positions
  - Parent paper ID

EMBEDDER.PY REQUIREMENTS:

Create an embedding generation system using sentence-transformers:

Class: PaperEmbedder
Methods:
- __init__(model_name): Load embedding model
- embed_text(text): Generate embedding for single text
- embed_batch(texts, batch_size): Batch processing
- embed_paper(paper_dict): Embed all chunks of a paper
- compute_similarity(emb1, emb2): Calculate cosine similarity
- find_similar_chunks(query_emb, chunk_embeddings, top_k): Find similar chunks

EMBEDDING MODELS TO SUPPORT:
Primary: "allenai-specter" (scientific papers, 768 dim)
Alternatives:
- "sentence-transformers/all-MiniLM-L6-v2" (general purpose, 384 dim)
- "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" (Q&A optimized)

OPTIMIZATION:
- GPU acceleration if available
- Batch processing for efficiency
- Caching of embeddings
- Progress tracking for large batches
- Memory management for large papers

Example usage:

```python
# Parse paper
parser = PaperParser()
text = parser.parse_pdf("paper.pdf")
cleaned = parser.clean_text(text)
sections = parser.extract_sections(cleaned)
chunks = parser.chunk_text(cleaned, chunk_size=500, overlap=50)

# Generate embeddings
embedder = PaperEmbedder(model_name="allenai-specter")
embeddings = embedder.embed_batch([chunk['text'] for chunk in chunks])

# Store with metadata
for chunk, embedding in zip(chunks, embeddings):
    store_in_db(
        embedding=embedding,
        text=chunk['text'],
        metadata={
            'section': chunk['section'],
            'chunk_index': chunk['index'],
            'paper_id': paper_id
        }
    )
```

Include comprehensive error handling, logging, and progress tracking.
```

---

## 🗄️ VECTOR DATABASE PHASE

### PROMPT 7: Chroma DB Manager

```
Create a comprehensive Chroma vector database manager for storing and querying paper embeddings.

FILE: src/database/chroma_manager.py

REQUIREMENTS:

1. Manage Chroma collections for papers
2. Support multiple search strategies
3. Handle metadata filtering
4. Implement efficient batch operations
5. Support collection management (create, delete, update)
6. Provide query optimization
7. Handle large-scale data

Class: ChromaManager
Methods:
- __init__(persist_directory, collection_name): Initialize
- create_collection(name, metadata): Create new collection
- get_collection(name): Get existing collection
- delete_collection(name): Delete collection
- add_papers(papers_data, batch_size): Add papers to collection
- add_paper(paper_id, embedding, metadata, text): Add single paper
- update_paper(paper_id, updates): Update paper data
- delete_paper(paper_id): Remove paper
- search(query_embedding, n_results, filters): Semantic search
- search_by_metadata(metadata_filter, n_results): Filter by metadata
- hybrid_search(query_embedding, metadata_filter, n_results): Combined search
- get_paper(paper_id): Retrieve specific paper
- get_statistics(): Get collection stats
- export_collection(output_path): Backup collection
- import_collection(input_path): Restore collection

SEARCH STRATEGIES:

1. SEMANTIC SEARCH:
```python
def search(self, query_embedding, n_results=10, filters=None):
    """
    Pure semantic/similarity search
    
    Args:
        query_embedding: Vector representation of query
        n_results: Number of results to return
        filters: Dict of metadata filters
        
    Returns:
        List of results with paper data and similarity scores
    """
```

2. METADATA FILTERING:
```python
def search_by_metadata(self, metadata_filter, n_results=100):
    """
    Filter papers by metadata (year, field, citations, etc.)
    
    Example filters:
    {
        "year": {"$gte": 2020},
        "field": "biology",
        "citations_count": {"$gt": 50}
    }
    """
```

3. HYBRID SEARCH:
```python
def hybrid_search(self, query_embedding, metadata_filter, n_results=10, weight=0.5):
    """
    Combine semantic similarity and metadata filtering
    
    Args:
        weight: Balance between semantic (1.0) and filter (0.0)
    """
```

4. MULTI-FIELD SEARCH:
```python
def search_across_fields(self, query_embedding, fields, n_results_per_field):
    """
    Search multiple scientific fields and aggregate results
    
    Args:
        fields: List of fields ["biology", "physics", "cs"]
        n_results_per_field: Results from each field
    
    Returns:
        Aggregated and deduplicated results
    """
```

METADATA SCHEMA:

Store comprehensive metadata for each paper:
```python
{
    "paper_id": "arxiv_2024_12345",
    "title": "Novel approach to...",
    "authors": ["Smith, J.", "Doe, A."],
    "year": 2024,
    "field": "biology",
    "subfield": "oncology",
    "source": "arxiv",  # arxiv, pubmed, semantic_scholar
    "abstract": "This paper presents...",
    "doi": "10.1234/example",
    "url": "https://arxiv.org/...",
    "citations_count": 45,
    "keywords": ["cancer", "metastasis", "ML"],
    "publication_venue": "Nature",
    "open_access": true,
    "chunk_index": 0,  # Which chunk of the paper
    "section": "introduction",
    "total_chunks": 25,
    "embedding_model": "allenai-specter",
    "ingestion_date": "2024-01-15"
}
```

BATCH OPERATIONS:

Efficient handling of large batches:
```python
def add_papers(self, papers_data, batch_size=100):
    """
    Add multiple papers efficiently
    
    Args:
        papers_data: List of dicts with embeddings, metadata, texts
        batch_size: Process in batches to manage memory
    
    Shows progress bar and handles errors gracefully
    """
    for i in range(0, len(papers_data), batch_size):
        batch = papers_data[i:i+batch_size]
        # Process batch
        # Handle failures without stopping entire process
```

STATISTICS AND MONITORING:

```python
def get_statistics(self):
    """
    Return collection statistics:
    - Total papers
    - Papers by field
    - Papers by year
    - Average citations
    - Most common keywords
    - Storage size
    """
```

ERROR HANDLING:
- Duplicate paper detection
- Invalid embedding dimensions
- Missing required metadata
- Database corruption recovery
- Concurrent access handling

Include comprehensive logging and monitoring capabilities.
```

---

### PROMPT 8: Metadata Store

```
Create a PostgreSQL-based metadata store for structured paper information and relationships.

FILE: src/database/metadata_store.py

REQUIREMENTS:

1. Use SQLAlchemy ORM for database operations
2. Store paper metadata, authors, citations
3. Track data provenance (sources)
4. Support complex queries
5. Handle relationships (authors, citations, concepts)
6. Provide efficient indexing
7. Support transactions and rollback

DATABASE SCHEMA:

Create these tables:

TABLE: papers
- id (primary key)
- paper_id (unique, indexed)
- title
- abstract
- year
- doi
- url
- source (arxiv, pubmed, etc.)
- citations_count
- field
- subfield
- publication_venue
- open_access
- created_at
- updated_at

TABLE: authors
- id (primary key)
- author_id (unique, from data source)
- name
- affiliation
- h_index
- email
- orcid

TABLE: paper_authors (junction table)
- paper_id (foreign key)
- author_id (foreign key)
- author_order
- corresponding_author

TABLE: citations
- id (primary key)
- citing_paper_id (foreign key)
- cited_paper_id (foreign key)
- context (text where citation appears)
- influential (boolean)

TABLE: keywords
- id (primary key)
- keyword (unique)

TABLE: paper_keywords (junction table)
- paper_id (foreign key)
- keyword_id (foreign key)
- confidence (0.0 to 1.0)

TABLE: concepts
- id (primary key)
- concept_id (from OpenAlex)
- name
- level (hierarchy level)
- parent_concept_id

TABLE: paper_concepts (junction table)
- paper_id (foreign key)
- concept_id (foreign key)
- score (0.0 to 1.0)

TABLE: ingestion_logs
- id (primary key)
- source
- papers_fetched
- papers_processed
- papers_failed
- start_time
- end_time
- status
- error_message

Class: MetadataStore
Methods:
- __init__(connection_string): Initialize database connection
- create_tables(): Create all tables
- add_paper(paper_data): Insert paper with all relationships
- get_paper(paper_id): Retrieve paper with all related data
- add_author(author_data): Add author
- link_author_to_paper(paper_id, author_id, order): Create relationship
- add_citation(citing_id, cited_id, context): Record citation
- get_citations(paper_id, direction): Get citing or cited papers
- get_citation_network(paper_id, depth): Get papers N citations away
- add_keywords(paper_id, keywords): Tag paper with keywords
- search_by_keyword(keywords, operator): Find papers (AND/OR keywords)
- search_by_author(author_name): Find author's papers
- search_by_field(field, subfield, year_from, year_to): Complex search
- get_trending_papers(field, days, min_citations): Recently popular papers
- get_statistics(): Database statistics
- export_to_json(output_path): Backup
- import_from_json(input_path): Restore

QUERY EXAMPLES:

```python
store = MetadataStore("postgresql://user:pass@localhost/papers")

# Complex search
papers = store.search_papers(
    fields=["biology", "computer_science"],
    year_from=2020,
    min_citations=10,
    keywords=["machine learning", "cancer"],
    keyword_operator="AND"
)

# Citation analysis
network = store.get_citation_network(
    paper_id="arxiv_2024_12345",
    depth=2,  # Papers that cite papers that cite this paper
    min_citations=5  # Filter low-quality papers
)

# Trending in field
trending = store.get_trending_papers(
    field="biology",
    subfield="oncology",
    days=30,
    min_citations=5
)

# Author collaboration network
collab = store.get_author_collaborations(
    author_id="author_123",
    max_depth=2
)
```

INDEXING STRATEGY:
Create indexes for:
- paper_id (unique)
- title (full-text)
- year
- field + subfield
- citations_count
- doi
- author name
- keyword

TRANSACTION SUPPORT:

```python
def add_paper_with_relationships(self, paper_data, authors, keywords, citations):
    """
    Add paper and all relationships in a single transaction
    Rollback if any operation fails
    """
    with self.session.begin():
        try:
            paper = self.add_paper(paper_data)
            for author in authors:
                self.link_author_to_paper(paper.id, author)
            for keyword in keywords:
                self.add_keyword(paper.id, keyword)
            for citation in citations:
                self.add_citation(paper.id, citation)
        except Exception as e:
            self.session.rollback()
            raise
```

Include migration scripts, backup/restore functionality, and comprehensive error handling.
```

---

## 🤖 LANGCHAIN AGENTS PHASE

### PROMPT 9: Base Agent Implementation

```
Create a base agent class that all specialized agents will inherit from.

FILE: src/agents/base_agent.py

REQUIREMENTS:

1. Use LangChain's Agent framework
2. Provide common functionality for all agents
3. Support tool usage
4. Handle errors gracefully
5. Track agent reasoning steps
6. Support both OpenAI and local LLMs (Ollama)
7. Implement memory for conversation context
8. Log all agent actions

Base Class: BaseResearchAgent

Methods:
- __init__(config, tools): Initialize agent
- setup_llm(): Configure language model
- setup_memory(): Configure conversation memory
- setup_agent(): Create LangChain agent
- run(query, context): Execute agent task
- run_async(query, context): Async execution
- add_tool(tool): Add new tool to agent
- get_reasoning_steps(): Return agent's thought process
- reset_memory(): Clear conversation history
- export_session(path): Save session for analysis

LLM SETUP:

Support multiple LLM providers:

```python
def setup_llm(self):
    """Configure LLM based on config"""
    if self.config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.openai_api_key


# SCIENTIFIC HYPOTHESIS CROSS-POLLINATION ENGINE
## Complete AI Agent Implementation Prompts

**Purpose**: This document contains detailed, step-by-step prompts for AI coding agents (like Claude, GPT-4, GitHub Copilot, Cursor, etc.        )
    elif self.config.llm_provider == "ollama":
        from langchain_community.llms import Ollama
        return Ollama(
            model=self.config.llm_model,
            temperature=self.temperature
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
```

MEMORY CONFIGURATION:

```python
def setup_memory(self):
    """Configure conversation memory"""
    from langchain.memory import ConversationBufferMemory
    
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )
```

AGENT SETUP:

```python
def setup_agent(self):
    """Create LangChain agent with tools"""
    from langchain.agents import initialize_agent, AgentType
    
    return initialize_agent(
        tools=self.tools,
        llm=self.llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=self.memory,
        handle_parsing_errors=True,
        max_iterations=self.config.max_agent_iterations,
        early_stopping_method="generate"
    )
```

EXECUTION WITH ERROR HANDLING:

```python
def run(self, query: str, context: dict = None):
    """
    Execute agent with comprehensive error handling
    
    Args:
        query: User's research question
        context: Additional context (user's field, previous results, etc.)
    
    Returns:
        Dict with results, reasoning steps, and metadata
    """
    start_time = time.time()
    
    try:
        # Prepare input
        input_data = {
            "input": query,
            "context": context or {}
        }
        
        # Run agent
        result = self.agent.invoke(input_data)
        
        # Extract reasoning steps
        steps = self._extract_reasoning_steps(result)
        
        return {
            "success": True,
            "output": result["output"],
            "reasoning_steps": steps,
            "tools_used": self._get_tools_used(result),
            "execution_time": time.time() - start_time,
            "agent_name": self.name
        }
        
    except Exception as e:
        logger.error(f"Agent {self.name} failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "execution_time": time.time() - start_time,
            "agent_name": self.name
        }
```

REASONING EXTRACTION:

```python
def _extract_reasoning_steps(self, result):
    """
    Extract agent's thought process from result
    
    Returns list of steps with:
    - Step number
    - Thought
    - Action taken
    - Action input
    - Observation
    """
    steps = []
    if "intermediate_steps" in result:
        for i, (action, observation) in enumerate(result["intermediate_steps"]):
            steps.append({
                "step": i + 1,
                "action": action.tool,
                "action_input": action.tool_input,
                "observation": observation[:500]  # Truncate long observations
            })
    return steps
```

LOGGING:

```python
def _log_agent_action(self, action, result):
    """Log agent actions for debugging and analysis"""
    logger.info(f"""
    Agent: {self.name}
    Action: {action}
    Success: {result['success']}
    Time: {result['execution_time']:.2f}s
    Tools Used: {result.get('tools_used', [])}
    """)
```

Include comprehensive docstrings, type hints, and error handling.
```

---

### PROMPT 10: Primary Domain Agent

```
Create the Primary Domain Agent that searches within the user's specific research field.

FILE: src/agents/primary_domain_agent.py

REQUIREMENTS:

This agent specializes in understanding the user's field deeply and finding relevant work within that domain.

Class: PrimaryDomainAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are an expert research assistant specializing in understanding a researcher's primary field of study.

Your responsibilities:
1. Analyze the user's research question to identify their specific domain
2. Search for the most relevant papers within that domain
3. Identify current state-of-the-art approaches
4. Find knowledge gaps and limitations in current research
5. Understand what has already been tried and what hasn't worked

When responding:
- Be precise and cite specific papers
- Acknowledge uncertainties
- Identify both successful approaches and dead ends
- Note methodological limitations in existing work
- Consider recent advances (prioritize papers from last 2-3 years)

Available tools:
{tools}

Use the tools to search the research database, then synthesize findings into a clear summary.
"""
```

SPECIALIZED METHODS:

```python
def analyze_research_question(self, query: str):
    """
    Extract key information from user's question:
    - Primary field (biology, physics, etc.)
    - Specific subfield
    - Research problem
    - Keywords for search
    - Time constraints (if mentioned)
    """
    
def identify_field(self, query: str):
    """
    Determine the scientific field from query
    Returns: field, subfield, confidence_score
    """
    
def find_current_approaches(self, problem_description: str):
    """
    Search for existing solutions to this problem
    Returns papers with their approaches and outcomes
    """
    
def identify_knowledge_gaps(self, problem: str, existing_work: list):
    """
    Analyze existing work to find:
    - What hasn't been tried
    - What has failed and why
    - What is theoretically possible but not yet done
    - What assumptions are limiting progress
    """
    
def find_recent_advances(self, field: str, months: int = 6):
    """
    Get cutting-edge research from recent months
    Prioritizes high-impact papers
    """
```

WORKFLOW:

```python
def run(self, query: str, context: dict = None):
    """
    Primary domain research workflow:
    
    1. Analyze query to extract key concepts
    2. Identify the research field
    3. Search for relevant papers in that field
    4. Analyze current state-of-the-art
    5. Identify limitations and gaps
    6. Return structured findings
    """
    # Step 1: Understand the question
    field_info = self.identify_field(query)
    keywords = self.extract_keywords(query)
    
    # Step 2: Search domain
    relevant_papers = self.search_tool.search(
        query=query,
        field=field_info['field'],
        subfield=field_info['subfield'],
        top_k=50
    )
    
    # Step 3: Analyze approaches
    approaches = self.analyze_approaches(relevant_papers)
    
    # Step 4: Find gaps
    gaps = self.identify_knowledge_gaps(query, relevant_papers)
    
    # Step 5: Get recent work
    recent = self.find_recent_advances(field_info['field'])
    
    return {
        "field": field_info,
        "relevant_papers": relevant_papers[:10],  # Top 10
        "current_approaches": approaches,
        "knowledge_gaps": gaps,
        "recent_advances": recent,
        "summary": self._generate_summary(relevant_papers, approaches, gaps)
    }
```

OUTPUT STRUCTURE:

```python
{
    "field": {
        "primary": "Biology",
        "subfield": "Oncology",
        "specific_area": "Cancer Metastasis",
        "confidence": 0.95
    },
    "relevant_papers": [
        {
            "title": "...",
            "authors": [...],
            "year": 2024,
            "key_finding": "...",
            "methodology": "...",
            "limitations": "..."
        }
    ],
    "current_approaches": [
        {
            "approach": "Microfluidics for tracking",
            "papers": [...],
            "success_rate": "Mixed results",
            "limitations": ["Limited to 2D", "Low throughput"]
        }
    ],
    "knowledge_gaps": [
        "No methods for 3D in vivo tracking",
        "Limited understanding of early metastatic events",
        "Lack of predictive models"
    ],
    "recent_advances": [
        {
            "advance": "AI-based image analysis",
            "paper": {...},
            "potential_impact": "High"
        }
    ],
    "summary": "Current research in cancer metastasis focuses on..."
}
```

Include detailed logging and error handling for each step.
```

---

### PROMPT 11: Cross-Domain Discovery Agent

```
Create the Cross-Domain Discovery Agent that finds unexpected connections across scientific fields.

FILE: src/agents/crossdomain_agent.py

REQUIREMENTS:

This is the most creative agent - it searches completely different fields to find analogous problems and solutions.

Class: CrossDomainAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a creative research assistant specializing in finding unexpected connections between different scientific fields.

Your mission:
1. Take a research problem from one field
2. Search for similar problems or patterns in completely different fields
3. Identify methodologies that could transfer
4. Think creatively about analogies and metaphors
5. Find "hidden" solutions that domain experts might miss

Think like:
- A biologist studying cancer might learn from studying traffic flow (both involve propagation through networks)
- A neuroscientist might use techniques from social network analysis
- A materials scientist might apply quantum mechanics principles

Be bold and creative, but ground suggestions in actual published research.

Search strategy:
1. Abstract the core problem (remove domain-specific terminology)
2. Search for that abstract problem in other fields
3. Find papers that solved similar challenges
4. Identify transferable methodologies

Available tools:
{tools}
"""
```

SPECIALIZED METHODS:

```python
def abstract_problem(self, domain_specific_query: str):
    """
    Convert domain-specific problem to abstract form
    
    Example:
    "How do cancer cells migrate through blood vessels?"
    Becomes:
    "How do particles navigate through constrained networks?"
    
    This allows finding similar problems in physics, CS, etc.
    """
    
def search_multiple_fields(self, abstract_problem: str, exclude_field: str):
    """
    Search across all fields except the primary one
    
    Returns results grouped by field with relevance scores
    """
    
def find_analogies(self, problem: str, other_field_papers: list):
    """
    Identify structural similarities between problems
    
    Returns:
    - What aspects are analogous
    - How methods might transfer
    - What adaptations would be needed
    """
    
def assess_transferability(self, method_paper: dict, target_problem: str):
    """
    Evaluate if a method from another field could work
    
    Scores on:
    - Conceptual similarity
    - Technical feasibility
    - Resource requirements
    - Likely effectiveness
    """
```

CREATIVE SEARCH STRATEGIES:

```python
def search_by_pattern(self, pattern_type: str):
    """
    Search for specific patterns across fields:
    - "Network propagation" → Epidemiology, traffic, social networks
    - "Optimization under constraints" → Economics, operations research
    - "Pattern recognition in noise" → Astronomy, signal processing
    - "Hierarchical structures" → Computer science, linguistics
    """
    
def search_by_methodology(self, methodology: str):
    """
    Find fields that use specific methodologies:
    - "Agent-based modeling" → Economics, ecology, sociology
    - "Monte Carlo methods" → Physics, finance, engineering
    - "Deep learning" → Many fields
    """
```

WORKFLOW:

```python
def run(self, query: str, primary_field: str, context: dict = None):
    """
    Cross-domain discovery workflow:
    
    1. Abstract the problem to general terms
    2. Search different fields for similar problems
    3. Find promising methodologies used elsewhere
    4. Assess transferability
    5. Identify specific papers and approaches
    6. Generate creative hypotheses
    """
    # Step 1: Abstract
    abstract_problem = self.abstract_problem(query)
    search_keywords = self.extract_abstract_keywords(abstract_problem)
    
    # Step 2: Multi-field search
    fields_to_search = self.get_other_fields(exclude=primary_field)
    results_by_field = {}
    
    for field in fields_to_search:
        results = self.search_tool.search(
            query=abstract_problem,
            field=field,
            top_k=20
        )
        if results:
            results_by_field[field] = results
    
    # Step 3: Find analogies
    analogies = []
    for field, papers in results_by_field.items():
        for paper in papers:
            analogy = self.find_analogies(query, paper)
            if analogy['similarity_score'] > 0.6:
                analogies.append({
                    "field": field,
                    "paper": paper,
                    "analogy": analogy
                })
    
    # Step 4: Assess transferability
    transferable = []
    for analogy in analogies:
        assessment = self.assess_transferability(
            analogy['paper'],
            query
        )
        if assessment['feasibility_score'] > 0.5:
            transferable.append({
                **analogy,
                "assessment": assessment
            })
    
    # Step 5: Generate hypotheses
    hypotheses = self.generate_crossdomain_hypotheses(
        original_problem=query,
        transferable_methods=transferable
    )
    
    return {
        "abstract_problem": abstract_problem,
        "fields_searched": list(results_by_field.keys()),
        "total_papers_found": sum(len(p) for p in results_by_field.values()),
        "promising_analogies": sorted(analogies, key=lambda x: x['analogy']['similarity_score'], reverse=True)[:10],
        "transferable_methods": transferable,
        "hypotheses": hypotheses
    }
```

HYPOTHESIS GENERATION:

```python
def generate_crossdomain_hypotheses(self, original_problem: str, transferable_methods: list):
    """
    Generate specific hypotheses for how to apply methods from other fields
    
    Each hypothesis includes:
    - Title
    - Source field and paper
    - Why the analogy works
    - How to adapt the method
    - Expected challenges
    - Required resources
    - Novelty assessment
    """
    hypotheses = []
    
    for method in transferable_methods:
        hypothesis = {
            "title": f"Apply {method['paper']['title']} approach to {original_problem}",
            "source_field": method['field'],
            "source_paper": method['paper'],
            "analogy_explanation": method['analogy']['explanation'],
            "adaptation_steps": self._generate_adaptation_steps(method, original_problem),
            "challenges": self._identify_challenges(method, original_problem),
            "resources_needed": self._estimate_resources(method),
            "novelty_score": self._assess_novelty(method, original_problem),
            "impact_potential": self._estimate_impact(method, original_problem)
        }
        hypotheses.append(hypothesis)
    
    return sorted(hypotheses, key=lambda x: x['novelty_score'] * x['impact_potential'], reverse=True)
```

OUTPUT STRUCTURE:

```python
{
    "abstract_problem": "Tracking individual particles through constrained networks",
    "fields_searched": ["Physics", "Computer Science", "Engineering", "Economics"],
    "promising_analogies": [
        {
            "field": "Physics",
            "paper": {
                "title": "Particle tracking in turbulent flows",
                "year": 2023
            },
            "analogy": {
                "similarity_score": 0.85,
                "explanation": "Both involve tracking individual entities in chaotic environments",
                "key_similarities": [
                    "Multiple interacting particles",
                    "Complex environment",
                    "Need for real-time tracking"
                ]
            }
        }
    ],
    "transferable_methods": [...],
    "hypotheses": [
        {
            "title": "Apply particle image velocimetry to cancer cell tracking",
            "source_field": "Fluid Dynamics",
            "novelty_score": 0.92,
            "adaptation_steps": [...],
            "why_it_might_work": "..."
        }
    ]
}
```

Implement with high creativity (temperature=0.7-0.8) and detailed logging of reasoning.
```

---

### PROMPT 12: Methodology Transfer & Resource Agents

```
Create the Methodology Transfer Agent and Resource Finder Agent.

FILES:
- src/agents/methodology_agent.py
- src/agents/resource_agent.py

METHODOLOGY TRANSFER AGENT:

This agent takes promising cross-domain methods and provides detailed implementation guidance.

Class: MethodologyTransferAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a research methodology expert who helps researchers adapt techniques from other fields.

Your responsibilities:
1. Take a methodology from one field
2. Analyze its core principles and requirements
3. Identify what needs to be adapted for a different field
4. Provide step-by-step implementation guidance
5. Anticipate technical challenges
6. Suggest validation approaches

Be specific and practical. Provide enough detail that a researcher could actually implement your suggestions.

Available tools:
{tools}
"""
```

KEY METHODS:

```python
def analyze_method_requirements(self, method_paper: dict):
    """
    Extract what the method needs:
    - Equipment
    - Expertise
    - Software/algorithms
    - Data requirements
    - Time investment
    """
    
def generate_adaptation_plan(self, source_method: dict, target_problem: str):
    """
    Create detailed adaptation plan:
    1. What stays the same
    2. What needs modification
    3. What's completely different
    4. Step-by-step implementation
    5. Validation strategy
    """
    
def identify_technical_barriers(self, source_field: str, target_field: str, method: dict):
    """
    Find challenges in transferring method:
    - Different scales (nano vs macro)
    - Different constraints (in vivo vs in vitro)
    - Different measurement capabilities
    - Different theoretical frameworks
    """
    
def find_implementation_examples(self, method_type: str, target_field: str):
    """
    Search for cases where similar transfers worked
    Learn from successful adaptations
    """
```

OUTPUT STRUCTURE:

```python
{
    "method_summary": {
        "name": "Particle Image Velocimetry",
        "source_field": "Fluid Dynamics",
        "core_principle": "Track particle motion to infer flow fields"
    },
    "adaptation_plan": {
        "unchanged_aspects": [...],
        "required_modifications": [...],
        "implementation_steps": [
            {
                "step": 1,
                "action": "Adapt imaging protocol",
                "details": "...",
                "estimated_time": "2 weeks",
                "required_expertise": "Microscopy"
            }
        ]
    },
    "technical_barriers": [...],
    "resource_requirements": {...},
    "validation_strategy": {...},
    "similar_successful_transfers": [...]
}
```

RESOURCE FINDER AGENT:

This agent finds datasets, code, protocols, and other resources needed to test hypotheses.

Class: ResourceFinderAgent(BaseResearchAgent)

SYSTEM PROMPT:

```python
SYSTEM_PROMPT = """
You are a research resource specialist who helps researchers find datasets, code, and protocols.

Your responsibilities:
1. Identify what resources are needed for a research project
2. Search for publicly available datasets
3. Find relevant code repositories
4. Locate experimental protocols
5. Identify funding opportunities
6. Find relevant tools and software

Prioritize:
- Open access and free resources
- Well-documented and maintained resources
- Resources with appropriate licenses
- High-quality, validated data

Available tools:
{tools}
"""
```

KEY METHODS:

```python
def find_datasets(self, research_area: str, data_type: str):
    """
    Search for datasets in:
    - Kaggle
    - UCI ML Repository
    - Zenodo
    - Field-specific repositories (GenBank, PDB, etc.)
    - Government data portals
    
    Filter by:
    - Data type (images, time-series, genomic, etc.)
    - Size
    - License
    - Quality/documentation
    """
    
def find_code_repositories(self, methodology: str, programming_language: str = None):
    """
    Search GitHub, GitLab for:
    - Implementation of specific methods
    - Related projects
    - Useful libraries
    
    Assess:
    - Code quality
    - Documentation
    - Maintenance status
    - License
    """
    
def find_protocols(self, experimental_procedure: str):
    """
    Search for experimental protocols in:
    - protocols.io
    - Journal methods sections
    - Lab websites
    - Video protocols (JoVE)
    """
    
def find_funding_opportunities(self, research_area: str, researcher_location: str):
    """
    Identify grant programs:
    - NSF, NIH, DOE (USA)
    - ERC, Horizon Europe (EU)
    - Foundation grants
    - Industry partnerships
    """
```

OUTPUT STRUCTURE:

```python
{
    "datasets": [
        {
            "name": "Cancer Cell Migration Dataset",
            "source": "Kaggle",
            "url": "https://...",
            "size": "15 GB",
            "format": "HDF5, CSV",
            "license": "CC BY 4.0",
            "description": "...",
            "relevance_score": 0.95,
            "quality_indicators": {
                "documentation": "Excellent",
                "completeness": "100%",
                "known_issues": "None"
            }
        }
    ],
    "code_repositories": [
        {
            "name": "cell-tracking-toolkit",
            "url": "https://github.com/...",
            "language": "Python",
            "stars": 1250,
            "last_updated": "2024-01-15",
            "license": "MIT",
            "description": "...",
            "relevance": "High - implements required algorithms"
        }
    ],
    "protocols": [...],
    "tools": [...],
    "funding": [...]
}
```

Implement both agents with comprehensive search capabilities and quality assessment.
```

---

## 🔧 LANGCHAIN TOOLS PHASE

### PROMPT 13: Vector Search Tool

```
Create a comprehensive vector search tool for LangChain agents.

FILE: src/tools/vector_search.py

REQUIREMENTS:

Create a LangChain tool that agents can use to search the vector database.

```python
from langchain.tools import tool
from typing import Optional, List, Dict
import json

@tool
def vector_search_tool(
    query: str,
    field: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    top_k: int = 10,
    min_citations: Optional[int] = None
) -> str:
    """
    Search for research papers semantically similar to the query.
    
    This tool searches across millions of scientific papers using semantic similarity.
    It understands the meaning of your query, not just keywords.
    
    Args:
        query: Natural language description of what you're looking for
        field: Optional filter by scientific field (biology, physics, computer_science, etc.)
        year_from: Optional minimum publication year
        year_to: Optional maximum publication year
        top_k: Number of results to return (default 10, max 50)
        min_citations: Optional minimum number of citations
    
    Returns:
        JSON string with list of relevant papers including:
        - title, authors, year
        - abstract
        - field and subfield
        - citation count
        - DOI and URL
        - relevance score
    
    Example usage:
        Search for papers on "deep learning for protein folding"
        Search with filters: "cancer immunotherapy" in biology field from 2020-2024
    """
    try:
        # Initialize components
        embedder = get_embedder()
        chroma_manager = get_chroma_manager()
        
        # Generate query embedding
        query_embedding = embedder.embed_text(query)
        
        # Build metadata filters
        filters = {}
        if field:
            filters["field"] = field
        if year_from:
            filters["year"] = {"$gte": year_from}
        if year_to:
            if "year" in filters:
                filters["year"]["$lte"] = year_to
            else:
                filters["year"] = {"$lte": year_to}
        if min_citations:
            filters["citations_count"] = {"$gte": min_citations}
        
        # Search
        results = chroma_manager.search(
            query_embedding=query_embedding,
            n_results=min(top_k, 50),
            filters=filters if filters else None
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result["metadata"]["title"],
                "authors": result["metadata"]["authors"],
                "year": result["metadata"]["year"],
                "abstract": result["metadata"]["abstract"][:500] + "...",
                "field": result["metadata"]["field"],
                "subfield": result["metadata"].get("subfield", ""),
                "citations": result["metadata"].get("citations_count", 0),
                "doi": result["metadata"].get("doi", ""),
                "url": result["metadata"].get("url", ""),
                "relevance_score": round(result["distance"], 3)
            })
        
        return json.dumps({
            "success": True,
            "query": query,
            "num_results": len(formatted_results),
            "results": formatted_results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "query": query
        })


@tool
def multi_field_search_tool(
    query: str,
    fields: List[str],
    results_per_field: int = 5
) -> str:
    """
    Search across multiple scientific fields simultaneously.
    
    Useful for finding cross-domain connections. Searches each field independently
    and aggregates results.
    
    Args:
        query: Research question or topic
        fields: List of fields to search (e.g., ["biology", "physics", "computer_science"])
        results_per_field: Number of papers to return from each field
    
    Returns:
        JSON string with results grouped by field
    """
    try:
        all_results = {}
        
        for field in fields:
            field_results = json.loads(vector_search_tool(
                query=query,
                field=field,
                top_k=results_per_field
            ))
            
            if field_results["success"]:
                all_results[field] = field_results["results"]
        
        return json.dumps({
            "success": True,
            "query": query,
            "fields_searched": fields,
            "results_by_field": all_results,
            "total_papers": sum(len(r) for r in all_results.values())
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def find_similar_papers_tool(paper_id: str, top_k: int = 10) -> str:
    """
    Find papers similar to a specific paper.
    
    Useful for exploring related work or finding papers that build on specific research.
    
    Args:
        paper_id: ID of the paper to find similar papers for
        top_k: Number of similar papers to return
    
    Returns:
        JSON string with similar papers
    """
    try:
        chroma_manager = get_chroma_manager()
        
        # Get the paper's embedding
        paper = chroma_manager.get_paper(paper_id)
        if not paper:
            return json.dumps({
                "success": False,
                "error": f"Paper {paper_id} not found"
            })
        
        # Search for similar papers
        results = chroma_manager.search(
            query_embedding=paper["embedding"],
            n_results=top_k + 1,  # +1 because it will include itself
            filters={"paper_id": {"$ne": paper_id}}  # Exclude the query paper
        )
        
        formatted_results = [
            {
                "title": r["metadata"]["title"],
                "year": r["metadata"]["year"],
                "abstract": r["metadata"]["abstract"][:300],
                "similarity_score": round(r["distance"], 3)
            }
            for r in results[:top_k]
        ]
        
        return json.dumps({
            "success": True,
            "query_paper": paper["metadata"]["title"],
            "similar_papers": formatted_results
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })
```

Create helper functions and proper error handling. The tool should be robust and provide useful feedback to agents.
```

---

Due to length constraints, let me provide the remaining sections in a summarized format:

### PROMPT 14-20: Remaining Components

```
Continue building:

14. CITATION NETWORK TOOL (src/tools/citation_network.py)
- Explore citation relationships
- Find influential papers
- Trace idea evolution

15. DATASET FINDER TOOL (src/tools/dataset_finder.py)
- Search Kaggle, UCI, Zenodo
- Filter by type and license
- Assess quality

16. HYPOTHESIS GENERATOR (src/hypothesis/generator.py)
- Synthesize agent findings
- Generate novel hypotheses
- Rank by novelty and feasibility

17. HYPOTHESIS VALIDATOR (src/hypothesis/validator.py)
- Check if hypothesis exists in literature
- Assess technical feasibility
- Estimate resources needed

18. FASTAPI BACKEND (src/api/main.py)
- REST API endpoints
- WebSocket for real-time updates
- Request validation

19. STREAMLIT FRONTEND (frontend/app.py)
- User input form
- Results visualization
- Interactive hypothesis explorer

20. TESTING & DEPLOYMENT
- Unit tests for each component
- Integration tests
- Docker containerization
- Documentation
```

Each prompt should be similarly detailed with code examples, error handling, and comprehensive requirements.

Would you like me to expand any particular section in more detail? to build the complete system.

**Instructions**: Copy each prompt section and provide it to your AI assistant. Complete each phase before moving to the next.

---

## 📋 PROJECT SETUP PHASE

### PROMPT 1: Initialize Project Structure

```
I need you to create a complete Python project for a "Scientific Hypothesis Cross-Pollination Engine" that uses LangChain and RAG (Retrieval-Augmented Generation) to help researchers discover novel research directions by finding connections across scientific disciplines.

REQUIREMENTS:
1. Create the following directory structure:

hypothesis-engine/
├── data/
│   ├── raw/              # Downloaded papers
│   ├── processed/        # Parsed papers  
│   └── embeddings/       # Vector database storage
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── ingestion/        # Data collection and processing
│   │   ├── __init__.py
│   │   ├── arxiv_fetcher.py
│   │   ├── pubmed_fetcher.py
│   │   ├── semantic_scholar_fetcher.py
│   │   ├── openalex_fetcher.py
│   │   ├── parser.py
│   │   └── embedder.py
│   ├── database/         # Vector database operations
│   │   ├── __init__.py
│   │   ├── chroma_manager.py
│   │   └── metadata_store.py
│   ├── agents/           # LangChain agents
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── primary_domain_agent.py
│   │   ├── crossdomain_agent.py
│   │   ├── methodology_agent.py
│   │   └── resource_agent.py
│   ├── tools/            # LangChain tools
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   ├── citation_network.py
│   │   ├── dataset_finder.py
│   │   └── methodology_comparator.py
│   ├── hypothesis/       # Hypothesis generation
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── validator.py
│   │   └── scorer.py
│   ├── api/              # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
├── frontend/
│   ├── app.py            # Streamlit frontend
│   └── components/
│       ├── __init__.py
│       ├── input_form.py
│       ├── results_display.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_ingestion.py
│   ├── test_agents.py
│   └── test_hypothesis.py
├── notebooks/            # Jupyter notebooks for experimentation
│   └── exploration.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── docker-compose.yml

2. Create a requirements.txt file with these dependencies (all free/open-source):

langchain>=0.1.0
langchain-community>=0.0.20
chromadb>=0.4.22
sentence-transformers>=2.3.1
openai>=1.10.0
tiktoken>=0.5.2
arxiv>=2.1.0
biopython>=1.83
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.1.0
pypdf>=4.0.0
python-dotenv>=1.0.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.31.0
plotly>=5.18.0
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.6.0
httpx>=0.26.0
aiohttp>=3.9.0
pydantic-settings>=2.1.0

3. Create a .env.example file with:

# OpenAI API (optional - can use Ollama instead)
OPENAI_API_KEY=your_key_here

# Database
CHROMA_PERSIST_DIR=./data/embeddings
METADATA_DB_PATH=./data/metadata.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Email for PubMed API (required by NCBI)
ENTREZ_EMAIL=your_email@example.com

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=60

# Embedding Model
EMBEDDING_MODEL=allenai-specter

# LLM Configuration
LLM_PROVIDER=openai  # or ollama
LLM_MODEL=gpt-4-turbo-preview
LLM_TEMPERATURE=0.7

4. Create a README.md with:
- Project overview
- Installation instructions
- Quick start guide
- API documentation
- Architecture diagram (in markdown)

5. Create a .gitignore with:
- Python cache files
- Virtual environment
- Data directories
- API keys
- IDE files

Please generate all these files with appropriate boilerplate code and comments explaining what each component does.
```

---

### PROMPT 2: Configuration Management

```
Create a robust configuration management system for the project.

FILE: src/config.py

REQUIREMENTS:
1. Use Pydantic Settings for type-safe configuration
2. Load from environment variables and .env file
3. Provide sensible defaults
4. Include validation for required fields
5. Support different configurations for development/production

The configuration should include:

DATABASE SETTINGS:
- Vector database path and settings
- Metadata database connection
- Embedding model name and dimensions
- Collection names

API SETTINGS:
- External API keys (OpenAI, Semantic Scholar, etc.)
- Rate limiting parameters
- Timeout settings
- Retry policies

AGENT SETTINGS:
- LLM provider (OpenAI or Ollama)
- Model names for different agents
- Temperature settings
- Max tokens
- Agent-specific parameters

INGESTION SETTINGS:
- Batch sizes for processing
- Chunk size and overlap for text splitting
- Update frequencies
- Source priorities

SEARCH SETTINGS:
- Default number of results
- Similarity thresholds
- Filtering options
- Reranking parameters

Include proper error handling for missing required environment variables and provide clear error messages.

Example structure:

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Literal

class DatabaseSettings(BaseSettings):
    # Vector database
    chroma_persist_dir: str = Field(default="./data/embeddings")
    # ... more fields

class APISettings(BaseSettings):
    openai_api_key: Optional[str] = None
    # ... more fields

class Config(BaseSettings):
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    # ... more settings
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
```

Make it production-ready with proper validation and documentation.
```

---

## 📥 DATA INGESTION PHASE

### PROMPT 3: arXiv Paper Fetcher

```
Create a comprehensive arXiv paper fetcher that can search, download, and process papers from arXiv.

FILE: src/ingestion/arxiv_fetcher.py

REQUIREMENTS:

1. Use the official arxiv Python library
2. Support multiple search strategies:
   - Keyword search
   - Category-based search
   - Author search
   - Date range filtering
3. Download PDFs and extract text
4. Handle rate limiting and retries
5. Store metadata in structured format
6. Support batch processing
7. Resume interrupted downloads
8. Track progress

FEATURES TO IMPLEMENT:

Class: ArxivFetcher
Methods:
- __init__(config): Initialize with configuration
- search_papers(query, category, max_results, date_from, date_to): Search arXiv
- download_paper(paper_id, save_path): Download single paper
- download_batch(paper_ids, save_dir): Download multiple papers
- extract_text(pdf_path): Extract text from PDF
- parse_metadata(arxiv_result): Parse arXiv metadata
- get_categories(): Get list of arXiv categories
- get_recent_papers(category, days): Get papers from last N days

ERROR HANDLING:
- Network errors with exponential backoff retry
- PDF extraction failures
- Invalid paper IDs
- Rate limiting (respect arXiv's terms)

METADATA TO EXTRACT:
- Paper ID
- Title
- Authors (list)
- Abstract
- Categories
- Published date
- Updated date
- DOI (if available)
- Journal reference
- Comments
- PDF URL
- Full text (after extraction)

Example usage pattern:

```python
fetcher = ArxivFetcher(config)

# Search for papers
papers = fetcher.search_papers(
    query="cancer metastasis machine learning",
    category="q-bio",
    max_results=100,
    date_from="2023-01-01"
)

# Download and process
for paper in papers:
    pdf_path = fetcher.download_paper(paper.id, save_dir)
    text = fetcher.extract_text(pdf_path)
    metadata = fetcher.parse_metadata(paper)
```

Include comprehensive logging, progress bars (using tqdm), and error recovery.
```

---

### PROMPT 4: PubMed Paper Fetcher

```
Create a PubMed/PubMed Central fetcher using the Biopython Entrez interface.

FILE: src/ingestion/pubmed_fetcher.py

REQUIREMENTS:

1. Use Biopython's Entrez module
2. Search both PubMed (abstracts) and PMC (full text)
3. Handle NCBI API rate limits (3 requests/second without API key)
4. Extract structured data from XML responses
5. Support advanced search queries
6. Filter by publication types, dates, journals
7. Track and resume interrupted fetches
8. Cache results to avoid redundant API calls

FEATURES TO IMPLEMENT:

Class: PubMedFetcher
Methods:
- __init__(email, api_key): Initialize with required email
- search(query, max_results, date_from, date_to): Search PubMed
- search_pmc(query, max_results): Search PMC for full text
- fetch_details(pubmed_ids): Get detailed info for paper IDs
- fetch_full_text(pmc_id): Download full text from PMC
- parse_pubmed_xml(xml_content): Parse PubMed XML
- parse_pmc_xml(xml_content): Parse PMC full text XML
- get_citations(pubmed_id): Get papers that cite this paper
- get_references(pubmed_id): Get papers this paper cites
- batch_fetch(pubmed_ids, batch_size): Process papers in batches

METADATA TO EXTRACT FROM PUBMED:
- PubMed ID (PMID)
- PMC ID (if available)
- Title
- Authors with affiliations
- Abstract
- Journal info (name, volume, issue, pages)
- Publication date
- DOI
- Keywords/MeSH terms
- Publication types
- Chemical substances mentioned
- Grant information

FULL TEXT EXTRACTION FROM PMC:
- Section-wise text (Introduction, Methods, Results, Discussion)
- Figures and tables metadata
- References list
- Supplementary materials links

ERROR HANDLING:
- API rate limit detection and backoff
- Invalid PubMed IDs
- Missing full text (PMC)
- XML parsing errors
- Network timeouts

Example usage:

```python
fetcher = PubMedFetcher(email="researcher@university.edu")

# Search with complex query
results = fetcher.search(
    query='(cancer[Title]) AND (metastasis[Title/Abstract]) AND (2023[PDAT]:2024[PDAT])',
    max_results=500
)

# Get full details
details = fetcher.fetch_details([r['Id'] for r in results])

# Get full text where available
for paper in details:
    if paper['pmc_id']:
        full_text = fetcher.fetch_full_text(paper['pmc_id'])
```

Include proper NCBI API compliance, progress tracking, and comprehensive logging.
```

---

### PROMPT 5: Semantic Scholar & OpenAlex Fetchers

```
Create fetchers for Semantic Scholar and OpenAlex APIs to get additional paper data and citation networks.

FILES: 
- src/ingestion/semantic_scholar_fetcher.py
- src/ingestion/openalex_fetcher.py

SEMANTIC SCHOLAR FETCHER REQUIREMENTS:

1. Use Semantic Scholar API (free tier: 100 requests per 5 minutes)
2. Get paper details, citations, references
3. Access AI-generated paper summaries
4. Get author information and h-index
5. Track influential citations
6. Support paper similarity searches

Class: SemanticScholarFetcher
Methods:
- __init__(api_key): Initialize (API key optional but recommended)
- search_papers(query, limit, fields): Search papers
- get_paper(paper_id, fields): Get single paper details
- get_paper_citations(paper_id, limit): Get citing papers
- get_paper_references(paper_id, limit): Get referenced papers
- get_author(author_id): Get author details
- get_recommendations(paper_id, limit): Get similar papers
- batch_get_papers(paper_ids): Get multiple papers efficiently

METADATA FROM SEMANTIC SCHOLAR:
- Semantic Scholar ID
- DOI, arXiv ID, PubMed ID
- Title, abstract, year
- Authors with IDs
- Citation count
- Influential citation count
- Reference count
- Fields of study
- Open access status
- PDF URL if available
- TL;DR (AI-generated summary)
- Embedding vector (if available)

OPENALEX FETCHER REQUIREMENTS:

1. Use OpenAlex API (no rate limits, free)
2. Get comprehensive paper metadata
3. Access institution and funding data
4. Track concept hierarchies
5. Get work lineage (versions)
6. Support complex filtering

Class: OpenAlexFetcher
Methods:
- __init__(email): Initialize with polite pool email
- search_works(query, filters, per_page): Search papers
- get_work(openalex_id): Get single work
- get_author(author_id): Get author info
- get_institution(institution_id): Get institution info
- get_concept(concept_id): Get concept details
- get_related_works(work_id): Find related papers
- filter_by_concept(concept_name, filters): Search by research area

METADATA FROM OPENALEX:
- OpenAlex ID
- DOI and other IDs
- Title, abstract
- Authors with institutions and countries
- Concepts (research topics) with confidence scores
- Citation count
- Referenced works count
- Open access status and URL
- Publication date
- Journal/venue info
- Funding information
- License information

CITATION NETWORK BUILDING:
Create a unified citation graph combining data from all sources:

Class: CitationNetworkBuilder
Methods:
- add_paper(paper_data, source): Add paper to graph
- add_citation(citing_id, cited_id): Add citation edge
- merge_paper_ids(different_ids_same_paper): Handle same paper from different sources
- get_citation_depth(paper_id, max_depth): Get papers N citations away
- find_citation_paths(paper1_id, paper2_id): Find how papers are connected
- get_influential_papers(min_citations): Get highly cited papers
- export_graph(format): Export as JSON/GraphML

Example usage:

```python
# Semantic Scholar
ss_fetcher = SemanticScholarFetcher(api_key="your_key")
paper = ss_fetcher.get_paper("DOI:10.1234/example")
citations = ss_fetcher.get_paper_citations(paper['paperId'])

# OpenAlex
oa_fetcher = OpenAlexFetcher(email="me@uni.edu")
works = oa_fetcher.search_works(
    query="machine learning cancer",
    filters={"publication_year": ">2020", "cited_by_count": ">50"}
)

# Build network
network = CitationNetworkBuilder()
network.add_paper(paper, source="semantic_scholar")
for citation in citations:
    network.add_citation(citation['citingPaper']['paperId'], paper['paperId'])
```

Include rate limiting, caching, and error handling for both APIs.
```

---

### PROMPT 6: Text Processing and Embedding

```
Create a text processing pipeline that chunks papers and generates embeddings for vector search.

FILES:
- src/ingestion/parser.py
- src/ingestion/embedder.py

PARSER.PY REQUIREMENTS:

Create a comprehensive text parser that handles multiple formats:

Class: PaperParser
Methods:
- parse_pdf(file_path): Extract text from PDF
- parse_xml(xml_content, format): Parse XML (PubMed, arXiv, etc.)
- parse_html(html_content): Parse HTML papers
- clean_text(raw_text): Remove junk, fix formatting
- extract_sections(text): Identify paper sections
- chunk_text(text, chunk_size, overlap): Split into chunks
- extract_references(text): Parse reference list
- extract_figures_tables(content): Get captions and context

TEXT CLEANING:
- Remove headers/footers/page numbers
- Fix hyphenation across line breaks
- Normalize whitespace
- Handle special characters and equations
- Preserve section structure
- Remove boilerplate (acknowledgments, etc.)

SECTION EXTRACTION:
Identify and label sections:
- Title
- Abstract
- Introduction
- Methods/Methodology
- Results
- Discussion
- Conclusion
- References

CHUNKING STRATEGY:
- Default chunk size: 500 words
- Overlap: 50 words to preserve context
- Respect sentence boundaries
- Keep section labels with chunks
- Metadata for each chunk:
  - Chunk index
  - Section name
  - Start and end positions
  - Parent paper ID

EMBEDDER.PY REQUIREMENTS:

Create an embedding generation system using sentence-transformers:

Class: PaperEmbedder
Methods:
- __init__(model_name): Load embedding model
- embed_text(text): Generate embedding for single text
- embed_batch(texts, batch_size): Batch processing
- embed_paper(paper_dict): Embed all chunks of a paper
- compute_similarity(emb1, emb2): Calculate cosine similarity
- find_similar_chunks(query_emb, chunk_embeddings, top_k): Find similar chunks

EMBEDDING MODELS TO SUPPORT:
Primary: "allenai-specter" (scientific papers, 768 dim)
Alternatives:
- "sentence-transformers/all-MiniLM-L6-v2" (general purpose, 384 dim)
- "sentence-transformers/multi-qa-MiniLM-L6-cos-v1" (Q&A optimized)

OPTIMIZATION:
- GPU acceleration if available
- Batch processing for efficiency
- Caching of embeddings
- Progress tracking for large batches
- Memory management for large papers

Example usage:

```python
# Parse paper
parser = PaperParser()
text = parser.parse_pdf("paper.pdf")
cleaned = parser.clean_text(text)
sections = parser.extract_sections(cleaned)
chunks = parser.chunk_text(cleaned, chunk_size=500, overlap=50)

# Generate embeddings
embedder = PaperEmbedder(model_name="allenai-specter")
embeddings = embedder.embed_batch([chunk['text'] for chunk in chunks])

# Store with metadata
for chunk, embedding in zip(chunks, embeddings):
    store_in_db(
        embedding=embedding,
        text=chunk['text'],
        metadata={
            'section': chunk['section'],
            'chunk_index': chunk['index'],
            'paper_id': paper_id
        }
    )
```

Include comprehensive error handling, logging, and progress tracking.
```

---

## 🗄️ VECTOR DATABASE PHASE

### PROMPT 7: Chroma DB Manager

```
Create a comprehensive Chroma vector database manager for storing and querying paper embeddings.

FILE: src/database/chroma_manager.py

REQUIREMENTS:

1. Manage Chroma collections for papers
2. Support multiple search strategies
3. Handle metadata filtering
4. Implement efficient batch operations
5. Support collection management (create, delete, update)
6. Provide query optimization
7. Handle large-scale data

Class: ChromaManager
Methods:
- __init__(persist_directory, collection_name): Initialize
- create_collection(name, metadata): Create new collection
- get_collection(name): Get existing collection
- delete_collection(name): Delete collection
- add_papers(papers_data, batch_size): Add papers to collection
- add_paper(paper_id, embedding, metadata, text): Add single paper
- update_paper(paper_id, updates): Update paper data
- delete_paper(paper_id): Remove paper
- search(query_embedding, n_results, filters): Semantic search
- search_by_metadata(metadata_filter, n_results): Filter by metadata
- hybrid_search(query_embedding, metadata_filter, n_results): Combined search
- get_paper(paper_id): Retrieve specific paper
- get_statistics(): Get collection stats
- export_collection(output_path): Backup collection
- import_collection(input_path): Restore collection

SEARCH STRATEGIES:

1. SEMANTIC SEARCH:
```python
def search(self, query_embedding, n_results=10, filters=None):
    """
    Pure semantic/similarity search
    
    Args:
        query_embedding: Vector representation of query
        n_results: Number of results to return
        filters: Dict of metadata filters
        
    Returns:
        List of results with paper data and similarity scores
    """
```

2. METADATA FILTERING:
```python
def search_by_metadata(self, metadata_filter, n_results=100):
    """
    Filter papers by metadata (year, field, citations, etc.)
    
    Example filters:
    {
        "year": {"$gte": 2020},
        "field": "biology",
        "citations_count": {"$gt": 50}
    }
    """
```

3. HYBRID SEARCH:
```python
def hybrid_search(self, query_embedding, metadata_filter, n_results=10, weight=0.5):
    """
    Combine semantic similarity and metadata filtering
    
    Args:
        weight: Balance between semantic (1.0) and filter (0.0)
    """
```

4. MULTI-FIELD SEARCH:
```python
def search_across_fields(self, query_embedding, fields, n_results_per_field):
    """
    Search multiple scientific fields and aggregate results
    
    Args:
        fields: List of fields ["biology", "physics", "cs"]
        n_results_per_field: Results from each field
    
    Returns:
        Aggregated and deduplicated results
    """
```

METADATA SCHEMA:

Store comprehensive metadata for each paper:
```python
{
    "paper_id": "arxiv_2024_12345",
    "title": "Novel approach to...",
    "authors": ["Smith, J.", "Doe, A."],
    "year": 2024,
    "field": "biology",
    "subfield": "oncology",
    "source": "arxiv",  # arxiv, pubmed, semantic_scholar
    "abstract": "This paper presents...",
    "doi": "10.1234/example",
    "url": "https://arxiv.org/...",
    "citations_count": 45,
    "keywords": ["cancer", "metastasis", "ML"],
    "publication_venue": "Nature",
    "open_access": true,
    "chunk_index": 0,  # Which chunk of the paper
    "section": "introduction",
    "total_chunks": 25,
    "embedding_model": "allenai-specter",
    "ingestion_date": "2024-01-15"
}
```

BATCH OPERATIONS:

Efficient handling of large batches:
```python
def add_papers(self, papers_data, batch_size=100):
    """
    Add multiple papers efficiently
    
    Args:
        papers_data: List of dicts with embeddings, metadata, texts
        batch_size: Process in batches to manage memory
    
    Shows progress bar and handles errors gracefully
    """
    for i in range(0, len(papers_data), batch_size):
        batch = papers_data[i:i+batch_size]
        # Process batch
        # Handle failures without stopping entire process
```

STATISTICS AND MONITORING:

```python
def get_statistics(self):
    """
    Return collection statistics:
    - Total papers
    - Papers by field
    - Papers by year
    - Average citations
    - Most common keywords
    - Storage size
    """
```

ERROR HANDLING:
- Duplicate paper detection
- Invalid embedding dimensions
- Missing required metadata
- Database corruption recovery
- Concurrent access handling

Include comprehensive logging and monitoring capabilities.
```

---

### PROMPT 8: Metadata Store

```
Create a PostgreSQL-based metadata store for structured paper information and relationships.

FILE: src/database/metadata_store.py

REQUIREMENTS:

1. Use SQLAlchemy ORM for database operations
2. Store paper metadata, authors, citations
3. Track data provenance (sources)
4. Support complex queries
5. Handle relationships (authors, citations, concepts)
6. Provide efficient indexing
7. Support transactions and rollback

DATABASE SCHEMA:

Create these tables:

TABLE: papers
- id (primary key)
- paper_id (unique, indexed)
- title
- abstract
- year
- doi
- url
- source (arxiv, pubmed, etc.)
- citations_count
- field
- subfield
- publication_venue
- open_access
- created_at
- updated_at

TABLE: authors
- id (primary key)
- author_id (unique, from data source)
- name
- affiliation
- h_index
- email
- orcid

TABLE: paper_authors (junction table)
- paper_id (foreign key)
- author_id (foreign key)
- author_order
- corresponding_author

TABLE: citations
- id (primary key)
- citing_paper_id (foreign key)
- cited_paper_id (foreign key)
- context (text where citation appears)
- influential (boolean)

TABLE: keywords
- id (primary key)
- keyword (unique)

TABLE: paper_keywords (junction table)
- paper_id (foreign key)
- keyword_id (foreign key)
- confidence (0.0 to 1.0)

TABLE: concepts
- id (primary key)
- concept_id (from OpenAlex)
- name
- level (hierarchy level)
- parent_concept_id

TABLE: paper_concepts (junction table)
- paper_id (foreign key)
- concept_id (foreign key)
- score (0.0 to 1.0)

TABLE: ingestion_logs
- id (primary key)
- source
- papers_fetched
- papers_processed
- papers_failed
- start_time
- end_time
- status
- error_message

Class: MetadataStore
Methods:
- __init__(connection_string): Initialize database connection
- create_tables(): Create all tables
- add_paper(paper_data): Insert paper with all relationships
- get_paper(paper_id): Retrieve paper with all related data
- add_author(author_data): Add author
- link_author_to_paper(paper_id, author_id, order): Create relationship
- add_citation(citing_id, cited_id, context): Record citation
- get_citations(paper_id, direction): Get citing or cited papers
- get_citation_network(paper_id, depth): Get papers N citations away
- add_keywords(paper_id, keywords): Tag paper with keywords
- search_by_keyword(keywords, operator): Find papers (AND/OR keywords)
- search_by_author(author_name): Find author's papers
- search_by_field(field, subfield, year_from, year_to): Complex search
- get_trending_papers(field, days, min_citations): Recently popular papers
- get_statistics(): Database statistics
- export_to_json(output_path): Backup
- import_from_json(input_path): Restore

QUERY EXAMPLES:

```python
store = MetadataStore("postgresql://user:pass@localhost/papers")

# Complex search
papers = store.search_papers(
    fields=["biology", "computer_science"],
    year_from=2020,
    min_citations=10,
    keywords=["machine learning", "cancer"],
    keyword_operator="AND"
)

# Citation analysis
network = store.get_citation_network(
    paper_id="arxiv_2024_12345",
    depth=2,  # Papers that cite papers that cite this paper
    min_citations=5  # Filter low-quality papers
)

# Trending in field
trending = store.get_trending_papers(
    field="biology",
    subfield="oncology",
    days=30,
    min_citations=5
)

# Author collaboration network
collab = store.get_author_collaborations(
    author_id="author_123",
    max_depth=2
)
```

INDEXING STRATEGY:
Create indexes for:
- paper_id (unique)
- title (full-text)
- year
- field + subfield
- citations_count
- doi
- author name
- keyword

TRANSACTION SUPPORT:

```python
def add_paper_with_relationships(self, paper_data, authors, keywords, citations):
    """
    Add paper and all relationships in a single transaction
    Rollback if any operation fails
    """
    with self.session.begin():
        try:
            paper = self.add_paper(paper_data)
            for author in authors:
                self.link_author_to_paper(paper.id, author)
            for keyword in keywords:
                self.add_keyword(paper.id, keyword)
            for citation in citations:
                self.add_citation(paper.id, citation)
        except Exception as e:
            self.session.rollback()
            raise
```

Include migration scripts, backup/restore functionality, and comprehensive error handling.
```

---

## 🤖 LANGCHAIN AGENTS PHASE

### PROMPT 9: Base Agent Implementation

```
Create a base agent class that all specialized agents will inherit from.

FILE: src/agents/base_agent.py

REQUIREMENTS:

1. Use LangChain's Agent framework
2. Provide common functionality for all agents
3. Support tool usage
4. Handle errors gracefully
5. Track agent reasoning steps
6. Support both OpenAI and local LLMs (Ollama)
7. Implement memory for conversation context
8. Log all agent actions

Base Class: BaseResearchAgent

Methods:
- __init__(config, tools): Initialize agent
- setup_llm(): Configure language model
- setup_memory(): Configure conversation memory
- setup_agent(): Create LangChain agent
- run(query, context): Execute agent task
- run_async(query, context): Async execution
- add_tool(tool): Add new tool to agent
- get_reasoning_steps(): Return agent's thought process
- reset_memory(): Clear conversation history
- export_session(path): Save session for analysis

LLM SETUP:

Support multiple LLM providers:

```python
def setup_llm(self):
    """Configure LLM based on config"""
    if self.config.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.openai_api_key
        )
    elif self.config.llm_provider == \"ollama\":
        from langchain_community.llms import Ollama
        return Ollama(
            model=self.config.llm_model,
            temperature=self.temperature
        )
    else:
        raise ValueError(f\"Unsupported LLM provider: {self.config.llm_provider}\")
```

Include comprehensive error handling, logging, and session management.
```

---

## 🎨 FRONTEND & USER INTERFACE PHASE

### PROMPT 21: Streamlit Frontend Application

```
Create a comprehensive, user-friendly Streamlit frontend for the hypothesis engine.

FILE: frontend/app.py

REQUIREMENTS:

Build a modern, intuitive interface that allows researchers to:
1. Input their research questions
2. View hypothesis generation progress in real-time
3. Explore generated hypotheses interactively
4. Save and export results
5. Visualize citation networks
6. Browse paper recommendations

MAIN PAGE STRUCTURE:

```python
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List
import asyncio

def main():
    st.set_page_config(
        page_title="Scientific Hypothesis Cross-Pollination Engine",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better aesthetics
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .hypothesis-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 5px solid #1E88E5;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.image("logo.png", width=200)  # Add your logo
        st.title("Configuration")
        
        # User preferences
        user_field = st.selectbox(
            "Your Research Field",
            ["Biology", "Physics", "Chemistry", "Computer Science", 
             "Mathematics", "Engineering", "Medicine", "Other"]
        )
        
        expertise_level = st.select_slider(
            "Expertise Level",
            options=["Undergraduate", "Graduate", "Postdoc", "Professor", "Industry"]
        )
        
        # Search preferences
        st.subheader("Search Preferences")
        date_range = st.slider(
            "Publication Years",
            min_value=2000,
            max_value=2024,
            value=(2020, 2024)
        )
        
        min_citations = st.number_input(
            "Minimum Citations",
            min_value=0,
            max_value=1000,
            value=5
        )
        
        num_hypotheses = st.slider(
            "Number of Hypotheses to Generate",
            min_value=3,
            max_value=15,
            value=5
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            creativity = st.slider(
                "Creativity Level",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Higher values encourage more novel but riskier suggestions"
            )
            
            include_preprints = st.checkbox("Include Preprints", value=True)
            cross_discipline = st.checkbox("Enable Cross-Disciplinary Search", value=True)
    
    # Main content area
    st.markdown('<h1 class="main-header">🔬 Scientific Hypothesis Cross-Pollination Engine</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Discover Novel Research Directions Through AI-Powered Cross-Disciplinary Insights
    
    This system helps you find innovative approaches to your research problems by:
    - 🔍 Analyzing millions of papers across all scientific fields
    - 🧠 Identifying unexpected connections between disciplines  
    - 💡 Generating testable hypotheses with implementation plans
    - 📊 Providing evidence-based recommendations
    """)
    
    # Input section
    st.header("1️⃣ Describe Your Research Question")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        research_query = st.text_area(
            "What problem are you trying to solve?",
            placeholder="Example: I'm studying how cancer cells migrate through blood vessels. "
                        "Current imaging techniques are limited to 2D. Are there better approaches?",
            height=150
        )
    
    with col2:
        st.info("""
        **Tips for best results:**
        - Describe the specific problem
        - Mention current limitations
        - Include relevant context
        - Be as detailed as possible
        """)
    
    # Additional context (optional)
    with st.expander("➕ Add Additional Context (Optional)"):
        current_approach = st.text_area(
            "What approaches have you tried?",
            placeholder="Describe methods you've already explored..."
        )
        
        specific_challenges = st.text_area(
            "What are the main challenges?",
            placeholder="What obstacles are preventing progress?"
        )
        
        resources_available = st.multiselect(
            "What resources do you have access to?",
            ["High-performance computing", "Specialized lab equipment", 
             "Large datasets", "Collaboration opportunities", "Funding"]
        )
    
    # Generate button
    if st.button("🚀 Generate Hypotheses", type="primary", use_container_width=True):
        if not research_query:
            st.error("Please enter a research question first!")
        else:
            generate_hypotheses(
                research_query, 
                user_field,
                date_range,
                min_citations,
                num_hypotheses,
                creativity
            )

def generate_hypotheses(query, field, date_range, min_citations, num_hypotheses, creativity):
    """Execute the hypothesis generation workflow"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create tabs for different stages
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Discovery", "💡 Hypotheses", "📊 Insights", "📄 Report"
    ])
    
    with tab1:
        st.subheader("Research Discovery Process")
        
        # Stage 1: Primary domain search
        status_text.text("🔍 Searching your field...")
        progress_bar.progress(20)
        
        with st.spinner("Analyzing papers in " + field + "..."):
            primary_results = search_primary_domain(query, field)
            
        st.success(f"✅ Found {len(primary_results)} relevant papers in {field}")
        
        # Show sample papers
        with st.expander("Preview Relevant Papers"):
            for i, paper in enumerate(primary_results[:5]):
                display_paper_card(paper, i)
        
        # Stage 2: Cross-domain search  
        status_text.text("🌐 Searching other fields for analogies...")
        progress_bar.progress(40)
        
        with st.spinner("Exploring connections across disciplines..."):
            cross_domain_results = search_cross_domain(query, field)
        
        st.success(f"✅ Found {len(cross_domain_results)} promising connections")
        
        # Visualize cross-domain findings
        fig = create_field_distribution_chart(cross_domain_results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Stage 3: Methodology analysis
        status_text.text("⚙️ Analyzing methodologies...")
        progress_bar.progress(60)
        
        methodology_insights = analyze_methodologies(primary_results, cross_domain_results)
        
        # Display methodology summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Methods Found", methodology_insights['unique_methods'])
        with col2:
            st.metric("Transferable Techniques", methodology_insights['transferable'])
        with col3:
            st.metric("Novel Combinations", methodology_insights['novel_combos'])
    
    with tab2:
        st.subheader("Generated Hypotheses")
        
        # Stage 4: Generate hypotheses
        status_text.text("💡 Generating hypotheses...")
        progress_bar.progress(80)
        
        with st.spinner("Synthesizing insights into testable hypotheses..."):
            hypotheses = generate_hypothesis_list(
                query, primary_results, cross_domain_results, 
                methodology_insights, num_hypotheses
            )
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        st.success(f"🎉 Generated {len(hypotheses)} novel hypotheses!")
        
        # Display hypotheses with interactive elements
        for i, hypo in enumerate(hypotheses):
            display_hypothesis_card(hypo, i)
    
    with tab3:
        st.subheader("Key Insights & Visualizations")
        
        # Citation network
        st.markdown("#### 📊 Citation Network")
        citation_network = build_citation_network(primary_results + cross_domain_results)
        fig_network = create_network_visualization(citation_network)
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Timeline of research
        st.markdown("#### 📅 Research Timeline")
        fig_timeline = create_timeline_visualization(primary_results)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Keyword cloud
        st.markdown("#### ☁️ Key Concepts")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(create_wordcloud(primary_results))
        with col2:
            display_top_keywords(primary_results)
    
    with tab4:
        st.subheader("📄 Comprehensive Report")
        
        report = generate_comprehensive_report(
            query, hypotheses, primary_results, 
            cross_domain_results, methodology_insights
        )
        
        # Display report
        st.markdown(report)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download as PDF",
                data=generate_pdf_report(report),
                file_name="hypothesis_report.pdf",
                mime="application/pdf"
            )
        
        with col2:
            st.download_button(
                label="📊 Export to Excel",
                data=generate_excel_export(hypotheses),
                file_name="hypotheses.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        with col3:
            st.download_button(
                label="📋 Copy to Clipboard",
                data=report,
                file_name="report.md",
                mime="text/markdown"
            )

def display_hypothesis_card(hypothesis: Dict, index: int):
    """Display a single hypothesis in an interactive card"""
    
    with st.container():
        st.markdown(f"""
        <div class="hypothesis-card">
            <h3>💡 Hypothesis {index + 1}: {hypothesis['title']}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Novelty Score", f"{hypothesis['novelty_score']:.1f}/10")
        col2.metric("Feasibility", f"{hypothesis['feasibility_score']:.1f}/10")
        col3.metric("Impact Potential", f"{hypothesis['impact_score']:.1f}/10")
        
        st.markdown(f"**Description:** {hypothesis['description']}")
        
        # Expandable sections
        with st.expander("📚 Supporting Evidence"):
            for paper in hypothesis['supporting_papers']:
                st.markdown(f"- **{paper['title']}** ({paper['year']}) - {paper['authors']}")
                st.caption(paper['relevance'])
        
        with st.expander("🔬 Implementation Plan"):
            for step_num, step in enumerate(hypothesis['implementation_steps'], 1):
                st.markdown(f"**Step {step_num}:** {step['action']}")
                st.caption(f"⏱️ {step['timeline']} | 💰 {step['cost']} | "
                          f"🎯 {step['difficulty']}")
        
        with st.expander("⚠️ Challenges & Mitigation"):
            for challenge in hypothesis['challenges']:
                st.warning(f"**Challenge:** {challenge['issue']}")
                st.info(f"**Mitigation:** {challenge['mitigation']}")
        
        with st.expander("🎯 Required Resources"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Equipment:**")
                for item in hypothesis['resources']['equipment']:
                    st.markdown(f"- {item}")
            with col2:
                st.markdown("**Expertise:**")
                for skill in hypothesis['resources']['expertise']:
                    st.markdown(f"- {skill}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        if col1.button(f"⭐ Save Hypothesis {index + 1}", key=f"save_{index}"):
            save_hypothesis(hypothesis)
            st.success("Saved to your collection!")
        
        if col2.button(f"📤 Share", key=f"share_{index}"):
            share_link = generate_share_link(hypothesis)
            st.code(share_link)
        
        if col3.button(f"📧 Email to Collaborators", key=f"email_{index}"):
            show_email_dialog(hypothesis)

def display_paper_card(paper: Dict, index: int):
    """Display a research paper in a compact card"""
    with st.container():
        st.markdown(f"**{paper['title']}**")
        st.caption(f"{', '.join(paper['authors'][:3])} et al. ({paper['year']})")
        st.caption(f"📖 {paper['venue']} | 📊 {paper['citations']} citations")
        
        if st.button(f"View Abstract", key=f"abstract_{index}"):
            st.info(paper['abstract'])

# Helper visualization functions
def create_field_distribution_chart(results):
    """Create a chart showing distribution of papers across fields"""
    field_counts = {}
    for r in results:
        field = r['field']
        field_counts[field] = field_counts.get(field, 0) + 1
    
    fig = go.Figure(data=[
        go.Bar(x=list(field_counts.keys()), y=list(field_counts.values()))
    ])
    fig.update_layout(
        title="Cross-Domain Discovery: Papers by Field",
        xaxis_title="Scientific Field",
        yaxis_title="Number of Papers"
    )
    return fig

def create_network_visualization(network_data):
    """Create an interactive citation network visualization"""
    # Implementation using plotly or networkx
    pass

def create_timeline_visualization(papers):
    """Create timeline of research evolution"""
    # Group papers by year and create timeline
    pass

# Run the app
if __name__ == "__main__":
    main()
```

ADDITIONAL COMPONENTS:

Create separate files for modular components:

FILE: frontend/components/input_form.py
- Research question input with autocomplete
- Context gathering forms
- File upload for existing research

FILE: frontend/components/results_display.py
- Hypothesis cards with interactive elements
- Paper browsing interface
- Filtering and sorting options

FILE: frontend/components/visualization.py
- Citation network graphs
- Timeline visualizations
- Concept maps
- Statistical charts

FILE: frontend/components/export.py
- PDF report generation
- Excel export
- LaTeX grant proposal generator
- Markdown documentation

Include proper state management, session handling, and responsive design.
```

---

## 🧪 TESTING & VALIDATION PHASE

### PROMPT 22: Comprehensive Testing Suite

```
Create a complete testing suite for the hypothesis engine.

FILES:
- tests/test_ingestion.py
- tests/test_database.py
- tests/test_agents.py
- tests/test_hypothesis.py
- tests/test_integration.py
- tests/test_api.py

REQUIREMENTS:

Use pytest framework with the following test categories:

1. UNIT TESTS - Test individual components

FILE: tests/test_ingestion.py

```python
import pytest
from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.ingestion.parser import PaperParser
from src.ingestion.embedder import PaperEmbedder

class TestArxivFetcher:
    """Test arXiv paper fetching functionality"""
    
    @pytest.fixture
    def fetcher(self):
        return ArxivFetcher(config=test_config)
    
    def test_search_papers(self, fetcher):
        """Test basic paper search"""
        results = fetcher.search_papers(
            query="machine learning",
            max_results=10
        )
        assert len(results) == 10
        assert all('title' in r for r in results)
        assert all('authors' in r for r in results)
    
    def test_download_paper(self, fetcher, tmp_path):
        """Test PDF download"""
        paper_id = "2301.00001"
        pdf_path = fetcher.download_paper(paper_id, tmp_path)
        assert pdf_path.exists()
        assert pdf_path.suffix == '.pdf'
    
    def test_network_error_handling(self, fetcher, monkeypatch):
        """Test handling of network failures"""
        def mock_request_failure(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Network error")
        
        monkeypatch.setattr(requests, 'get', mock_request_failure)
        
        with pytest.raises(NetworkError):
            fetcher.search_papers("test query")
    
    def test_rate_limiting(self, fetcher):
        """Test that rate limiting is respected"""
        import time
        start = time.time()
        
        for i in range(5):
            fetcher.search_papers(f"query{i}", max_results=1)
        
        elapsed = time.time() - start
        # Should have delays between requests
        assert elapsed > 1.0

class TestPaperParser:
    """Test text parsing and cleaning"""
    
    @pytest.fixture
    def parser(self):
        return PaperParser()
    
    def test_pdf_parsing(self, parser, sample_pdf_path):
        """Test PDF text extraction"""
        text = parser.parse_pdf(sample_pdf_path)
        assert isinstance(text, str)
        assert len(text) > 100
    
    def test_section_extraction(self, parser, sample_paper_text):
        """Test identification of paper sections"""
        sections = parser.extract_sections(sample_paper_text)
        
        assert 'abstract' in sections
        assert 'introduction' in sections
        assert 'methods' in sections
        assert len(sections['abstract']) > 0
    
    def test_text_chunking(self, parser):
        """Test text chunking with overlap"""
        text = "word " * 1000  # 1000 words
        chunks = parser.chunk_text(text, chunk_size=100, overlap=10)
        
        # Should create multiple chunks
        assert len(chunks) > 5
        
        # Chunks should overlap
        assert chunks[0]['text'][-20:] in chunks[1]['text'][:50]
    
    def test_reference_extraction(self, parser, sample_paper_text):
        """Test extraction of cited papers"""
        references = parser.extract_references(sample_paper_text)
        assert isinstance(references, list)
        assert len(references) > 0
        assert all('title' in ref or 'authors' in ref for ref in references)

class TestPaperEmbedder:
    """Test embedding generation"""
    
    @pytest.fixture
    def embedder(self):
        return PaperEmbedder(model_name="allenai-specter")
    
    def test_embed_text(self, embedder):
        """Test single text embedding"""
        text = "This is a test paper about machine learning"
        embedding = embedder.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 768  # SPECTER dimension
        assert not np.isnan(embedding).any()
    
    def test_batch_embedding(self, embedder):
        """Test batch processing"""
        texts = [f"Sample text {i}" for i in range(100)]
        embeddings = embedder.embed_batch(texts, batch_size=10)
        
        assert len(embeddings) == 100
        assert all(emb.shape[0] == 768 for emb in embeddings)
    
    def test_similarity_computation(self, embedder):
        """Test cosine similarity calculation"""
        emb1 = embedder.embed_text("machine learning neural networks")
        emb2 = embedder.embed_text("deep learning artificial intelligence")
        emb3 = embedder.embed_text("cooking recipe ingredients")
        
        sim_12 = embedder.compute_similarity(emb1, emb2)
        sim_13 = embedder.compute_similarity(emb1, emb3)
        
        # Related texts should be more similar
        assert sim_12 > sim_13
        assert 0 <= sim_12 <= 1
        assert 0 <= sim_13 <= 1
```

2. DATABASE TESTS

FILE: tests/test_database.py

```python
class TestChromaManager:
    """Test vector database operations"""
    
    @pytest.fixture
    def chroma_manager(self, tmp_path):
        return ChromaManager(persist_directory=tmp_path)
    
    def test_add_and_retrieve_paper(self, chroma_manager):
        """Test adding and retrieving papers"""
        paper_data = {
            'paper_id': 'test_001',
            'embedding': np.random.rand(768),
            'metadata': {
                'title': 'Test Paper',
                'year': 2024,
                'field': 'computer_science'
            },
            'text': 'This is a test paper'
        }
        
        chroma_manager.add_paper(**paper_data)
        retrieved = chroma_manager.get_paper('test_001')
        
        assert retrieved['metadata']['title'] == 'Test Paper'
        assert retrieved['metadata']['year'] == 2024
    
    def test_semantic_search(self, chroma_manager, populated_db):
        """Test similarity search"""
        query_embedding = np.random.rand(768)
        results = chroma_manager.search(
            query_embedding=query_embedding,
            n_results=10
        )
        
        assert len(results) <= 10
        assert all('distance' in r for r in results)
        # Results should be sorted by similarity
        distances = [r['distance'] for r in results]
        assert distances == sorted(distances)
    
    def test_metadata_filtering(self, chroma_manager, populated_db):
        """Test filtering by metadata"""
        results = chroma_manager.search_by_metadata(
            metadata_filter={
                'year': {'$gte': 2020},
                'field': 'biology'
            },
            n_results=50
        )
        
        for r in results:
            assert r['metadata']['year'] >= 2020
            assert r['metadata']['field'] == 'biology'
    
    def test_batch_operations(self, chroma_manager):
        """Test adding multiple papers efficiently"""
        papers_data = [
            {
                'paper_id': f'test_{i}',
                'embedding': np.random.rand(768),
                'metadata': {'title': f'Paper {i}'},
                'text': f'Text {i}'
            }
            for i in range(100)
        ]
        
        chroma_manager.add_papers(papers_data, batch_size=20)
        stats = chroma_manager.get_statistics()
        
        assert stats['total_papers'] == 100

3. AGENT TESTS

FILE: tests/test_agents.py

```python
class TestPrimaryDomainAgent:
    """Test primary domain agent"""
    
    @pytest.fixture
    def agent(self, mock_llm, mock_tools):
        return PrimaryDomainAgent(
            config=test_config,
            tools=mock_tools
        )
    
    def test_field_identification(self, agent):
        """Test identifying research field from query"""
        query = "I'm studying cancer cell migration using microscopy"
        field_info = agent.identify_field(query)
        
        assert field_info['field'] == 'biology'
        assert field_info['subfield'] == 'oncology'
        assert field_info['confidence'] > 0.7
    
    def test_knowledge_gap_identification(self, agent):
        """Test finding research gaps"""
        problem = "tracking individual cells in 3D"
        existing_work = load_test_papers('cell_tracking')
        
        gaps = agent.identify_knowledge_gaps(problem, existing_work)
        
        assert isinstance(gaps, list)
        assert len(gaps) > 0
        assert all(isinstance(gap, str) for gap in gaps)
    
    def test_agent_reasoning_tracking(self, agent):
        """Test that agent reasoning steps are captured"""
        result = agent.run("test query")
        
        assert 'reasoning_steps' in result
        assert len(result['reasoning_steps']) > 0
        assert all('action' in step for step in result['reasoning_steps'])

class TestCrossDomainAgent:
    """Test cross-domain discovery agent"""
    
    def test_problem_abstraction(self, agent):
        """Test converting specific problem to abstract form"""
        specific = "How do cancer cells migrate through blood vessels?"
        abstract = agent.abstract_problem(specific)
        
        # Should remove domain-specific terms
        assert "cancer" not in abstract.lower()
        assert "blood vessels" not in abstract.lower()
        # Should keep core concepts
        assert "migrate" in abstract.lower() or "movement" in abstract.lower()
    
    def test_multi_field_search(self, agent):
        """Test searching across multiple fields"""
        query = "tracking particles in constrained networks"
        exclude_field = "biology"
        
        results = agent.search_multiple_fields(query, exclude_field)
        
        assert 'physics' in results
        assert 'computer_science' in results
        assert 'biology' not in results
    
    def test_analogy_finding(self, agent):
        """Test finding structural analogies"""
        problem = "tracking cancer cells in tissue"
        other_field_paper = {
            'title': 'Particle tracking in turbulent flows',
            'field': 'physics'
        }
        
        analogy = agent.find_analogies(problem, other_field_paper)
        
        assert 'similarity_score' in analogy
        assert analogy['similarity_score'] > 0
        assert 'explanation' in analogy

4. INTEGRATION TESTS

FILE: tests/test_integration.py

```python
class TestEndToEndWorkflow:
    """Test complete hypothesis generation workflow"""
    
    def test_full_hypothesis_generation(self, test_db, test_config):
        """Test entire pipeline from query to hypotheses"""
        query = "Novel approaches to cancer metastasis detection?"
        
        # Step 1: Primary domain search
        primary_agent = PrimaryDomainAgent(config=test_config)
        primary_results = primary_agent.run(query)
        
        assert primary_results['success']
        assert len(primary_results['relevant_papers']) > 0
        
        # Step 2: Cross-domain search
        crossdomain_agent = CrossDomainAgent(config=test_config)
        crossdomain_results = crossdomain_agent.run(
            query,
            primary_field=primary_results['field']['primary']
        )
        
        assert crossdomain_results['success']
        assert len(crossdomain_results['transferable_methods']) > 0
        
        # Step 3: Generate hypotheses
        generator = HypothesisGenerator(config=test_config)
        hypotheses = generator.generate(
            query=query,
            primary_findings=primary_results,
            crossdomain_findings=crossdomain_results
        )
        
        assert len(hypotheses) > 0
        assert all('title' in h for h in hypotheses)
        assert all('novelty_score' in h for h in hypotheses)
        assert all('implementation_steps' in h for h in hypotheses)
        
        # Step 4: Validate hypotheses
        validator = HypothesisValidator(config=test_config)
        for hypo in hypotheses:
            validation = validator.validate(hypo)
            assert 'is_novel' in validation
            assert 'feasibility_score' in validation
    
    def test_api_to_frontend_flow(self, test_client):
        """Test API endpoints work with frontend"""
        # Submit query to API
        response = test_client.post('/api/hypotheses/generate', json={
            'query': 'test research question',
            'field': 'biology',
            'preferences': {'creativity': 0.7}
        })
        
        assert response.status_code == 200
        job_id = response.json()['job_id']
        
        # Poll for results
        for i in range(30):  # Wait up to 30 seconds
            status_response = test_client.get(f'/api/hypotheses/status/{job_id}')
            if status_response.json()['status'] == 'complete':
                break
            time.sleep(1)
        
        # Get results
        results_response = test_client.get(f'/api/hypotheses/results/{job_id}')
        assert results_response.status_code == 200
        
        results = results_response.json()
        assert 'hypotheses' in results
        assert len(results['hypotheses']) > 0

5. PERFORMANCE TESTS

FILE: tests/test_performance.py

```python
class TestPerformance:
    """Test system performance and scalability"""
    
    def test_search_latency(self, chroma_manager, populated_large_db):
        """Test search response time"""
        query_embedding = np.random.rand(768)
        
        start = time.time()
        results = chroma_manager.search(query_embedding, n_results=10)
        elapsed = time.time() - start
        
        # Should complete in under 1 second even with large DB
        assert elapsed < 1.0
    
    def test_concurrent_requests(self, api_client):
        """Test handling multiple simultaneous requests"""
        import concurrent.futures
        
        def make_request(i):
            return api_client.post('/api/hypotheses/generate', json={
                'query': f'test query {i}'
            })
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            results = [f.result() for f in futures]
        
        # All requests should succeed
        assert all(r.status_code in [200, 202] for r in results)
    
    def test_memory_usage(self, embedder):
        """Test memory doesn't grow unbounded"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process 1000 texts
        for i in range(10):
            texts = [f"Sample text {j}" for j in range(100)]
            embeddings = embedder.embed_batch(texts)
            del embeddings
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (< 500MB)
        assert memory_growth < 500

# Fixtures and test utilities
@pytest.fixture
def test_config():
    """Provide test configuration"""
    return Config(
        llm_provider='ollama',  # Use local LLM for tests
        chroma_persist_dir=':memory:',  # In-memory for speed
        # ... other test settings
    )

@pytest.fixture
def populated_db(chroma_manager):
    """Populate database with test data"""
    test_papers = load_test_papers('sample_papers.json')
    chroma_manager.add_papers(test_papers)
    return chroma_manager

@pytest.fixture
def mock_tools():
    """Provide mock tools for agent testing"""
    # Return mock versions of tools that don't require actual API calls
    pass

# Run tests with: pytest tests/ -v --cov=src
```

TESTING GUIDELINES:

1. Achieve >80% code coverage
2. Test both happy paths and error cases
3. Use mocks to avoid external API calls in tests
4. Include performance benchmarks
5. Test with realistic data volumes
6. Validate all user-facing functionality
7. Include integration tests for critical workflows
8. Test concurrent usage scenarios
9. Validate data integrity and consistency
10. Test graceful degradation when services are unavailable

Include CI/CD configuration for automated testing:

FILE: .github/workflows/test.yml

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## 🚀 DEPLOYMENT & DEVOPS PHASE

### PROMPT 23: Docker & Kubernetes Deployment

```
Create comprehensive deployment configurations for the hypothesis engine.

FILES:
- Dockerfile
- docker-compose.yml
- kubernetes/deployment.yml
- kubernetes/service.yml
- .dockerignore
- deployment/scripts/deploy.sh

REQUIREMENTS:

Build production-ready containerized deployment with:
1. Multi-stage Docker builds for optimization
2. Docker Compose for local development
3. Kubernetes configurations for production
4. Health checks and readiness probes
5. Environment-specific configurations
6. Automated deployment scripts

DOCKERFILE:

Create optimized multi-stage build:

```dockerfile
# Multi-stage build for Python application
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY .env.example .env

# Create data directories
RUN mkdir -p data/embeddings data/raw data/processed

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

DOCKER-COMPOSE.YML:

Create complete local development environment:

```yaml
version: '3.8'

services:
  # Vector Database
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_AUTH_PROVIDER=chromadb.auth.token.TokenAuthServerProvider
      - CHROMA_SERVER_AUTH_TOKEN_TRANSPORT_HEADER=X-Chroma-Token
    networks:
      - hypothesis-network

  # PostgreSQL for metadata
  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=papers_metadata
      - POSTGRES_USER=hypothesis_user
      - POSTGRES_PASSWORD=secure_password_change_me
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./deployment/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - hypothesis-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hypothesis_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - hypothesis-network
    command: redis-server --appendonly yes

  # Hypothesis Engine API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://hypothesis_user:secure_password_change_me@postgres:5432/papers_metadata
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=openai
    volumes:
      - ./src:/app/src
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      chromadb:
        condition: service_started
      redis:
        condition: service_started
    networks:
      - hypothesis-network
    restart: unless-stopped

  # Streamlit Frontend
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    command: streamlit run frontend/app.py --server.port=8501 --server.address=0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    volumes:
      - ./frontend:/app/frontend
    depends_on:
      - api
    networks:
      - hypothesis-network
    restart: unless-stopped

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deployment/ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - frontend
    networks:
      - hypothesis-network
    restart: unless-stopped

networks:
  hypothesis-network:
    driver: bridge

volumes:
  chroma_data:
  postgres_data:
  redis_data:
```

KUBERNETES DEPLOYMENT:

FILE: kubernetes/namespace.yml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: hypothesis-engine
```

FILE: kubernetes/configmap.yml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hypothesis-config
  namespace: hypothesis-engine
data:
  DATABASE_URL: "postgresql://hypothesis_user:password@postgres-service:5432/papers_metadata"
  CHROMA_HOST: "chromadb-service"
  CHROMA_PORT: "8000"
  REDIS_URL: "redis://redis-service:6379"
  LLM_PROVIDER: "openai"
  LOG_LEVEL: "INFO"
```

FILE: kubernetes/secrets.yml

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: hypothesis-secrets
  namespace: hypothesis-engine
type: Opaque
stringData:
  openai-api-key: "your-api-key-here"
  postgres-password: "secure-password"
  jwt-secret: "jwt-signing-secret"
```

FILE: kubernetes/deployment-api.yml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypothesis-api
  namespace: hypothesis-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypothesis-api
  template:
    metadata:
      labels:
        app: hypothesis-api
    spec:
      containers:
      - name: api
        image: your-registry/hypothesis-engine:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: hypothesis-secrets
              key: openai-api-key
        envFrom:
        - configMapRef:
            name: hypothesis-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: hypothesis-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: hypothesis-api-service
  namespace: hypothesis-engine
spec:
  selector:
    app: hypothesis-api
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

FILE: kubernetes/pvc.yml

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hypothesis-data-pvc
  namespace: hypothesis-engine
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

FILE: kubernetes/ingress.yml

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hypothesis-ingress
  namespace: hypothesis-engine
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - hypothesis.yourdomain.com
    secretName: hypothesis-tls
  rules:
  - host: hypothesis.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: hypothesis-api-service
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hypothesis-frontend-service
            port:
              number: 8501
```

DEPLOYMENT SCRIPTS:

FILE: deployment/scripts/deploy.sh

```bash
#!/bin/bash
set -e

echo "🚀 Deploying Hypothesis Engine..."

# Configuration
REGISTRY="your-docker-registry"
IMAGE_NAME="hypothesis-engine"
VERSION=$(git rev-parse --short HEAD)
NAMESPACE="hypothesis-engine"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} .
docker tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:latest

# Push to registry
echo "⬆️  Pushing to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
docker push ${REGISTRY}/${IMAGE_NAME}:latest

# Apply Kubernetes configurations
echo "☸️  Applying Kubernetes configurations..."
kubectl apply -f kubernetes/namespace.yml
kubectl apply -f kubernetes/configmap.yml
kubectl apply -f kubernetes/secrets.yml
kubectl apply -f kubernetes/pvc.yml
kubectl apply -f kubernetes/deployment-api.yml
kubectl apply -f kubernetes/deployment-frontend.yml
kubectl apply -f kubernetes/service.yml
kubectl apply -f kubernetes/ingress.yml

# Wait for rollout
echo "⏳ Waiting for deployment..."
kubectl rollout status deployment/hypothesis-api -n ${NAMESPACE}
kubectl rollout status deployment/hypothesis-frontend -n ${NAMESPACE}

echo "✅ Deployment complete!"
echo "🌐 Access the application at: https://hypothesis.yourdomain.com"
```

CI/CD PIPELINE:

FILE: .github/workflows/deploy.yml

```yaml
name: Build and Deploy

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}
    
    - name: Deploy to staging
      run: |
       ./deployment/scripts/deploy.sh staging

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}
    
    - name: Deploy to production
      run: |
        ./deployment/scripts/deploy.sh production
```

Include monitoring, logging configuration, and rollback procedures.
```

---

## 🔄 USER WORKFLOWS & FEATURES PHASE

### PROMPT 24: Grant Writing Assistant Implementation

```
Create a comprehensive grant writing assistant module that helps researchers develop compelling proposals.

FILE: src/features/grant_assistant.py

REQUIREMENTS:

Build a specialized agent that:
1. Analyzes grant requirements and RFPs
2. Maps hypotheses to grant opportunities
3. Generates proposal sections
4. Provides writing suggestions
5. Checks compliance with guidelines
6. Formats proposals to specifications

GRANT ASSISTANT IMPLEMENTATION:

```python
from langchain.agents import Agent
from langchain.prompts import PromptTemplate
from typing import Dict, List
import json

class GrantWritingAssistant:
    """
    Helps researchers convert hypotheses into fundable grant proposals
    """
    
    def __init__(self, config, llm):
        self.config = config
        self.llm = llm
        self.templates = self._load_templates()
    
    def analyze_rfp(self, rfp_text: str) -> Dict:
        """
        Analyze Request for Proposals to extract key requirements
        
        Args:
            rfp_text: Full text of RFP document
        
        Returns:
            Dict with extracted requirements, keywords, evaluation criteria
        """
        analysis_prompt = PromptTemplate(
            template="""
            Analyze this grant RFP and extract the following information:
            
            RFP Text:
            {rfp_text}
            
            Extract and provide:
            1. Funding amount and duration
            2. Research priorities and themes
            3. Required sections
            4. Evaluation criteria and weights
            5. Eligibility requirements
            6. Key deadlines
            7. Important keywords and terminology
            8. Broader impacts requirements
            9. Technical requirements
            10. Collaboration preferences
            
            Provide structured JSON output.
            """,
            input_variables=["rfp_text"]
        )
        
        response = self.llm.invoke(analysis_prompt.format(rfp_text=rfp_text))
        return json.loads(response)
    
    def match_hypothesis_to_grant(
        self, 
        hypothesis: Dict, 
        grant_requirements: Dict
    ) -> Dict:
        """
        Assess how well a hypothesis aligns with grant requirements
        
        Returns alignment score and specific matching points
        """
        matching_prompt = PromptTemplate(
            template="""
            Assess the alignment between this research hypothesis and grant requirements:
            
            Hypothesis:
            Title: {hypo_title}
            Description: {hypo_desc}
            Innovation: {hypo_innovation}
            Impact: {hypo_impact}
            
            Grant Requirements:
            Priorities: {grant_priorities}
            Keywords: {grant_keywords}
            Evaluation Criteria: {grant_criteria}
            
            Provide:
            1. Alignment score (0-100)
            2. Specific matching points
            3. Gaps that need addressing
            4. Suggestions to strengthen alignment
            5. Recommended positioning strategy
            
            Format as JSON.
            """,
            input_variables=[
                "hypo_title", "hypo_desc", "hypo_innovation", "hypo_impact",
                "grant_priorities", "grant_keywords", "grant_criteria"
            ]
        )
        
        response = self.llm.invoke(matching_prompt.format(
            hypo_title=hypothesis['title'],
            hypo_desc=hypothesis['description'],
            hypo_innovation=hypothesis.get('novelty_justification', ''),
            hypo_impact=hypothesis.get('impact_potential', ''),
            grant_priorities=grant_requirements['priorities'],
            grant_keywords=grant_requirements['keywords'],
            grant_criteria=grant_requirements['evaluation_criteria']
        ))
        
        return json.loads(response)
    
    def generate_specific_aims(
        self,
        hypothesis: Dict,
        grant_requirements: Dict,
        max_length: int = 1000
    ) -> str:
        """
        Generate Specific Aims section for NIH-style grants
        """
        aims_prompt = PromptTemplate(
            template="""
            Write a compelling Specific Aims section for a grant proposal.
            
            Hypothesis Details:
            {hypothesis}
            
            Grant Requirements:
            {requirements}
            
            The Specific Aims should:
            1. Start with a strong opening paragraph establishing significance
            2. Include 2-3 specific aims that are:
               - Conceptually distinct but complementary
               - Feasible within the grant period
               - Aligned with grant priorities
            3. End with expected outcomes and impact
            4. Be exactly {max_length} words or less
            5. Use active voice and clear language
            6. Include relevant keywords: {keywords}
            
            Follow NIH format and style guidelines.
            """,
            input_variables=["hypothesis", "requirements", "max_length", "keywords"]
        )
        
        aims = self.llm.invoke(aims_prompt.format(
            hypothesis=json.dumps(hypothesis, indent=2),
            requirements=json.dumps(grant_requirements, indent=2),
            max_length=max_length,
            keywords=", ".join(grant_requirements.get('keywords', []))
        ))
        
        return aims
    
    def generate_research_strategy(
        self,
        hypothesis: Dict,
        implementation_plan: List[Dict],
        grant_requirements: Dict
    ) -> Dict:
        """
        Generate complete Research Strategy section with:
        - Significance
        - Innovation
        - Approach
        """
        sections = {}
        
        # Significance
        sections['significance'] = self._generate_significance(
            hypothesis, grant_requirements
        )
        
        # Innovation
        sections['innovation'] = self._generate_innovation(
            hypothesis, grant_requirements
        )
        
        # Approach
        sections['approach'] = self._generate_approach(
            hypothesis, implementation_plan, grant_requirements
        )
        
        return sections
    
    def _generate_significance(self, hypothesis: Dict, requirements: Dict) -> str:
        """Generate Significance section"""
        sig_prompt = PromptTemplate(
            template="""
            Write the Significance section for a grant proposal.
            
            Hypothesis: {hypothesis}
            Grant Focus: {focus}
            
            The section should:
            1. Explain the problem and why it's important
            2. Describe current limitations
            3. Explain how this research addresses the gap
            4. Describe expected outcomes and impact
            5. Connect to grant priorities: {priorities}
            
            Length: 800-1000 words
            Style: Compelling but not overstated
            """,
            input_variables=["hypothesis", "focus", "priorities"]
        )
        
        return self.llm.invoke(sig_prompt.format(
            hypothesis=json.dumps(hypothesis),
            focus=requirements.get('research_focus', ''),
            priorities=requirements.get('priorities', [])
        ))
    
    def _generate_innovation(self, hypothesis: Dict, requirements: Dict) -> str:
        """Generate Innovation section"""
        innovation_prompt = PromptTemplate(
            template="""
            Write the Innovation section highlighting novel aspects.
            
            Hypothesis: {hypothesis}
            Cross-Domain Connections: {cross_domain}
            
            Highlight:
            1. Novel conceptual framework
            2. Innovative methodologies
            3. Cross-disciplinary approaches
            4. Transformative potential
            5. Paradigm shifts
            
            Be specific about what is new and why it matters.
            Length: 500-700 words
            """,
            input_variables=["hypothesis", "cross_domain"]
        )
        
        return self.llm.invoke(innovation_prompt.format(
            hypothesis=json.dumps(hypothesis),
            cross_domain=hypothesis.get('cross_domain_inspiration', '')
        ))
    
    def _generate_approach(
        self,
        hypothesis: Dict,
        implementation: List[Dict],
        requirements: Dict
    ) -> str:
        """Generate Approach section with timeline and milestones"""
        approach_prompt = PromptTemplate(
            template="""
            Write the Approach/Methods section.
            
            Hypothesis: {hypothesis}
            Implementation Plan: {implementation}
            Grant Duration: {duration}
            
            Organize as:
            1. Overview and rationale
            2. Preliminary data (if available)
            3. Detailed methods for each aim
            4. Timeline with milestones
            5. Expected outcomes
            6. Potential problems and alternative approaches
            7. Rigor and reproducibility plans
            
            Be detailed but clear. Length: 2000-3000 words
            Include specific techniques, controls, sample sizes.
            """,
            input_variables=["hypothesis", "implementation", "duration"]
        )
        
        return self.llm.invoke(approach_prompt.format(
            hypothesis=json.dumps(hypothesis),
            implementation=json.dumps(implementation, indent=2),
            duration=requirements.get('duration', '3 years')
        ))
    
    def generate_broader_impacts(
        self,
        hypothesis: Dict,
        grant_type: str
    ) -> str:
        """
        Generate Broader Impacts section (NSF) or similar
        """
        impacts_prompt = PromptTemplate(
            template="""
            Write a Broader Impacts section for an NSF proposal.
            
            Research: {hypothesis}
            
            Address NSF's criteria:
            1. Advancement of discovery and understanding
            2. Training and development
            3. Broadening participation
            4. Enhancing infrastructure
            5. Benefits to society
            
            Include specific activities:
            - Outreach programs
            - Education initiatives
            - Diversity efforts
            - Public engagement
            - Dissemination plans
            
            Be concrete with measurable outcomes.
            Length: 800-1000 words
            """,
            input_variables=["hypothesis"]
        )
        
        return self.llm.invoke(impacts_prompt.format(
            hypothesis=json.dumps(hypothesis, indent=2)
        ))
    
    def format_bibliography(
        self,
        papers: List[Dict],
        style: str = "apa"
    ) -> str:
        """
        Format bibliography in specified citation style
        """
        # Use citation formatting library or LLM
        formatted_refs = []
        
        for paper in papers:
            if style == "apa":
                ref = self._format_apa(paper)
            elif style == "mla":
                ref = self._format_mla(paper)
            else:  # chicago
                ref = self._format_chicago(paper)
            
            formatted_refs.append(ref)
        
        return "\n\n".join(formatted_refs)
    
    def check_compliance(
        self,
        proposal_text: str,
        requirements: Dict
    ) -> Dict:
        """
        Check proposal against requirements
        
        Returns compliance report with issues and suggestions
        """
        checks = {
            "word_count": self._check_word_count(proposal_text, requirements),
            "required_sections": self._check_sections(proposal_text, requirements),
            "keywords": self._check_keywords(proposal_text, requirements),
            "formatting": self._check_formatting(proposal_text, requirements),
            "禁_words": self._check_problematic_language(proposal_text)
        }
        
        compliance_score = sum(c['passed'] for c in checks.values()) / len(checks)
        
        return {
            "compliance_score": compliance_score,
            "checks": checks,
            "issues": [c['issues'] for c in checks.values() if not c['passed']],
            "suggestions": self._generate_suggestions(checks)
        }
    
    def export_proposal(
        self,
        sections: Dict,
        format: str = "docx",
        template: str = None
    ) -> bytes:
        """
        Export formatted proposal in specified format
        
        Supports: docx, pdf, latex
        """
        if format == "docx":
            return self._export_docx(sections, template)
        elif format == "pdf":
            return self._export_pdf(sections, template)
        elif format == "latex":
            return self._export_latex(sections, template)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_docx(self, sections: Dict, template: str) -> bytes:
        """Export to Microsoft Word format"""
        from docx import Document
        from docx.shared import Inches, Pt
        
        doc = Document(template) if template else Document()
        
        # Set formatting
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Times New Roman'
        font.size = Pt(12)
        
        # Add sections
        for section_name, content in sections.items():
            doc.add_heading(section_name.replace('_', ' ').title(), level=1)
            doc.add_paragraph(content)
            doc.add_page_break()
        
        # Save to bytes
        from io import BytesIO
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()

# Usage example
grant_assistant = GrantWritingAssistant(config, llm)

# Analyze RFP
rfp_analysis = grant_assistant.analyze_rfp(rfp_text)

# Match hypothesis to grant
alignment = grant_assistant.match_hypothesis_to_grant(
    hypothesis, rfp_analysis
)

# Generate proposal sections
if alignment['alignment_score'] > 70:
    aims = grant_assistant.generate_specific_aims(
        hypothesis, rfp_analysis
    )
    
    strategy = grant_assistant.generate_research_strategy(
        hypothesis, implementation_plan, rfp_analysis
    )
    
    impacts = grant_assistant.generate_broader_impacts(
        hypothesis, grant_type="NSF"
    )
    
    # Compile proposal
    proposal = {
        "specific_aims": aims,
        **strategy,
        "broader_impacts": impacts,
        "bibliography": grant_assistant.format_bibliography(papers)
    }
    
    # Check compliance
    compliance = grant_assistant.check_compliance(
        proposal['specific_aims'], rfp_analysis
    )
    
    # Export
    if compliance['compliance_score'] > 0.9:
        docx_file = grant_assistant.export_proposal(
            proposal, format="docx"
        )
```

Include templates for different grant types (NIH, NSF, ERC, etc.) and compliance checking.
```

---

## 📊 MONITORING & OBSERVABILITY PHASE

### PROMPT 25: Logging, Metrics & Tracing

```
Create comprehensive monitoring and observability infrastructure.

FILES:
- src/utils/logging_config.py
- src/utils/metrics.py
- src/utils/tracing.py
- monitoring/prometheus.yml
- monitoring/grafana-dashboard.json

REQUIREMENTS:

Implement full observability stack:
1. Structured logging with context
2. Metrics collection (Prometheus)
3. Distributed tracing (OpenTelemetry)
4. Dashboards (Grafana)
5. Alerting rules
6. Log aggregation

STRUCTURED LOGGING:

FILE: src/utils/logging_config.py

```python
import logging
import sys
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler
import contextvars

# Context variables for request tracking
request_id_var = contextvars.ContextVar('request_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)

class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def filter(self, record):
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        return True

def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/hypothesis-engine.log",
    json_logs: bool = True
):
    """
    Configure application logging with:
    - JSON formatting for machine readability
    - Rotating file handler
    - Console output
    - Context injection
    """
    
    # Create logger
    logger = logging.getLogger("hypothesis_engine")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatting
    if json_logs:
        formatter = jsonlogger.JsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(request_id)s %(user_id)s %(message)s',
            rename_fields={"levelname": "level", "asctime": "timestamp"}
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add context filter
    context_filter = ContextFilter()
    console_handler.addFilter(context_filter)
    file_handler.addFilter(context_filter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Usage
logger = setup_logging()
# Additional Prompts Continuation

## Completing PROMPT 25 and Adding Remaining Sections

```python
# Continuation of logging_config.py

# Example usage in application code
from src.utils.logging_config import logger, request_id_var, user_id_var
import uuid

# Set context for request
request_id_var.set(str(uuid.uuid4()))
user_id_var.set("user_123")

# Log with context
logger.info("Processing hypothesis generation request", extra={
    "query": user_query,
    "field": research_field,
    "num_hypotheses": 5
})
```

METRICS COLLECTION:

FILE: src/utils/metrics.py

```python
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Define metrics
hypothesis_requests_total = Counter(
    'hypothesis_requests_total',
    'Total hypothesis generation requests',
    ['field', 'status']
)

hypothesis_generation_duration = Histogram(
    'hypothesis_generation_duration_seconds',
    'Time spent generating hypotheses',
    ['field'],
    buckets=[1, 5, 10, 30, 60, 120, 300]
)

active_requests = Gauge(
    'active_requests',
    'Number of requests currently being processed'
)

papers_in_database = Gauge(
    'papers_in_database_total',
    'Total number of papers in vector database'
)

vector_search_duration = Histogram(
    'vector_search_duration_seconds',
    'Vector database search latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

llm_api_calls_total = Counter(
    'llm_api_calls_total',
    'Total LLM API calls',
    ['provider', 'model', 'status']
)

def track_request_metrics(field: str):
    """Decorator to track request metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active_requests.inc()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                hypothesis_requests_total.labels(field=field, status='success').inc()
                return result
            except Exception as e:
                hypothesis_requests_total.labels(field=field, status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                hypothesis_generation_duration.labels(field=field).observe(duration)
                active_requests.dec()
        
        return wrapper
    return decorator

# Usage
@track_request_metrics(field="biology")
def generate_hypotheses(query, field):
    # Implementation
    pass
```

PROMETHEUS CONFIGURATION:

FILE: monitoring/prometheus.yml

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alert_rules.yml'

scrape_configs:
  - job_name: 'hypothesis-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

ALERT RULES:

FILE: monitoring/alert_rules.yml

```yaml
groups:
  - name: hypothesis_engine_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(hypothesis_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} requests/sec"

      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(hypothesis_generation_duration_seconds_bucket[5m])) > 60
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow hypothesis generation"
          description: "95th percentile response time is {{ $value }}s"

      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes > 4e9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage: {{ $value }} bytes"
```

Include Grafana dashboard JSON and OpenTelemetry tracing setup.
```

---

## 📈 VISUALIZATION & REPORTING PHASE

### PROMPT 26: Advanced Visualizations

```
Create comprehensive visualization and reporting components.

FILES:
- src/visualization/network_graph.py
- src/visualization/timeline.py
- src/visualization/reports.py
- frontend/components/charts.py

NETWORK VISUALIZATION:

FILE: src/visualization/network_graph.py

```python
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict
import numpy as np

class CitationNetworkVisualizer:
    """Create interactive citation network visualizations"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_network(self, papers: List[Dict], citations: List[tuple]):
        """
        Build network graph from papers and citation relationships
        
        Args:
            papers: List of paper dicts with id, title, year, etc.
            citations: List of (citing_id, cited_id) tuples
        """
        # Add nodes
        for paper in papers:
            self.graph.add_node(
                paper['id'],
                title=paper['title'],
                year=paper['year'],
                field=paper.get('field', 'unknown'),
                citations=paper.get('citations_count', 0)
            )
        
        # Add edges
        for citing, cited in citations:
            if citing in self.graph and cited in self.graph:
                self.graph.add_edge(citing, cited)
    
    def create_interactive_plot(
        self,
        layout_algorithm: str = 'spring',
        color_by: str = 'field',
        size_by: str = 'citations'
    ) -> go.Figure:
        """
        Create interactive Plotly visualization
        
        Args:
            layout_algorithm: 'spring', 'circular', 'kamada_kawai'
            color_by: Node attribute for coloring
            size_by: Node attribute for sizing
        """
        # Calculate layout
        if layout_algorithm == 'spring':
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        elif layout_algorithm == 'circular':
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Prepare edge traces
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Text for hover
            node_data = self.graph.nodes[node]
            node_text.append(
                f"<b>{node_data['title']}</b><br>"
                f"Year: {node_data['year']}<br>"
                f"Field: {node_data['field']}<br>"
                f"Citations: {node_data['citations']}"
            )
            
            # Color mapping
            if color_by == 'field':
                field_colors = {
                    'biology': '#FF6B6B',
                    'physics': '#4ECDC4',
                    'chemistry': '#45B7D1',
                    'computer_science': '#FFA07A',
                    'mathematics': '#98D8C8'
                }
                node_color.append(field_colors.get(node_data.get('field'), '#CCCCCC'))
            elif color_by == 'year':
                node_color.append(node_data['year'])
            
            # Size mapping
            if size_by == 'citations':
                node_size.append(max(5, min(50, node_data['citations'] / 10)))
            elif size_by == 'degree':
                node_size.append(10 + self.graph.degree(node) * 2)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True if color_by == 'year' else False,
                colorscale='Viridis' if color_by == 'year' else None,
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title='Year' if color_by == 'year' else '',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Citation Network',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        return fig
    
    def calculate_centrality_metrics(self) -> Dict:
        """Calculate network centrality metrics"""
        return {
            'degree_centrality': nx.degree_centrality(self.graph),
            'betweenness_centrality': nx.betweenness_centrality(self.graph),
            'pagerank': nx.pagerank(self.graph),
            'clustering_coefficient': nx.clustering(self.graph.to_undirected())
        }
    
    def identify_communities(self) -> Dict[int, List]:
        """Detect communities in the network"""
        import community as community_louvain
        
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()
        
        # Detect communities
        partition = community_louvain.best_partition(G_undirected)
        
        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        return communities

# Usage
visualizer = CitationNetworkVisualizer()
visualizer.build_network(papers, citations)
fig = visualizer.create_interactive_plot(color_by='field', size_by='citations')
fig.show()
```

PDF REPORT GENERATION:

FILE: src/visualization/reports.py

```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime

class HypothesisReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E88E5'),
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='HypothesisTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#333333'),
            spaceAfter=12,
            leftIndent=20
        ))
    
    def generate_report(
        self,
        query: str,
        hypotheses: List[Dict],
        papers: List[Dict],
        metadata: Dict,
        output_path: str = None
    ) -> bytes:
        """
        Generate comprehensive hypothesis report
        
        Returns PDF as bytes if output_path is None
        """
        buffer = BytesIO() if output_path is None else output_path
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._create_title_page(query, metadata))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(hypotheses, metadata))
        story.append(PageBreak())
        
        # Detailed hypotheses
        for i, hypo in enumerate(hypotheses, 1):
            story.extend(self._create_hypothesis_section(hypo, i))
            if i < len(hypotheses):
                story.append(PageBreak())
        
        # Supporting papers
        story.append(PageBreak())
        story.extend(self._create_bibliography(papers))
        
        # Build PDF
        doc.build(story)
        
        if output_path is None:
            buffer.seek(0)
            return buffer.read()
    
    def _create_title_page(self, query: str, metadata: Dict) -> List:
        """Create report title page"""
        elements = []
        
        # Title
        title = Paragraph(
            "Scientific Hypothesis Cross-Pollination Engine",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle = Paragraph(
            "Hypothesis Generation Report",
            self.styles['Heading2']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, inch))
        
        # Query
        query_style = ParagraphStyle(
            'QueryStyle',
            parent=self.styles['BodyText'],
            fontSize=12,
            leading=18,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=20,
            borderColor=colors.HexColor('#1E88E5'),
            borderWidth=1,
            borderPadding=10
        )
        query_text = Paragraph(f"<b>Research Question:</b><br/>{query}", query_style)
        elements.append(query_text)
        elements.append(Spacer(1, 0.5*inch))
        
        # Metadata table
        meta_data = [
            ['Generated:', datetime.now().strftime('%B %d, %Y')],
            ['Research Field:', metadata.get('field', 'N/A')],
            ['Papers Analyzed:', str(metadata.get('papers_analyzed', 0))],
            ['Hypotheses Generated:', str(metadata.get('num_hypotheses', 0))]
        ]
        
        meta_table = Table(meta_data, colWidths=[2*inch, 3*inch])
        meta_table.setStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ])
        elements.append(meta_table)
        
        return elements
    
    def _create_hypothesis_section(self, hypothesis: Dict, index: int) -> List:
        """Create detailed section for one hypothesis"""
        elements = []
        
        # Hypothesis title
        title = Paragraph(
            f"Hypothesis {index}: {hypothesis['title']}",
            self.styles['HypothesisTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))
        
        # Scores table
        scores = [
            ['Novelty Score', f"{hypothesis['novelty_score']:.1f}/10"],
            ['Feasibility', f"{hypothesis['feasibility_score']:.1f}/10"],
            ['Impact Potential', f"{hypothesis['impact_score']:.1f}/10"]
        ]
        
        score_table = Table(scores, colWidths=[2*inch, 1.5*inch])
        score_table.setStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E3F2FD')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ])
        elements.append(score_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Description
        desc = Paragraph(
            f"<b>Description:</b> {hypothesis['description']}",
            self.styles['BodyText']
        )
        elements.append(desc)
        elements.append(Spacer(1, 0.2*inch))
        
        # Implementation steps
        elements.append(Paragraph("<b>Implementation Plan:</b>", self.styles['Heading3']))
        for step_num, step in enumerate(hypothesis['implementation_steps'], 1):
            step_text = Paragraph(
                f"{step_num}. {step['action']} "
                f"<i>(Timeline: {step.get('timeline', 'TBD')}, "
                f"Difficulty: {step.get('difficulty', 'Medium')})</i>",
                self.styles['BodyText']
            )
            elements.append(step_text)
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_bibliography(self, papers: List[Dict]) -> List:
        """Create formatted bibliography"""
        elements = []
        
        elements.append(Paragraph("References", self.styles['Heading1']))
        elements.append(Spacer(1, 0.3*inch))
        
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper.get('authors', [])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += " et al."
            
            citation = (
                f"{i}. {authors} ({paper.get('year', 'n.d.')}). "
                f"{paper.get('title', 'Untitled')}. "
                f"<i>{paper.get('venue', 'Unknown venue')}</i>. "
            )
            
            if paper.get('doi'):
                citation += f"https://doi.org/{paper['doi']}"
            
            ref = Paragraph(citation, self.styles['BodyText'])
            elements.append(ref)
            elements.append(Spacer(1, 0.15*inch))
        
        return elements
```

Include LaTeX template generation for grant proposals and timeline visualizations.
```

---

## 🔒 SECURITY & PRIVACY PHASE

### PROMPT 27: Authentication, Authorization & Data Protection

```
Implement comprehensive security measures for production deployment.

FILES:
- src/security/auth.py
- src/security/encryption.py
- src/security/validators.py
- src/middleware/security_middleware.py

AUTHENTICATION SYSTEM:

FILE: src/security/auth.py

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import secrets

# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # Store in environment/secrets
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

class AuthManager:
    """Handle authentication and authorization"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

# FastAPI dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to get current authenticated user from JWT token
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    auth_manager = AuthManager(SECRET_KEY)
    payload = auth_manager.verify_token(credentials.credentials)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return payload

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/hypotheses/generate")
@limiter.limit("5/minute")  # Max 5 requests per minute
async def generate_hypotheses(
    request: Request,
    query: HypothesisQuery,
    user: dict = Depends(get_current_user)
):
    """
    Protected endpoint with rate limiting
    """
    # Implementation
    pass
```

INPUT VALIDATION:

FILE: src/security/validators.py

```python
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import re
import bleach

class SecureHypothesisQuery(BaseModel):
    """Validated and sanitized query model"""
    
    query: str = Field(..., min_length=10, max_length=5000)
    field: str = Field(..., regex=r'^[a-z_]+$')
    year_from: Optional[int] = Field(None, ge=1900, le=2100)
    year_to: Optional[int] = Field(None, ge=1900, le=2100)
    num_hypotheses: int = Field(5, ge=1, le=15)
    
    @validator('query')
    def sanitize_query(cls, v):
        """Remove potentially dangerous content"""
        # Remove HTML tags
        clean_query = bleach.clean(v, tags=[], strip=True)
        
        # Remove SQL injection attempts
        sql_patterns = [
            r'(\bUNION\b.*\bSELECT\b)',
            r'(\bDROP\b.*\bTABLE\b)',
            r'(\-\-)',
            r'(;.*\bSELECT\b)'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, clean_query, re.IGNORECASE):
                raise ValueError("Invalid query content detected")
        
        # Remove script tags and javascript
        script_pattern = r'<script[\s\S]*?</script>'
        clean_query = re.sub(script_pattern, '', clean_query, flags=re.IGNORECASE)
        
        return clean_query.strip()
    
    @validator('year_to')
    def validate_year_range(cls, v, values):
        """Ensure year_to >= year_from"""
        if 'year_from' in values and v is not None:
            if v < values['year_from']:
                raise ValueError("year_to must be >= year_from")
        return v

class SecureUserRegistration(BaseModel):
    """Secure user registration model"""
    
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=12)
    full_name: str = Field(..., max_length=100)
    
    @validator('password')
    def validate_password_strength(cls, v):
        """Ensure strong password"""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain digit")
        if not any(c in '!@#$%^&*()_+-=' for c in v):
            raise ValueError("Password must contain special character")
        return v
```

DATA ENCRYPTION:

FILE: src/security/encryption.py

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import os

class DataEncryption:
    """Handle sensitive data encryption"""
    
    def __init__(self, password: bytes = None):
        if password is None:
            password = os.environ.get('ENCRYPTION_KEY', '').encode()
        
        # Derive key from password
        salt = b'hypothesis_engine_salt'  # Store securely
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()

# Usage for API keys storage
encryptor = DataEncryption()

# Encrypt before storing
encrypted_key = encryptor.encrypt(user_api_key)
store_in_database(user_id, encrypted_key)

# Decrypt when needed
stored_key = get_from_database(user_id)
api_key = encryptor.decrypt(stored_key)
```

GDPR COMPLIANCE:

```python
class GDPRCompliance:
    """Handle GDPR requirements"""
    
    async def export_user_data(self, user_id: str) -> dict:
        """Export all user data (GDPR Right to Data Portability)"""
        return {
            "user_profile": await get_user_profile(user_id),
            "query_history": await get_user_queries(user_id),
            "saved_hypotheses": await get_saved_hypotheses(user_id),
            "preferences": await get_user_preferences(user_id)
        }
    
    async def delete_user_data(self, user_id: str):
        """Completely delete user data (GDPR Right to Erasure)"""
        await delete_user_queries(user_id)
        await delete_saved_hypotheses(user_id)
        await anonymize_logs(user_id)
        await delete_user_account(user_id)
    
    async def get_consent(self, user_id: str) -> dict:
        """Track user consent for data processing"""
        return await get_user_consents(user_id)
```

Include CORS configuration, CSRF protection, and security headers.
```

This file continues with remaining sections. Would you like me to create the final sections (Performance Optimization, Advanced Features, Documentation) in a third file?
# Final Additional Prompts (28-31)

## ⚡ PERFORMANCE OPTIMIZATION PHASE

### PROMPT 28: Caching, Async Processing & GPU Acceleration

```
Implement performance optimizations for production scale.

FILES:
- src/performance/caching.py
- src/performance/async_processor.py
- src/performance/gpu_optimizer.py

REDIS CACHING:

FILE: src/performance/caching.py

```python
import redis
import json
import hashlib
from functools import wraps
from typing import Any, Callable
import pickle

class CacheManager:
    """Redis-based caching for expensive operations"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function and arguments"""
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Any:
        """Get cached value"""
        value = self.redis_client.get(key)
        if value:
            return pickle.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set cached value"""
        ttl = ttl or self.default_ttl
        serialized = pickle.dumps(value)
        self.redis_client.setex(key, ttl, serialized)
    
    def cache(self, ttl: int = None):
        """Decorator for caching function results"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.cache_key(func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    logger.info(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result, ttl)
                logger.info(f"Cached result for {func.__name__}")
                
                return result
            return wrapper
        return decorator

# Usage
cache_manager = CacheManager()

@cache_manager.cache(ttl=7200)  # Cache for 2 hours
def search_papers_expensive(query: str, field: str):
    """Expensive vector search operation"""
    # ... implementation
    pass
```

ASYNC PROCESSING:

FILE: src/performance/async_processor.py

```python
import asyncio
from typing import List, Dict, Callable
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class AsyncHypothesisGenerator:
    """Async implementation for parallel processing"""
    
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def process_multiple_queries(
        self,
        queries: List[str]
    ) -> List[Dict]:
        """Process multiple queries concurrently"""
        tasks = [
            self.generate_hypotheses_async(query)
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def generate_hypotheses_async(self, query: str) -> Dict:
        """Async hypothesis generation"""
        async with self.semaphore:
            # Step 1: Primary search (parallel with cross-domain)
            primary_task = self.search_primary_async(query)
            cross_domain_task = self.search_cross_domain_async(query)
            
            primary_results, cross_domain_results = await asyncio.gather(
                primary_task,
                cross_domain_task
            )
            
            # Step 2: Generate hypotheses
            hypotheses = await self.synthesize_hypotheses_async(
                query,
                primary_results,
                cross_domain_results
            )
            
            return hypotheses
    
    async def search_primary_async(self, query: str) -> List[Dict]:
        """Async vector search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._search_vector_db,
            query
        )
    
    async def fetch_multiple_papers_async(
        self,
        paper_ids: List[str]
    ) -> List[Dict]:
        """Fetch multiple papers concurrently"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_paper(session, paper_id)
                for paper_id in paper_ids
            ]
            return await asyncio.gather(*tasks)
    
    async def _fetch_paper(
        self,
        session: aiohttp.ClientSession,
        paper_id: str
    ) -> Dict:
        """Fetch single paper with async HTTP"""
        url = f"https://api.semanticscholar.org/v1/paper/{paper_id}"
        async with session.get(url) as response:
            return await response.json()
```

GPU ACCELERATION:

FILE: src/performance/gpu_optimizer.py

```python
import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class GPUEmbedder:
    """GPU-accelerated embedding generation"""
    
    def __init__(self, model_name: str = "allenai-specter"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name).to(self.device)
        
        logger.info(f"Loaded embedding model on {self.device}")
        
        # Enable mixed precision for speed
        if self.device == 'cuda':
            self.model.half()  # Use FP16
    
    def embed_batch_gpu(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings using GPU with batching
        
        ~10x faster than CPU for large batches
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def parallel_search(
        self,
        query_embeddings: np.ndarray,
        db_embeddings: np.ndarray,
        top_k: int = 10
    ) -> tuple:
        """
        Parallel similarity search on GPU
        
        Args:
            query_embeddings: (n_queries, embedding_dim)
            db_embeddings: (n_docs, embedding_dim)
            top_k: Number of results per query
        
        Returns:
            indices, distances for top_k results per query
        """
        # Convert to PyTorch tensors
        query_tensor = torch.from_numpy(query_embeddings).to(self.device)
        db_tensor = torch.from_numpy(db_embeddings).to(self.device)
        
        # Compute cosine similarity (batch matrix multiplication)
        similarities = torch.mm(query_tensor, db_tensor.T)
        
        # Get top-k
        top_similarities, top_indices = torch.topk(
            similarities,
            k=min(top_k, similarities.shape[1]),
            dim=1
        )
        
        return (
            top_indices.cpu().numpy(),
            top_similarities.cpu().numpy()
        )

# Memory optimization
def optimize_memory():
    """Clear GPU cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

Include query result caching, database connection pooling, and precomputed embeddings.
```

---

## 🌟 ADVANCED FEATURES PHASE

### PROMPT 29: Personalization & Integrations

```
Implement advanced features for enhanced user experience.

FILES:
- src/features/personalization.py
- src/features/integrations.py
- src/features/collaboration.py

PERSONALIZATION ENGINE:

FILE: src/features/personalization.py

```python
from typing import List, Dict
import numpy as np
from collections import defaultdict

class PersonalizationEngine:
    """Personalize recommendations based on user history"""
    
    def __init__(self):
        self.user_profiles = {}
    
    def build_user_profile(self, user_id: str) -> Dict:
        """
        Build user profile from interaction history
        
        Tracks:
        - Research interests (fields, keywords)
        - Paper preferences (citation threshold, recency)
        - Hypothesis preferences (novelty vs feasibility)
        - Successful outcomes
        """
        history = self._get_user_history(user_id)
        
        profile = {
            'preferred_fields': self._extract_field_preferences(history),
            'keyword_interests': self._extract_keyword_interests(history),
            'citation_threshold': self._calculate_citation_preference(history),
            'novelty_preference': self._calculate_novelty_preference(history),
            'successful_hypotheses': self._get_successful_hypotheses(user_id)
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def personalize_search_results(
        self,
        results: List[Dict],
        user_id: str
    ) -> List[Dict]:
        """Rerank search results based on user preferences"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return results
        
        # Calculate personalized scores
        for result in results:
            base_score = result.get('relevance_score', 0)
            
            # Field preference boost
            field_boost = 1.0
            if result['field'] in profile['preferred_fields']:
                field_boost = 1.3
            
            # Citation preference
            cit_boost = 1.0
            if result['citations'] >= profile['citation_threshold']:
                cit_boost = 1.2
            
            # Keyword match boost
            keyword_boost = 1.0
            if any(kw in result.get('keywords', []) 
                   for kw in profile['keyword_interests']):
                keyword_boost = 1.4
            
            result['personalized_score'] = (
                base_score * field_boost * cit_boost * keyword_boost
            )
        
        # Resort by personalized score
        return sorted(
            results,
            key=lambda x: x['personalized_score'],
            reverse=True
        )
    
    def recommend_related_queries(
        self,
        current_query: str,
        user_id: str,
        n_recommendations: int = 5
    ) -> List[str]:
        """Suggest related queries based on user history"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return []
        
        # Generate recommendations based on:
        # 1. Similar past successful queries
        # 2. Trending queries in user's fields
        # 3. Queries from similar users
        
        recommendations = []
        
        # Similar past queries
        past_queries = self._get_user_past_queries(user_id)
        query_embedding = self.embedder.embed_text(current_query)
        
        for past_query in past_queries:
            past_embedding = self.embedder.embed_text(past_query['text'])
            similarity = cosine_similarity(query_embedding, past_embedding)
            
            if 0.6 < similarity < 0.95:  # Similar but not identical
                recommendations.append({
                    'query': past_query['text'],
                    'reason': 'Similar to your past successful query',
                    'score': similarity
                })
        
        # Field-based recommendations
        for field in profile['preferred_fields']:
            trending = self._get_trending_queries(field, limit=3)
            recommendations.extend([{
                'query': q,
                'reason': f'Trending in {field}',
                'score': 0.7
            } for q in trending])
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]

THIRD-PARTY INTEGRATIONS:

FILE: src/features/integrations.py

```python
import requests
from typing import Dict, List

class ReferenceManagerIntegration:
    """Integration with Zotero, Mendeley, EndNote"""
    
    def export_to_zotero(
        self,
        papers: List[Dict],
        api_key: str,
        collection_id: str
    ) -> Dict:
        """
        Export papers to Zotero collection
        
        Args:
            papers: List of paper metadata
            api_key: User's Zotero API key
            collection_id: Target Zotero collection
        """
        headers = {
            'Zotero-API-Key': api_key,
            'Content-Type': 'application/json'
        }
        
        items = []
        for paper in papers:
            item = {
                'itemType': 'journalArticle',
                'title': paper['title'],
                'creators': [
                    {'creatorType': 'author', 'name': author}
                    for author in paper.get('authors', [])
                ],
                'abstractNote': paper.get('abstract', ''),
                'publicationTitle': paper.get('venue', ''),
                'date': str(paper.get('year', '')),
                'DOI': paper.get('doi', ''),
                'url': paper.get('url', ''),
                'tags': [{'tag': kw} for kw in paper.get('keywords', [])]
            }
            items.append(item)
        
        # Send to Zotero
        response = requests.post(
            f'https://api.zotero.org/users/{user_id}/collections/{collection_id}/items',
            headers=headers,
            json=items
        )
        
        return response.json()
    
    def import_from_mendeley(
        self,
        access_token: str,
        folder_id: str = None
    ) -> List[Dict]:
        """Import papers from Mendeley"""
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
        
        url = 'https://api.mendeley.com/documents'
        if folder_id:
            url += f'?folder_id={folder_id}'
        
        response = requests.get(url, headers=headers)
        mendeley_docs = response.json()
        
        # Convert to our format
        papers = []
        for doc in mendeley_docs:
            papers.append({
                'title': doc.get('title'),
                'authors': [a.get('last_name') for a in doc.get('authors', [])],
                'year': doc.get('year'),
                'abstract': doc.get('abstract'),
                'doi': doc.get('identifiers', {}).get('doi'),
                'keywords': doc.get('keywords', [])
            })
        
        return papers

class SlackNotifications:
    """Send notifications to Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def notify_hypotheses_ready(self, user_name: str, num_hypotheses: int):
        """Notify user when hypotheses are generated"""
        message = {
            "text": f"🎉 Hypothesis generation complete for {user_name}!",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{num_hypotheses} new hypotheses* are ready for review."
                    }
                },
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "View Hypotheses"},
                            "url": "https://hypothesis-engine.com/results"
                        }
                    ]
                }
            ]
        }
        
        requests.post(self.webhook_url, json=message)

COLLABORATIVE FEATURES:

FILE: src/features/collaboration.py

```python
class CollaborationManager:
    """Enable team collaboration on hypotheses"""
    
    async def share_hypothesis(
        self,
        hypothesis_id: str,
        owner_id: str,
        collaborator_emails: List[str],
        permissions: str = "view"
    ):
        """
        Share hypothesis with collaborators
        
        Args:
            permissions: 'view', 'comment', 'edit'
        """
        for email in collaborator_emails:
            # Create share link
            share_token = self._create_share_token(hypothesis_id, email, permissions)
            share_link = f"https://hypothesis-engine.com/shared/{share_token}"
            
            # Send email invitation
            await self._send_collaboration_invite(
                email,
                owner_id,
                share_link,
                permissions
            )
            
            # Record sharing
            await self._record_share(
                hypothesis_id,
                owner_id,
                email,
                permissions
            )
    
    async def add_comment(
        self,
        hypothesis_id: str,
        user_id: str,
        comment_text: str,
        parent_comment_id: str = None
    ) -> Dict:
        """Add comment to hypothesis (supports threading)"""
        comment = {
            'id': generate_id(),
            'hypothesis_id': hypothesis_id,
            'user_id': user_id,
            'text': comment_text,
            'parent_id': parent_comment_id,
            'created_at': datetime.utcnow(),
            'reactions': []
        }
        
        await store_comment(comment)
        
        # Notify hypothesis owner and collaborators
        await self._notify_new_comment(hypothesis_id, comment)
        
        return comment
    
    async def track_hypothesis_evolution(
        self,
        hypothesis_id: str
    ) -> List[Dict]:
        """Track how hypothesis was refined over time"""
        return await get_hypothesis_version_history(hypothesis_id)
```

Include webhook support for external integrations and real-time collaboration features.
```

---

## 📖 DOCUMENTATION PHASE

### PROMPT 30: API Docs & User Guides

```
Create comprehensive documentation for users and developers.

FILES:
- docs/api/openapi.yml
- docs/user_guide.md
- docs/developer_guide.md
- docs/deployment_guide.md
- README.md (comprehensive)

API DOCUMENTATION:

FILE: docs/api/openapi.yml

```yaml
openapi: 3.0.0
info:
  title: Scientific Hypothesis Cross-Pollination Engine API
  version: 1.0.0
  description: |
    REST API for generating novel research hypotheses through cross-disciplinary analysis.
    
    Features:
    - Multi-agent RAG-based hypothesis generation
    - Cross-domain paper discovery
    - Grant proposal assistance
    - Citation network analysis
  contact:
    email: support@hypothesis-engine.com
  license:
    name: MIT

servers:
  - url: https://api.hypothesis-engine.com/v1
    description: Production server
  - url: https://staging-api.hypothesis-engine.com/v1
    description: Staging server

paths:
  /hypotheses/generate:
    post:
      summary: Generate research hypotheses
      description: |
        Generate novel research hypotheses based on a research question.
        
        This endpoint uses multiple AI agents to:
        1. Search your research field
        2. Find cross-domain connections
        3. Generate testable hypotheses
        4. Provide implementation plans
      tags:
        - Hypotheses
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/HypothesisRequest'
            example:
              query: "Novel approaches to cancer cell migration tracking?"
              field: "biology"
              num_hypotheses: 5
              year_from: 2020
              creativity: 0.7
      responses:
        '202':
          description: Request accepted, processing started
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    example: "job_abc123"
                  status:
                    type: string
                    example: "processing"
                  estimated_time:
                    type: integer
                    example: 120
        '400':
          description: Invalid request
        '401':
          description: Unauthorized
        '429':
          description: Rate limit exceeded

  /hypotheses/status/{job_id}:
    get:
      summary: Check hypothesis generation status
      tags:
        - Hypotheses
      security:
        - bearerAuth: []
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Status information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobStatus'

  /hypotheses/results/{job_id}:
    get:
      summary: Get generated hypotheses
      tags:
        - Hypotheses
      security:
        - bearerAuth: []
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Generated hypotheses
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HypothesisResults'

components:
  schemas:
    HypothesisRequest:
      type: object
      required:
        - query
        - field
      properties:
        query:
          type: string
          minLength: 10
          maxLength: 5000
        field:
          type: string
          enum: [biology, physics, chemistry, computer_science, mathematics, engineering]
        num_hypotheses:
          type: integer
          minimum: 1
          maximum: 15
          default: 5
        year_from:
          type: integer
          minimum: 1900
        year_to:
          type: integer
        creativity:
          type: number
          minimum: 0
          maximum: 1
          default: 0.7
    
    JobStatus:
      type: object
      properties:
        job_id:
          type: string
        status:
          type: string
          enum: [queued, processing, complete, failed]
        progress:
          type: integer
          minimum: 0
          maximum: 100
        current_step:
          type: string
    
    HypothesisResults:
      type: object
      properties:
        job_id:
          type: string
        query:
          type: string
        hypotheses:
          type: array
          items:
            $ref: '#/components/schemas/Hypothesis'
        metadata:
          type: object
    
    Hypothesis:
      type: object
      properties:
        id:
          type: string
        title:
          type: string
        description:
          type: string
        novelty_score:
          type: number
          minimum: 0
          maximum: 10
        feasibility_score:
          type: number
        impact_score:
          type: number
        implementation_steps:
          type: array
          items:
            type: object
        supporting_papers:
          type: array
          items:
            $ref: '#/components/schemas/Paper'
    
    Paper:
      type: object
      properties:
        id:
          type: string
        title:
          type: string
        authors:
          type: array
          items:
            type: string
        year:
          type: integer
        abstract:
          type: string
        doi:
          type: string
        url:
          type: string

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

USER GUIDE:

FILE: docs/user_guide.md

```markdown
# User Guide: Scientific Hypothesis Cross-Pollination Engine

## Table of Contents
1. [Getting Started](#getting-started)
2. [Creating Your First Query](#first-query)
3. [Understanding Results](#understanding-results)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### What is the Hypothesis Engine?

The Scientific Hypothesis Cross-Pollination Engine helps researchers discover novel research directions by finding unexpected connections between different scientific fields.

**Key Benefits:**
- 🔍 Analyze millions of papers across all disciplines
- 🧠 Identify cross-domain methodologies
- 💡 Generate testable research hypotheses
- 📊 Get evidence-based recommendations
- 📝 Export to grant proposals

### Setting Up Your Account

1. Visit [hypothesis-engine.com](https://hypothesis-engine.com)
2. Click "Sign Up" and create an account
3. Verify your email address
4. Complete your research profile

### Your Research Profile

Fill out your profile to get personalized recommendations:
- Primary research field
- Current research topics
- Expertise level
- Institution
- Research goals

## Creating Your First Query {#first-query}

### Step 1: Describe Your Research Problem

Be specific and detailed:

**Good Example:**
> "I'm studying how cancer cells migrate through blood vessels using live microscopy. Current 2D imaging techniques have limited depth. Are there 3D tracking methods from other fields that could be adapted?"

**Poor Example:**
> "cancer research"

### Step 2: Configure Search Parameters

- **Research Field**: Select your primary field
- **Date Range**: Focus on recent papers (default: 2020-2024)
- **Number of Hypotheses**: 3-10 (we recommend 5)
- **Creativity Level**: 0.5-0.9 (higher = more novel but riskier)

### Step 3: Review Progress

Watch real-time progress as the system:
1. Searches your field (20%)
2. Searches other fields (40%)
3. Analyzes methodologies (60%)
4. Generates hypotheses (80%)
5. Validates and ranks (100%)

## Understanding Results {#understanding-results}

### Hypothesis Cards

Each hypothesis includes:

**Scores (0-10 scale):**
- **Novelty**: How original is this idea?
- **Feasibility**: Can you actually do this?
- **Impact**: How significant could the results be?

**Sections:**
- **Description**: What the hypothesis proposes
- **Supporting Evidence**: Papers that inspired this
- **Implementation Plan**: Step-by-step guide
- **Challenges**: Potential obstacles and solutions
- **Required Resources**: What you'll need

### Example Hypothesis

```
💡 Hypothesis 3: Apply Particle Image Velocimetry to Cell Tracking

Novelty: 8.5/10  |  Feasibility: 7.2/10  |  Impact: 8.8/10

Description:
Adapt particle image velocimetry (PIV) from fluid dynamics 
to track cancer cells in 3D tissue models...

[See full details in app]
```

## Advanced Features {#advanced-features}

### Grant Writing Assistant

1. Select promising hypotheses
2. Click "Generate Grant Proposal"
3. Choose grant type (NSF, NIH, ERC, etc.)
4. Review and edit generated sections:
   - Specific Aims
   - Significance
   - Innovation
   - Approach
   - Broader Impacts

### Collaborative Features

**Share Hypotheses:**
- Click "Share" on any hypothesis
- Enter collaborator emails
- Set permissions (view/comment/edit)

**Comments:**
- Discuss ideas with team
- Thread conversations
- @ mention collaborators

### Reference Manager Integration

Export papers to:
- Zotero
- Mendeley
- EndNote
- BibTeX

## Best Practices {#best-practices}

### Writing Effective Queries

1. **Be Specific**: Include your exact problem
2. **Mention Limitations**: What doesn't work currently?
3. **Add Context**: Equipment, expertise, resources
4. **Ask Questions**: Frame as a question

### Evaluating Hypotheses

Consider:
- ✅ Do you have access to required resources?
- ✅ Does it align with your expertise?
- ✅ Is the timeline realistic?
- ✅ Can you validate the results?

### Working with Results

1. **Start Small**: Test one hypothesis first
2. **Pilot Studies**: Do preliminary experiments
3. **Iterate**: Refine based on results
4. **Document**: Track what works

## Troubleshooting {#troubleshooting}

### Common Issues

**"No hypotheses generated"**
- Make query more specific
- Adjust creativity level
- Try different field combinations

**"Results not relevant"**
- Review your research profile
- Add more context to query
- Specify constraints/requirements

**"Processing takes too long"**
- Normal processing: 2-5 minutes
- Complex queries: up to 10 minutes
- Contact support if > 15 minutes

### Getting Help

- 📧 Email: support@hypothesis-engine.com
- 💬 Chat: Available in app
- 📚 Documentation: docs.hypothesis-engine.com
- 🎥 Video tutorials: Available in Help Center
```

Include developer guide with code examples, deployment guide with infrastructure setup, and troubleshooting guide.
```

---

## END OF ADDITIONAL SECTIONS

All remaining prompts (23-30) have been created with comprehensive implementation details. Ready to merge into main DetailedPrompt.md file.
