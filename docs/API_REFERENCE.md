# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

API key required in header:
```
Authorization: Bearer YOUR_API_KEY
```

---

## Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "operational",
  "version": "1.0.0",
  "timestamp": "2024-12-08T12:00:00Z"
}
```

---

### Generate Hypotheses

```
POST /generate-hypotheses
```

Generate novel research hypotheses from a research question.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| research_question | string | Yes | The research question (min 10 chars) |
| field | string | No | Primary research field |
| year_from | int | No | Minimum publication year |
| year_to | int | No | Maximum publication year |
| max_hypotheses | int | No | Max hypotheses (1-50, default 10) |

**Example Request:**

```json
{
  "research_question": "How can machine learning improve early cancer detection?",
  "field": "biology",
  "year_from": 2020,
  "year_to": 2024,
  "max_hypotheses": 5
}
```

**Response:**

```json
{
  "request_id": "uuid-string",
  "query": "How can machine learning...",
  "num_hypotheses": 5,
  "hypotheses": [
    {
      "id": "hyp_1",
      "type": "methodology_transfer",
      "title": "Apply NLP transformers to genomic sequence analysis",
      "description": "Transfer transformer architecture...",
      "novelty_score": 0.85,
      "feasibility_score": 0.72,
      "impact_potential": 0.88,
      "composite_score": 0.82,
      "rank": 1,
      "validation": {
        "overall_score": 0.78,
        "recommendation": "Highly recommended for pursuit"
      }
    }
  ],
  "execution_time": 45.2,
  "created_at": "2024-12-08T12:00:00Z"
}
```

---

### Search Papers

```
POST /search
```

Search for papers using semantic similarity.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Search query |
| field | string | No | Filter by field |
| top_k | int | No | Number of results (1-100, default 10) |

**Example Request:**

```json
{
  "query": "deep learning for protein structure prediction",
  "field": "biology",
  "top_k": 10
}
```

**Response:**

```json
{
  "query": "deep learning for protein structure prediction",
  "num_results": 10,
  "results": [
    {
      "title": "AlphaFold: Improved protein structure prediction",
      "authors": ["..."],
      "year": 2021,
      "abstract": "...",
      "field": "biology",
      "citations": 5000,
      "doi": "10.1038/...",
      "relevance_score": 0.95
    }
  ]
}
```

---

### Get Paper Details

```
GET /papers/{paper_id}
```

**Response:**

```json
{
  "paper_id": "arxiv_2024_12345",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "year": 2024,
  "abstract": "Full abstract...",
  "field": "computer_science",
  "subfield": "machine_learning",
  "citations": 42,
  "references": ["ref_1", "ref_2"],
  "doi": "10.xxxx/...",
  "url": "https://..."
}
```

---

### Get Available Fields

```
GET /fields
```

**Response:**

```json
{
  "fields": [
    "biology",
    "physics", 
    "computer_science",
    "chemistry",
    "mathematics",
    "engineering",
    "medicine"
  ]
}
```

---

### System Status

```
GET /status
```

**Response:**

```json
{
  "status": "operational",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "database": "healthy",
    "chroma": "healthy",
    "llm": "healthy"
  },
  "metrics": {
    "total_papers": 1000000,
    "hypotheses_generated": 5432,
    "uptime": "24h 30m"
  },
  "timestamp": "2024-12-08T12:00:00Z"
}
```

---

## WebSocket API

### Connection

```
ws://localhost:8000/ws
```

### Progress Updates

Receive real-time progress during hypothesis generation:

```json
{
  "type": "progress",
  "data": {
    "step": "Analyzing research question",
    "progress": 0.2,
    "details": "Identifying primary field..."
  },
  "timestamp": "2024-12-08T12:00:00Z"
}
```

### Result

Final result when complete:

```json
{
  "type": "result",
  "data": {
    "hypotheses": [...]
  },
  "timestamp": "2024-12-08T12:00:00Z"
}
```

---

## Error Responses

All errors return:

```json
{
  "detail": "Error message",
  "status_code": 400
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request / validation error |
| 401 | Unauthorized / invalid API key |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| /generate-hypotheses | 10 req/min |
| /search | 60 req/min |
| /papers/* | 100 req/min |

---

## SDKs

### Python

```python
from hypothesis_engine import Client

client = Client(api_key="your-key")

# Generate hypotheses
result = client.generate_hypotheses(
    "Your research question",
    field="biology"
)

for h in result.hypotheses:
    print(h.title, h.novelty_score)
```

---

*API Version: 1.0.0*
*Last Updated: December 2024*
