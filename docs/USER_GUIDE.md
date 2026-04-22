# User Guide: Scientific Hypothesis Cross-Pollination Engine

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Using the Web Interface](#using-the-web-interface)
5. [Using the API](#using-the-api)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

The Scientific Hypothesis Cross-Pollination Engine is an AI-powered research tool that helps scientists discover novel research directions by finding unexpected connections across different scientific fields.

### What It Does

- **Analyzes** your research question to understand your domain
- **Searches** millions of scientific papers semantically
- **Discovers** cross-domain connections and analogies
- **Generates** novel, testable research hypotheses
- **Validates** hypotheses for novelty and feasibility

---

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key (or Ollama for local LLM)
- PostgreSQL database
- ChromaDB instance

### Quick Setup

```bash
# Clone and install
cd hypothesis-engine
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up -d

# Run API
uvicorn src.api.main:app --reload

# Run frontend (new terminal)
streamlit run frontend/app.py
```

### Access

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

---

## Core Concepts

### Research Question

Your starting point. The more specific, the better:

**Good**: "How can machine learning techniques used in natural language processing be applied to predict protein-protein interactions?"

**Less Effective**: "How can ML help biology?"

### Primary Domain Analysis

The system first understands your field:
- Identifies your research domain
- Finds current state-of-the-art approaches
- Identifies knowledge gaps
- Notes what has been tried

### Cross-Domain Discovery

The core innovation:
- Abstracts your problem to find analogies
- Searches completely different fields
- Finds transferable methodologies
- Scores potential connections

### Hypothesis Generation

Three types of hypotheses:
1. **Gap-Filling**: Addresses identified knowledge gaps
2. **Methodology Transfer**: Applies methods from other fields
3. **Cross-Domain Analogy**: Leverages structural similarities

---

## Using the Web Interface

### Step 1: Enter Your Research Question

Navigate to the "Generate Hypotheses" tab and enter your question:

```
How can we improve early detection of pancreatic cancer using imaging techniques?
```

### Step 2: Configure Settings

In the sidebar:
- **Primary Field**: Select or auto-detect
- **Year Range**: Focus on recent research
- **Max Hypotheses**: Number to generate (1-20)

### Step 3: Generate

Click "🚀 Generate Hypotheses" and wait. The process takes 1-3 minutes.

### Step 4: Review Results

Each hypothesis shows:
- **Title**: Brief summary
- **Type**: Gap-filling, transfer, or analogy
- **Scores**: Novelty, feasibility, impact (0-1 scale)
- **Description**: Detailed explanation
- **Validation**: Recommendation and resource estimates

### Step 5: Export

Download results as JSON for further analysis.

---

## Using the API

### Authentication

```bash
export API_KEY="your-api-key"
```

### Generate Hypotheses

```python
import requests

response = requests.post(
    "http://localhost:8000/generate-hypotheses",
    json={
        "research_question": "Your question here",
        "field": "biology",
        "year_from": 2020,
        "year_to": 2024,
        "max_hypotheses": 5
    },
    headers={"Authorization": f"Bearer {API_KEY}"}
)

data = response.json()
for hyp in data['hypotheses']:
    print(f"{hyp['title']} (Novelty: {hyp['novelty_score']:.2f})")
```

### Search Papers

```python
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "deep learning protein folding",
        "field": "biology",
        "top_k": 10
    }
)

for paper in response.json()['results']:
    print(f"- {paper['title']} ({paper['year']})")
```

---

## Interpreting Results

### Novelty Score (0-1)

- **0.9-1.0**: Highly novel, rarely explored
- **0.7-0.9**: Novel with some related work
- **0.5-0.7**: Moderately novel
- **Below 0.5**: Incremental improvement

### Feasibility Score (0-1)

- **0.8-1.0**: Highly feasible with current technology
- **0.6-0.8**: Feasible with moderate effort
- **0.4-0.6**: Challenging but possible
- **Below 0.4**: Significant barriers exist

### Impact Potential (0-1)

- **0.8-1.0**: Potentially transformative
- **0.6-0.8**: High impact
- **0.4-0.6**: Moderate impact
- **Below 0.4**: Incremental impact

### Composite Score

Weighted combination: `0.4*novelty + 0.3*feasibility + 0.3*impact`

---

## Best Practices

### Crafting Good Questions

✅ **Do**:
- Be specific about the problem
- Include domain context
- Mention constraints if any

❌ **Don't**:
- Ask overly broad questions
- Use too much jargon (system may misinterpret)
- Include multiple unrelated questions

### Iteration Strategy

1. Start broad to explore the space
2. Refine based on interesting hypotheses
3. Re-run with more specific questions
4. Combine insights from multiple runs

### Validation

Always:
- Check cited papers actually exist
- Verify cross-domain analogies make sense
- Consult domain experts before pursuing

---

## Troubleshooting

### "No results found"

- Try broader search terms
- Check year range isn't too restrictive
- Verify database is populated

### "Generation taking too long"

- Reduce max_hypotheses
- Narrow the field filter
- Check API rate limits

### "Low quality hypotheses"

- Make question more specific
- Ensure enough papers in database
- Try different wording

### API Errors

```bash
# Check service status
curl http://localhost:8000/health

# Check logs
docker-compose logs api
```

---

## Getting Help

- **Documentation**: See `/docs` directory
- **API Reference**: http://localhost:8000/docs
- **Issues**: GitHub Issues

---

*Last Updated: December 2024*
