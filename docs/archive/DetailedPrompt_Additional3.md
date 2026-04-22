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
