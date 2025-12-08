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
