"""
Deployment Configuration for Kubernetes

Kubernetes manifests for production deployment.
"""

# deployment.yaml content as Python dict for template generation
DEPLOYMENT_CONFIG = {
    'api': {
        'replicas': 3,
        'image': 'hypothesis-engine-api:latest',
        'port': 8000,
        'resources': {
            'requests': {'memory': '512Mi', 'cpu': '500m'},
            'limits': {'memory': '2Gi', 'cpu': '2000m'}
        }
    },
    'worker': {
        'replicas': 2,
        'image': 'hypothesis-engine-worker:latest',
        'resources': {
            'requests': {'memory': '1Gi', 'cpu': '1000m'},
            'limits': {'memory': '4Gi', 'cpu': '4000m'}
        }
    }
}

KUBERNETES_DEPLOYMENT = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hypothesis-engine-api
  labels:
    app: hypothesis-engine
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hypothesis-engine
      component: api
  template:
    metadata:
      labels:
        app: hypothesis-engine
        component: api
    spec:
      containers:
      - name: api
        image: hypothesis-engine-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: hypothesis-engine-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: hypothesis-engine-secrets
              key: openai-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: hypothesis-engine-api
spec:
  selector:
    app: hypothesis-engine
    component: api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: hypothesis-engine-config
data:
  LOG_LEVEL: "INFO"
  CHROMA_HOST: "chromadb-service"
  CHROMA_PORT: "8000"
---
apiVersion: v1
kind: Secret
metadata:
  name: hypothesis-engine-secrets
type: Opaque
stringData:
  database-url: "postgresql://user:password@postgres-service:5432/hypothesis"
  openai-api-key: "sk-your-key-here"
"""

DOCKERFILE = """
# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""


def generate_kubernetes_manifests(output_dir: str):
    """Generate Kubernetes deployment files"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'deployment.yaml'), 'w') as f:
        f.write(KUBERNETES_DEPLOYMENT)
    
    with open(os.path.join(output_dir, 'Dockerfile'), 'w') as f:
        f.write(DOCKERFILE)
    
    print(f"Generated Kubernetes manifests in {output_dir}")


if __name__ == "__main__":
    generate_kubernetes_manifests("./deploy")
