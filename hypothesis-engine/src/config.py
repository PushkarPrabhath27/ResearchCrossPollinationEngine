"""
Configuration management for the Scientific Hypothesis Cross-Pollination Engine.

This module provides type-safe configuration using Pydantic Settings,
loading from environment variables and .env files with proper validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from typing import Optional, Literal, List
import os
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Vector database and metadata store configuration"""
    
    # ChromaDB settings
    chroma_persist_dir: str = Field(
        default="./data/embeddings",
        description="Directory for ChromaDB persistent storage"
    )
    chroma_host: str = Field(
        default="localhost",
        description="ChromaDB server host"
    )
    chroma_port: int = Field(
        default=8001,
        description="ChromaDB server port"
    )
    chroma_collection_name: str = Field(
        default="scientific_papers",
        description="Primary collection name"
    )
    
    # PostgreSQL/SQLite metadata store
    metadata_db_path: str = Field(
        default="./data/metadata.db",
        description="Path to SQLite metadata database"
    )
    database_url: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URL (optional, uses SQLite if not provided)"
    )
    
    # Embedding settings
    embedding_model: str = Field(
        default="allenai-specter",
        description="Sentence transformer model for embeddings"
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension of embedding vectors"
    )
    
    model_config = SettingsConfigDict(env_prefix='DB_')


class APISettings(BaseSettings):
    """External API configuration"""
    
    # LLM APIs
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key (optional)"
    )
    
    # Academic APIs
    semantic_scholar_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional, increases rate limit)"
    )
    entrez_email: str = Field(
        default="user@example.com",
        description="Email for NCBI Entrez API (required)"
    )
    entrez_api_key: Optional[str] = Field(
        default=None,
        description="NCBI API key (optional, increases rate limit)"  
    )
    
    # Rate limiting
    max_requests_per_minute: int = Field(
        default=60,
        description="Global rate limit for API calls"
    )
    request_timeout: int = Field(
        default=30,
        description="Timeout for API requests in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    retry_delay: int = Field(
        default=2,
        description="Delay between retries in seconds"
    )
    
    model_config = SettingsConfigDict(env_prefix='API_')
    
    @field_validator('entrez_email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format"""
        if '@' not in v or '.' not in v.split('@')[1]:
            raise ValueError("Invalid email format for ENTREZ_EMAIL")
        return v


class AgentSettings(BaseSettings):
    """LangChain agent and LLM configuration"""
    
    # LLM provider
    llm_provider: Literal["openai", "ollama", "anthropic"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    llm_model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model name for primary LLM"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation"
    )
    llm_max_tokens: int = Field(
        default=2000,
        description="Maximum tokens per LLM response"
    )
    
    # Ollama settings (if using local LLM)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    ollama_model: str = Field(
        default="llama2",
        description="Ollama model name"
    )
    
    # Agent-specific settings
    primary_agent_model: Optional[str] = Field(
        default=None,
        description="Model for primary domain agent (uses llm_model if not set)"
    )
    cross_domain_agent_model: Optional[str] = Field(
        default=None,
        description="Model for cross-domain agent"
    )
    methodology_agent_model: Optional[str] = Field(
        default=None,
        description="Model for methodology transfer agent"
    )
    
    # Agent parameters
    agent_max_iterations: int = Field(
        default=15,
        description="Maximum iterations for agent reasoning"
    )
    agent_verbose: bool = Field(
        default=False,
        description="Enable verbose logging for agents"
    )
    
    model_config = SettingsConfigDict(env_prefix='AGENT_')
    
    @field_validator('llm_temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is in valid range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class IngestionSettings(BaseSettings):
    """Data ingestion and processing configuration"""
    
    # Batch processing
    batch_size: int = Field(
        default=50,
        description="Number of papers to process in a batch"
    )
    max_papers_per_source: int = Field(
        default=1000,
        description="Maximum papers to fetch per source"
    )
    
    # Text processing
    chunk_size: int = Field(
        default=1000,
        description="Size of text chunks for embedding"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between text chunks"
    )
    
    # Update frequency
    update_interval_days: int = Field(
        default=7,
        description="Days between data source updates"
    )
    
    # Source priorities (1-10 scale)
    source_priorities: dict = Field(
        default={
            "arxiv": 9,
            "pubmed": 9,
            "semantic_scholar": 8,
            "openalex": 7
        },
        description="Priority scores for different data sources"
    )
    
    # Storage paths
    raw_data_dir: str = Field(
        default="./data/raw",
        description="Directory for raw downloaded papers"
    )
    processed_data_dir: str = Field(
        default="./data/processed",
        description="Directory for processed papers"
    )
    
    model_config = SettingsConfigDict(env_prefix='INGEST_')


class SearchSettings(BaseSettings):
    """Search and retrieval configuration"""
    
    # Default  parameters
    default_top_k: int = Field(
        default=20,
        description="Default number of results to return"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results"
    )
    
    # Reranking
    enable_reranking: bool = Field(
        default=True,  
        description="Enable result reranking"
    )
    rerank_top_k: int = Field(
        default=100,
        description="Number of results to fetch for reranking"
    )
    
    # Filtering
    min_citation_count: int = Field(
        default=0,
        description="Minimum citations for papers"
    )
    max_paper_age_years: Optional[int] = Field(
        default=None,
        description="Maximum age of papers in years (None = no limit)"
    )
    
    # Multi-field search weights
    title_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    abstract_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    content_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    
    model_config = SettingsConfigDict(env_prefix='SEARCH_')
    
    @model_validator(mode='after')
    def validate_weights(self) -> 'SearchSettings':
        """Ensure search weights sum to 1.0"""
        total = self.title_weight + self.abstract_weight + self.content_weight
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError("Search weights must sum to 1.0")
        return self


class HypothesisSettings(BaseSettings):
    """Hypothesis generation configuration"""
    
    # Generation parameters
    default_num_hypotheses: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of hypotheses to generate"
    )
    creativity_level: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Creativity level for hypothesis generation"
    )
    
    # Validation
    enable_validation: bool = Field(
        default=True,
        description="Enable automatic hypothesis validation"
    )
    min_supporting_papers: int = Field(
        default=3,
        description="Minimum supporting papers for hypothesis"
    )
    
    # Scoring weights
    novelty_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    feasibility_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    impact_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    
    model_config =  SettingsConfigDict(env_prefix='HYPO_')


class ServerSettings(BaseSettings):
    """API server configuration"""
    
    host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    port: int = Field(
        default=8000,
        description="API server port"
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload (development only)"
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes"
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Logging level"
    )
    
    # CORS
    allow_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        description="Allowed CORS origins"
    )
    
    model_config = SettingsConfigDict(env_prefix='SERVER_')


class Settings(BaseSettings):
    """Main configuration class combining all settings"""
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    api: APISettings = Field(default_factory=APISettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    hypothesis: HypothesisSettings = Field(default_factory=HypothesisSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )
    
    @model_validator(mode='after')
    def validate_api_keys(self) -> 'Settings':
        """Validate that required API keys are present"""
        if self.agent.llm_provider == "openai" and not self.api.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when using OpenAI as LLM provider"
            )
        if self.agent.llm_provider == "anthropic" and not self.api.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when using Anthropic as LLM provider"
            )
        return self
    
    @model_validator(mode='after')
    def create_directories(self) -> 'Settings':
        """Ensure required directories exist"""
        directories = [
            self.database.chroma_persist_dir,
            self.ingestion.raw_data_dir,
            self.ingestion.processed_data_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        return self
    
    def get_llm_config(self, agent_type: str = "default") -> dict:
        """
        Get LLM configuration for specific agent type
        
        Args:
            agent_type: Type of agent ("primary", "cross_domain", "methodology", or "default")
        
        Returns:
            Dictionary with LLM configuration
        """
        model = self.agent.llm_model
        
        # Use agent-specific model if configured
        if agent_type == "primary" and self.agent.primary_agent_model:
            model = self.agent.primary_agent_model
        elif agent_type == "cross_domain" and self.agent.cross_domain_agent_model:
            model = self.agent.cross_domain_agent_model
        elif agent_type == "methodology" and self.agent.methodology_agent_model:
            model = self.agent.methodology_agent_model
        
        config = {
            "provider": self.agent.llm_provider,
            "model": model,
            "temperature": self.agent.llm_temperature,
            "max_tokens": self.agent.llm_max_tokens,
        }
        
        if self.agent.llm_provider == "openai":
            config["api_key"] = self.api.openai_api_key
        elif self.agent.llm_provider == "ollama":
            config["base_url"] = self.agent.ollama_base_url
            config["model"] = self.agent.ollama_model
        elif self.agent.llm_provider == "anthropic":
            config["api_key"] = self.api.anthropic_api_key
        
        return config
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create global settings instance
    
    This function ensures only one Settings instance exists (singleton pattern)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment/file"""
    global _settings
    _settings = Settings()
    return _settings


# Convenience function for quick access
def get_config() -> Settings:
    """Alias for get_settings()"""
    return get_settings()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_settings()
        print("✅ Configuration loaded successfully!")
        print(f"Environment: {config.environment}")
        print(f"LLM Provider: {config.agent.llm_provider}")
        print(f"Database: {config.database.chroma_persist_dir}")
        print(f"Embedding Model: {config.database.embedding_model}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
