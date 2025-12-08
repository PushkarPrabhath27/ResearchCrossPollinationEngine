"""
FastAPI Main Application

This is the entry point for the Hypothesis Engine API.
Handles initialization, middleware, and application configuration.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from typing import Dict

from src.config import get_settings
from src.utils.logger import setup_logging, get_logger

# Import routes (will be created in later prompts)
# from src.api.routes import router

# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting Hypothesis Engine API...")
    config = get_settings()
    
    # Initialize logging
    setup_logging(
        level=config.server.log_level.upper(),
        json_format=config.is_production()
    )
    
    # Initialize database connections
    logger.info("Initializing database connections...")
    # TODO: Initialize ChromaDB and PostgreSQL connections
    
    # Initialize LLM providers
    logger.info(f"Initializing LLM provider: {config.agent.llm_provider}")
    # TODO: Initialize LangChain LLM
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Hypothesis Engine API...")
    # TODO: Close database connections
    # TODO: Cleanup resources
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Scientific Hypothesis Cross-Pollination Engine",
    description=(
        "AI-powered research assistant that discovers novel research directions "
        "by finding unexpected connections across scientific disciplines"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Configure CORS
config = get_settings()
if config.server.allow_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Include routers (will be uncommented when routes.py is created)
# app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
async def root() -> Dict[str, str]:
    """
    Root endpoint
    Returns API information
    """
    return {
        "name": "Scientific Hypothesis Cross-Pollination Engine",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    Used by container orchestration and monitoring systems
    """
    # TODO: Check database connections
    # TODO: Check LLM provider availability
    
    return {
        "status": "healthy",
        "database": "ok",  # TODO: Actual check
        "llm": "ok"  # TODO: Actual check
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint
    Indicates if the service is ready to accept requests
    """
    # TODO: Verify all dependencies are initialized
    
    return {
        "status": "ready"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if not config.is_production() else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    config = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        log_level=config.server.log_level
    )
