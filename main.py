"""
FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import get_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    Yields:
        None: Application is ready
    """
    # Startup
    logger.info("Starting application...")
    yield
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Self-Improving Engine API",
    description="FastAPI application with PostgreSQL database",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """
    Root endpoint.
    
    Returns:
        dict: Welcome message
    """
    return {"message": "Welcome to Self-Improving Engine API"}


@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint.
    
    Checks database connectivity and returns service status.
    
    Args:
        db: Database session dependency
    
    Returns:
        JSONResponse: Health status with database connection status
    """
    try:
        # Test database connection
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"
    
    return JSONResponse(
        status_code=200 if db_status == "connected" else 503,
        content={
            "status": "healthy" if db_status == "connected" else "unhealthy",
            "database": db_status,
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

