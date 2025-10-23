"""
FastAPI application entry point.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, Optional

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
    description="FastAPI application with PostgreSQL database and ACE system",
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
async def health_check(db: Session = Depends(get_db)):
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
        db.execute(text("SELECT 1"))
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


# Lazy initialization of ACE components
_ace_initialized = False
fraud_agent = None
selector = None
reflector = None
curator = None
training_pipeline = None
playbook = None


def init_ace_components():
    """Initialize ACE components lazily."""
    global _ace_initialized, fraud_agent, selector, reflector, curator, training_pipeline, playbook
    
    if _ace_initialized:
        return
    
    try:
        from ace.bullet_playbook import BulletPlaybook
        from ace.hybrid_selector import HybridSelector
        from ace.reflector import Reflector
        from ace.curator import Curator
        from ace.training_pipeline import TrainingPipeline
        from agents.mock_fraud_agent import MockFraudAgent
        
        fraud_agent = MockFraudAgent()
        selector = HybridSelector()
        reflector = Reflector()
        curator = Curator()
        training_pipeline = TrainingPipeline(fraud_agent, selector, reflector, curator)
        playbook = BulletPlaybook()
        
        _ace_initialized = True
        logger.info("ACE components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ACE components: {e}")
        raise


# Request models
class AnalyzeRequest(BaseModel):
    transaction: Dict[str, Any]
    mode: str = "hybrid"


class EvaluateRequest(BaseModel):
    input_text: str
    node: str
    output: str
    ground_truth: Optional[str] = None


class GetBulletsRequest(BaseModel):
    query: str
    node: str
    mode: str = "hybrid"


@app.post("/api/v1/analyze")
async def analyze_transaction(request: AnalyzeRequest):
    """Analyze transaction with specified mode."""
    init_ace_components()
    
    try:
        if request.mode == "vanilla":
            result = await fraud_agent.analyze(request.transaction)
            return {"mode": "vanilla", **result}
        
        elif request.mode == "offline_online":
            result = await fraud_agent.analyze(request.transaction)
            return {"mode": "offline_online", **result}
        
        elif request.mode == "online_only":
            result = await fraud_agent.analyze(request.transaction)
            return {"mode": "online_only", **result}
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
    
    except Exception as e:
        logger.error(f"Error in analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate")
async def evaluate_and_generate_bullets(request: EvaluateRequest):
    """Evaluate agent output and generate bullets using LLM as judge."""
    init_ace_components()
    
    try:
        is_correct = await judge_correctness(request.input_text, request.output, request.ground_truth)
        
        bullet_id = await training_pipeline.add_bullet_from_reflection(
            query=request.input_text,
            predicted=request.output,
            correct=request.ground_truth or request.output,
            node=request.node,
            agent_reasoning=request.output,
            playbook=playbook,
            source="online"
        )
        
        return {
            "node": request.node,
            "is_correct": is_correct,
            "bullet_id": bullet_id,
            "bullet_added": bullet_id is not None
        }
    
    except Exception as e:
        logger.error(f"Error in evaluate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/get-bullets")
async def get_bullets(request: GetBulletsRequest):
    """Get bullets for a query and node based on mode."""
    init_ace_components()
    
    try:
        if request.mode == "vanilla":
            return {"mode": "vanilla", "bullets": []}
        
        elif request.mode == "offline_online":
            bullets_offline, _ = selector.select_bullets(
                query=request.query, node=request.node, playbook=playbook,
                n_bullets=5, iteration=0, source="offline"
            )
            bullets_online, _ = selector.select_bullets(
                query=request.query, node=request.node, playbook=playbook,
                n_bullets=5, iteration=0, source="online"
            )
            
            return {"mode": "offline_online", "bullets": {"offline": bullets_offline, "online": bullets_online}}
        
        elif request.mode == "online_only":
            bullets, _ = selector.select_bullets(
                query=request.query, node=request.node, playbook=playbook,
                n_bullets=5, iteration=0, source="online"
            )
            return {"mode": "online_only", "bullets": bullets}
        
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
    
    except Exception as e:
        logger.error(f"Error in get-bullets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/playbook/stats")
async def get_playbook_stats():
    """Get statistics for the playbook."""
    init_ace_components()
    stats = playbook.get_stats()
    return {"stats": stats, "total_bullets": stats["total_bullets"]}


@app.get("/api/v1/playbook/{node}")
async def get_node_playbook(node: str):
    """Get all bullets for a specific node."""
    init_ace_components()
    bullets = playbook.get_bullets_for_node(node)
    return {"node": node, "bullets": bullets}


@app.post("/api/v1/test/comprehensive")
async def run_comprehensive_test():
    """
    Run comprehensive test suite comparing all 3 modes.
    
    This endpoint triggers the full test suite and returns results.
    Note: This is a long-running operation and may take several minutes.
    """
    import subprocess
    import asyncio
    
    try:
        # Run the test script asynchronously
        process = await asyncio.create_subprocess_exec(
            "python",
            "test_ace_comprehensive.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Test failed: {stderr.decode()}")
            raise HTTPException(
                status_code=500,
                detail=f"Test execution failed: {stderr.decode()}"
            )
        
        return {
            "status": "completed",
            "output": stdout.decode(),
            "message": "Comprehensive test completed successfully. Check logs for detailed results."
        }
    
    except Exception as e:
        logger.error(f"Error running comprehensive test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def judge_correctness(input_text: str, output: str, ground_truth: Optional[str] = None) -> bool:
    """Use LLM to judge if output is correct."""
    from ace.llm_judge import get_judge
    
    judge = get_judge()
    is_correct, confidence = await judge.judge(input_text, output, ground_truth)
    
    return is_correct


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

