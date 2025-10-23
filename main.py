"""
FastAPI application entry point.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio

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


class TrainRequest(BaseModel):
    """Request model for offline training."""
    dataset: list[Dict[str, Any]]
    node: str = "fraud_detection"
    max_samples: int = 10


class TraceRequest(BaseModel):
    """Request model for tracing (online learning)."""
    input_text: str
    node: str
    output: str
    ground_truth: Optional[str] = None
    agent_reasoning: Optional[str] = None
    bullet_ids: Optional[Dict[str, list[str]]] = None  # {"full": [...], "online": [...]}


class ContextRequest(BaseModel):
    """Request model for getting context."""
    input_text: str
    node: str
    max_bullets_per_evaluator: int = 10


@app.post("/api/v1/train")
async def train_offline(request: TrainRequest, db: Session = Depends(get_db)):
    """
    Train offline mode with provided dataset.
    
    Receives dataset with inputs and outputs, trains bullets on it with Darwin-Gödel evolution.
    
    Args:
        request: Training request with dataset
        db: Database session
    
    Returns:
        dict: Training results with statistics
    """
    init_ace_components()
    
    try:
        from ace.training_pipeline import TrainingPipeline
        from ace.bullet_playbook import BulletPlaybook
        from ace.hybrid_selector import HybridSelector
        from ace.reflector import Reflector
        from ace.curator import Curator
        from ace.pattern_manager import PatternManager
        from ace.darwin_bullet_evolver import DarwinBulletEvolver
        
        # Initialize components with database
        pattern_manager = PatternManager(db_session=db)
        selector = HybridSelector(db_session=db, pattern_manager=pattern_manager)
        reflector = Reflector()
        curator = Curator()
        darwin_evolver = DarwinBulletEvolver(db_session=db)
        
        # Create dummy agent (we don't actually use it in training)
        class DummyAgent:
            pass
        
        training_pipeline = TrainingPipeline(DummyAgent(), selector, reflector, curator, darwin_evolver=darwin_evolver)
        playbook = BulletPlaybook(db_session=db)
        
        # Process dataset
        dataset = request.dataset[:request.max_samples]
        logger.info(f"Training on {len(dataset)} samples for node: {request.node}")
        
        bullets_generated = []
        for idx, item in enumerate(dataset):
            query = item.get('query', str(item))
            ground_truth = item.get('answer', item.get('output', 'DECLINE'))
            predicted = item.get('predicted', ground_truth)  # Use provided prediction or ground truth
            
            # Get pattern classification
            pattern_id, confidence = pattern_manager.classify_input_to_category(
                input_summary=query,
                node=request.node
            )
            
            # Generate bullet
            bullet_id = await training_pipeline.add_bullet_from_reflection(
                query=query,
                predicted=predicted,
                correct=ground_truth,
                node=request.node,
                agent_reasoning="",
                playbook=playbook,
                source="offline",
                evaluator=request.node
            )
            
            if bullet_id:
                bullets_generated.append(bullet_id)
        
        # Get final stats
        final_bullets = playbook.get_bullets_for_node(request.node)
        
        return {
            "status": "success",
            "node": request.node,
            "samples_processed": len(dataset),
            "bullets_generated": len(bullets_generated),
            "total_bullets": len(final_bullets),
            "unique_bullets": len(set(b.content for b in final_bullets))
        }
    
    except Exception as e:
        logger.error(f"Error in train: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_trace_background(
    request: TraceRequest,
    transaction_id: int,
    pattern_id: int,
    is_correct: Optional[bool]
):
    """
    Background task to process trace logic (bullet generation and effectiveness tracking).
    
    This runs asynchronously after the API response is returned.
    If ground_truth is not provided, uses LLM judge for evaluation.
    """
    from database import SessionLocal
    from ace.training_pipeline import TrainingPipeline
    from ace.bullet_playbook import BulletPlaybook
    from ace.hybrid_selector import HybridSelector
    from ace.reflector import Reflector
    from ace.curator import Curator
    from ace.pattern_manager import PatternManager
    from ace.darwin_bullet_evolver import DarwinBulletEvolver
    from models import Transaction, LLMJudge
    from openai import OpenAI
    import json
    import os
    
    # Create a new database session for background task
    db = SessionLocal()
    
    try:
        # Evaluate with LLM judge if ground_truth not provided
        if is_correct is None:
            # Get LLM judge for this node
            judge = db.query(LLMJudge).filter(
                LLMJudge.node == request.node,
                LLMJudge.is_active == True
            ).first()
            
            if judge:
                logger.info(f"Evaluating transaction {transaction_id} with LLM judge: {judge.node}")
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Use judge's system prompt and evaluation criteria
                evaluation_prompt = f"""{judge.system_prompt}

Input: {request.input_text}

Output: {request.output}

Evaluate the output based on these criteria:
{json.dumps(judge.evaluation_criteria, indent=2) if judge.evaluation_criteria else "Use your best judgment"}

Respond in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
                
                response = client.chat.completions.create(
                    model=judge.model,
                    messages=[
                        {"role": "system", "content": f"You are an LLM judge for {request.node}."},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=judge.temperature
                )
                
                result = json.loads(response.choices[0].message.content)
                is_correct = result.get("is_correct", False)
                judge_reasoning = result.get("reasoning", "")
                
                # Update transaction with judge evaluation
                txn = db.query(Transaction).filter(Transaction.id == transaction_id).first()
                if txn:
                    txn.is_correct = is_correct
                    db.commit()
                
                logger.info(f"LLM judge evaluation for {request.node}: is_correct={is_correct}")
            else:
                # No judge available, default to False and log warning
                is_correct = False
                logger.warning(f"No LLM judge found for node '{request.node}' in database. Please create an LLM judge for this node.")
        
        # Initialize components with database
        pattern_manager = PatternManager(db_session=db)
        selector = HybridSelector(db_session=db, pattern_manager=pattern_manager)
        reflector = Reflector()
        curator = Curator()
        darwin_evolver = DarwinBulletEvolver(db_session=db)
        
        # Create dummy agent (we don't actually use it in trace)
        class DummyAgent:
            pass
        
        training_pipeline = TrainingPipeline(DummyAgent(), selector, reflector, curator, darwin_evolver=darwin_evolver)
        playbook = BulletPlaybook(db_session=db)
        
        # Online learning: Generate bullet
        bullet_id = await training_pipeline.add_bullet_from_reflection(
            query=request.input_text,
            predicted=request.output,
            correct=request.ground_truth or request.output,
            node=request.node,
            agent_reasoning=request.agent_reasoning or "",
            playbook=playbook,
            source="online",
            evaluator=request.node
        )
        
        # Record bullet effectiveness for bullets that were used
        if request.bullet_ids and pattern_id:
            # Record effectiveness for full context bullets (offline + online)
            if "full" in request.bullet_ids:
                for bullet_id_used in request.bullet_ids["full"]:
                    pattern_manager.record_bullet_effectiveness(
                        pattern_id=pattern_id,
                        bullet_id=bullet_id_used,
                        node=request.node,
                        is_helpful=is_correct
                    )
            
            # Record effectiveness for online-only bullets
            if "online" in request.bullet_ids:
                for bullet_id_used in request.bullet_ids["online"]:
                    pattern_manager.record_bullet_effectiveness(
                        pattern_id=pattern_id,
                        bullet_id=bullet_id_used,
                        node=request.node,
                        is_helpful=is_correct
                    )
        
        logger.info(f"Background trace processing completed for transaction {transaction_id}")
        
    except Exception as e:
        logger.error(f"Error in background trace processing: {e}")
    finally:
        db.close()


@app.post("/api/v1/trace")
async def trace_and_learn(request: TraceRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Trace transaction and learn online (non-blocking).
    
    Saves transaction to database immediately and executes online training in background.
    Uses LLM judge for evaluation if ground_truth is not provided.
    
    Args:
        request: Trace request with input, output, and ground truth
        background_tasks: FastAPI background tasks
        db: Database session
    
    Returns:
        dict: Transaction info (returns immediately, processing continues in background)
    """
    try:
        from ace.pattern_manager import PatternManager
        from models import Transaction
        
        # Initialize pattern manager
        pattern_manager = PatternManager(db_session=db)
        
        # Get pattern classification
        pattern_id, confidence = pattern_manager.classify_input_to_category(
            input_summary=request.input_text,
            node=request.node
        )
        
        # Determine correctness
        if request.ground_truth:
            # If ground truth provided, use direct comparison
            is_correct = (request.output == request.ground_truth)
            correct_decision = request.ground_truth
        else:
            # If no ground truth, will be evaluated by LLM judge in background
            is_correct = None  # Unknown - will be evaluated by judge
            correct_decision = None  # Unknown - will be evaluated by judge
        
        # Save transaction (quick operation)
        transaction_data = {
            "systemprompt": "You are a fraud detection expert analyzing transactions.",
            "userprompt": request.input_text,
            "output": request.output,
            "reasoning": request.agent_reasoning or ""
        }
        
        txn = Transaction(
            transaction_data=transaction_data,
            mode="online",
            node=request.node,
            predicted_decision=request.output,
            correct_decision=correct_decision or request.output,  # Use output as placeholder if None
            is_correct=is_correct or False,  # Default to False if None
            input_pattern_id=pattern_id
        )
        db.add(txn)
        db.commit()
        db.flush()
        
        # Schedule background task for heavy processing
        background_tasks.add_task(
            process_trace_background,
            request,
            txn.id,
            pattern_id,
            is_correct
        )
        
        # Return immediately without waiting for background processing
        return {
            "status": "success",
            "node": request.node,
            "transaction_id": txn.id,
            "pattern_id": pattern_id,
            "is_correct": is_correct,
            "message": "Processing in background"
        }
    
    except Exception as e:
        logger.error(f"Error in trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/context")
async def get_context(request: ContextRequest, db: Session = Depends(get_db)):
    """
    Get context for agent prompt.
    
    Returns full text context with bullets organized by evaluator.
    Returns both full context (offline + online) and online-only context.
    
    Args:
        request: Context request with input and node
        db: Database session
    
    Returns:
        dict: Context with 'full' and 'online' keys containing full text
    """
    init_ace_components()
    
    try:
        from ace.bullet_playbook import BulletPlaybook
        from ace.hybrid_selector import HybridSelector
        from ace.pattern_manager import PatternManager
        from models import Bullet as BulletModel
        
        # Initialize components with database
        pattern_manager = PatternManager(db_session=db)
        selector = HybridSelector(db_session=db, pattern_manager=pattern_manager)
        playbook = BulletPlaybook(db_session=db)
        
        # Get pattern classification
        pattern_id, confidence = pattern_manager.classify_input_to_category(
            input_summary=request.input_text,
            node=request.node
        )
        
        # Get all evaluators from database
        all_evaluators = db.query(BulletModel.evaluator).filter(
            BulletModel.node == request.node,
            BulletModel.evaluator.isnot(None)
        ).distinct().all()
        
        evaluator_names = [e[0] for e in all_evaluators] if all_evaluators else [request.node]
        
        # Build full context (offline + online bullets)
        full_context = ""
        online_context = ""
        full_bullet_ids = []  # Track bullets for full context
        online_bullet_ids = []  # Track bullets for online context
        
        for evaluator_name in evaluator_names:
            # Get ALL bullets for this evaluator
            evaluator_bullets = playbook.get_bullets_for_node(request.node, evaluator=evaluator_name)
            
            # Select top bullets for this evaluator using intelligent selection
            if evaluator_bullets:
                # Temporarily create a filtered playbook for this evaluator
                temp_bullets = [b for b in evaluator_bullets]
                temp_playbook = BulletPlaybook()
                temp_playbook.bullets = temp_bullets
                temp_playbook._node_index[request.node] = temp_bullets
                
                selected, _ = selector.select_bullets(
                    query=request.input_text,
                    node=request.node,
                    playbook=temp_playbook,
                    n_bullets=min(request.max_bullets_per_evaluator, len(temp_bullets)),
                    iteration=0,
                    pattern_id=pattern_id
                )
                
                # Add to full context
                if selected:
                    full_context += f"\n\n{evaluator_name.upper()} Rules:\n"
                    for bullet in selected[:request.max_bullets_per_evaluator]:
                        full_context += f"- {bullet.content}\n"
                        full_bullet_ids.append(bullet.id)  # Track bullet IDs for full context
                    
                    # Add only online bullets to online context
                    online_bullets = [b for b in selected if b.source == "online"]
                    if online_bullets:
                        online_context += f"\n\n{evaluator_name.upper()} Rules:\n"
                        for bullet in online_bullets[:request.max_bullets_per_evaluator]:
                            online_context += f"- {bullet.content}\n"
                            online_bullet_ids.append(bullet.id)  # Track bullet IDs for online context
        
        return {
            "status": "success",
            "node": request.node,
            "pattern_id": pattern_id,
            "bullet_ids": {
                "full": full_bullet_ids,      # Bullets for offline + online context
                "online": online_bullet_ids   # Bullets for online-only context
            },
            "context": {
                "full": full_context.strip(),
                "online": online_context.strip()
            }
        }
    
    except Exception as e:
        logger.error(f"Error in context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
async def get_node_playbook(node: str, limit: int = 10, query: Optional[str] = None):
    """
    Get bullets for a specific node.
    
    If query is provided, uses intelligent selection to return top bullets.
    Otherwise, returns all bullets up to limit.
    
    Args:
        node: Agent node name
        limit: Maximum number of bullets to return (default: 10)
        query: Optional query text for intelligent bullet selection
    """
    init_ace_components()
    
    if query:
        # Use intelligent selection
        bullets, _ = selector.select_bullets(
            query=query,
            node=node,
            playbook=playbook,
            n_bullets=limit,
            iteration=0
        )
        return {"node": node, "bullets": bullets, "selection_method": "intelligent"}
    else:
        # Return all bullets up to limit
        bullets = playbook.get_bullets_for_node(node)[:limit]
        return {"node": node, "bullets": bullets, "selection_method": "all"}


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

