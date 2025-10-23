"""
FastAPI application entry point.
"""
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio

# Load environment variables from .env file
load_dotenv()

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
    model_type: Optional[str] = None  # Model type identifier
    session_id: Optional[str] = None
    run_id: Optional[str] = None
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
        
        # Initialize Darwin-Gödel evolution if enabled
        darwin_evolver = None
        if settings.enable_darwin_evolution:
            darwin_evolver = DarwinBulletEvolver(db_session=db)
            logger.info("Darwin-Gödel evolution enabled")
        else:
            logger.info("Darwin-Gödel evolution disabled")
        
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
    is_correct: Optional[bool]
):
    """
    Background task to process trace logic (bullet generation and effectiveness tracking).
    
    This runs asynchronously after the API response is returned.
    Pattern classification and LLM judge evaluation happen here.
    If ground_truth is not provided, uses LLM judge for evaluation.
    """
    from dotenv import load_dotenv
    load_dotenv()  # Ensure environment variables are loaded
    
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
        # Debug: Verify environment is loaded
        logger.debug(f"OPENAI_API_KEY loaded: {bool(os.getenv('OPENAI_API_KEY'))}")
        
        # Step 1: Pattern classification (does LLM call for embedding)
        pattern_manager = PatternManager(db_session=db)
        pattern_id, confidence = pattern_manager.classify_input_to_category(
            input_summary=request.input_text,
            node=request.node
        )
        
        # Update transaction with pattern_id
        txn = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        if txn:
            txn.input_pattern_id = pattern_id
            db.commit()
        
        # Step 2: Evaluate with LLM judge if ground_truth not provided
        if is_correct is None:
            # Get LLM judge for this node
            judge = db.query(LLMJudge).filter(
                LLMJudge.node == request.node,
                LLMJudge.is_active == True
            ).first()
            
            if judge:
                logger.info(f"Evaluating transaction {transaction_id} with LLM judge: {judge.node}")
                
                # Debug: Check if API key is loaded
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.error("OPENAI_API_KEY not found in environment variables!")
                    raise ValueError("OPENAI_API_KEY is not set")
                
                client = OpenAI()  # Automatically reads OPENAI_API_KEY from environment
                
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
        
        # Initialize components with database (pattern_manager already initialized above)
        selector = HybridSelector(db_session=db, pattern_manager=pattern_manager)
        reflector = Reflector()
        curator = Curator()
        
        # Initialize Darwin-Gödel evolution if enabled
        darwin_evolver = None
        enable_evolution = os.getenv("ENABLE_DARWIN_EVOLUTION", "false").lower() == "true"
        if enable_evolution:
            darwin_evolver = DarwinBulletEvolver(db_session=db)
            logger.info("Darwin-Gödel evolution enabled")
        else:
            logger.info("Darwin-Gödel evolution disabled")
        
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
        
        # Update session run metrics if session_id and run_id are provided
        if request.session_id and request.run_id:
            from models import SessionRunMetrics
            
            # Get or create metrics record for this session/run/evaluator combination
            metrics = db.query(SessionRunMetrics).filter(
                SessionRunMetrics.session_id == request.session_id,
                SessionRunMetrics.run_id == request.run_id,
                SessionRunMetrics.node == request.node,
                SessionRunMetrics.evaluator == request.node
            ).first()
            
            if not metrics:
                # Create new metrics record
                metrics = SessionRunMetrics(
                    session_id=request.session_id,
                    run_id=request.run_id,
                    node=request.node,
                    evaluator=request.node,
                    correct_count=0,
                    total_count=0,
                    accuracy=0.0
                )
                db.add(metrics)
            
            # Update metrics
            metrics.total_count += 1
            if is_correct:
                metrics.correct_count += 1
            metrics.accuracy = metrics.correct_count / metrics.total_count if metrics.total_count > 0 else 0.0
            
            db.commit()
            logger.info(f"Updated metrics for session={request.session_id}, run={request.run_id}, evaluator={request.node}: accuracy={metrics.accuracy:.2%}")
        
        logger.info(f"Background trace processing completed for transaction {transaction_id}")
        
    except Exception as e:
        logger.error(f"Error in background trace processing: {e}")
    finally:
        db.close()


@app.post("/api/v1/trace")
async def trace_and_learn(request: TraceRequest, db: Session = Depends(get_db)):
    """
    Trace transaction and learn online (synchronous).
    
    Saves transaction to database and executes online training.
    Uses LLM judge for evaluation if ground_truth is not provided.
    Returns only after all processing is complete.
    
    Args:
        request: Trace request with input, output, and ground truth
        db: Database session
    
    Returns:
        dict: Transaction info with pattern_id and is_correct
    """
    try:
        from models import Transaction, LLMJudge
        from ace.pattern_manager import PatternManager
        from ace.training_pipeline import TrainingPipeline
        from ace.bullet_playbook import BulletPlaybook
        from ace.hybrid_selector import HybridSelector
        from ace.reflector import Reflector
        from ace.curator import Curator
        from ace.darwin_bullet_evolver import DarwinBulletEvolver
        from openai import OpenAI
        import json
        import os
        
        logger.info(f"Processing trace for node={request.node}, session_id={request.session_id}, run_id={request.run_id}")
        
        # Step 1: Determine mode from model_type (map "full" to "offline_online")
        mode = request.model_type or "online"
        if mode == "full":
            mode = "offline_online"
        
        # Step 2: Pattern classification (does LLM call for embedding)
        pattern_manager = PatternManager(db_session=db)
        pattern_id, confidence = pattern_manager.classify_input_to_category(
            input_summary=request.input_text,
            node=request.node
        )
        logger.info(f"Pattern classified: pattern_id={pattern_id}, confidence={confidence:.2f}")
        
        # Step 3: Evaluate with LLM judge (always use judge to track performance)
        judge_evaluation_result = None
        judge_reasoning = None
        judge_confidence = None
        judge_is_correct = None
        
        # Always get a judge for the node
        judge = db.query(LLMJudge).filter(
            LLMJudge.node == request.node,
            LLMJudge.is_active == True
        ).first()
        
        if judge:
            logger.info(f"Evaluating with LLM judge: {judge.node}")
            client = OpenAI()
            
            # Build evaluation prompt
            if request.ground_truth:
                evaluation_prompt = f"""{judge.system_prompt}

Input: {request.input_text}

Output: {request.output}

Ground Truth: {request.ground_truth}

Evaluate the output based on these criteria:
{json.dumps(judge.evaluation_criteria, indent=2) if judge.evaluation_criteria else "Use your best judgment"}

Respond in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
            else:
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
            judge_is_correct = result.get("is_correct", False)
            judge_reasoning = result.get("reasoning", "")
            judge_confidence = result.get("confidence", 0.5)
            
            # Determine final is_correct
            if request.ground_truth:
                # Use ground truth if available
                is_correct = (request.output == request.ground_truth)
                correct_decision = request.ground_truth
            else:
                # Use judge's decision if no ground truth
                is_correct = judge_is_correct
                correct_decision = request.output
            
            judge_evaluation_result = (judge.id, judge_reasoning, judge_confidence, judge_is_correct)
            logger.info(f"LLM judge evaluation: is_correct={is_correct}, confidence={judge_confidence}")
        else:
            logger.warning(f"No LLM judge found for node '{request.node}'")
            # Fallback: use ground truth if available
            if request.ground_truth:
                is_correct = (request.output == request.ground_truth)
                correct_decision = request.ground_truth
            else:
                is_correct = False
                correct_decision = request.output
        
        # Step 4: Save transaction
        transaction_data = {
            "systemprompt": "You are a fraud detection expert analyzing transactions.",
            "userprompt": request.input_text,
            "output": request.output,
            "reasoning": request.agent_reasoning or ""
        }
        
        txn = Transaction(
            transaction_data=transaction_data,
            mode=mode,
            node=request.node,
            session_id=request.session_id,
            run_id=request.run_id,
            predicted_decision=request.output,
            correct_decision=correct_decision or request.output,
            is_correct=is_correct,
            input_pattern_id=pattern_id
        )
        db.add(txn)
        db.commit()
        db.flush()
        
        # Step 4.5: Save judge evaluation if LLM judge was used
        if judge_evaluation_result:
            from models import JudgeEvaluation
            judge_id, judge_reasoning, judge_confidence, judge_is_correct = judge_evaluation_result
            
            # Determine if judge was correct (only if ground_truth was available)
            judge_was_correct = None
            if request.ground_truth:
                # Compare judge's decision with actual ground truth
                actual_correct = (request.output == request.ground_truth)
                judge_was_correct = (judge_is_correct == actual_correct)
            
            judge_eval = JudgeEvaluation(
                judge_id=judge_id,
                transaction_id=txn.id,
                input_text=request.input_text,
                output_text=request.output,
                ground_truth=request.ground_truth,
                is_correct=judge_is_correct,  # Save judge's decision, not actual correctness
                confidence=judge_confidence,
                reasoning=judge_reasoning,
                judge_was_correct=judge_was_correct
            )
            db.add(judge_eval)
            db.commit()
            logger.info(f"Saved judge evaluation for transaction {txn.id} (judge decision: {judge_is_correct}, judge_was_correct: {judge_was_correct})")
        
        # Step 5: Get evaluators for this node from LLM judges table
        from models import LLMJudge
        evaluators = db.query(LLMJudge.evaluator).filter(
            LLMJudge.node == request.node,
            LLMJudge.is_active == True
        ).distinct().all()
        
        evaluator_names = [e[0] for e in evaluators] if evaluators else []
        
        # Only generate bullets if evaluators exist for this node
        if not evaluator_names:
            logger.info(f"No LLM judges found for node {request.node}. Skipping bullet generation.")
            return {
                "status": "success",
                "node": request.node,
                "transaction_id": txn.id,
                "pattern_id": pattern_id,
                "is_correct": is_correct,
                "message": "Transaction saved, but no LLM judges found for this node"
            }
        
        # Process improvement (bullet generation) for nodes with evaluators
        logger.info(f"Processing with evaluators: {evaluator_names}")
        
        # Initialize components for online learning
        selector = HybridSelector(db_session=db, pattern_manager=pattern_manager)
        reflector = Reflector()
        curator = Curator()
        
        # Initialize Darwin-Gödel evolution if enabled
        darwin_evolver = None
        enable_evolution = os.getenv("ENABLE_DARWIN_EVOLUTION", "false").lower() == "true"
        if enable_evolution:
            darwin_evolver = DarwinBulletEvolver(db_session=db)
            logger.info("Darwin-Gödel evolution enabled")
        
        class DummyAgent:
            pass
        
        training_pipeline = TrainingPipeline(DummyAgent(), selector, reflector, curator, darwin_evolver=darwin_evolver)
        playbook = BulletPlaybook(db_session=db)
        
        # Step 6: Online learning - Generate bullet
        bullet_id = await training_pipeline.add_bullet_from_reflection(
            query=request.input_text,
            predicted=request.output,
            correct=request.ground_truth or request.output,
            node=request.node,
            agent_reasoning=request.agent_reasoning or "",
            playbook=playbook,
            source="online",
            evaluator=evaluator_names[0]
        )
        
        logger.info(f"Generated bullet: {bullet_id}")
        
        # Step 7: Record bullet effectiveness
        if request.bullet_ids and pattern_id:
            if "full" in request.bullet_ids:
                for bullet_id_used in request.bullet_ids["full"]:
                    pattern_manager.record_bullet_effectiveness(
                        pattern_id=pattern_id,
                        bullet_id=bullet_id_used,
                        node=request.node,
                        is_helpful=is_correct
                    )
            
            if "online" in request.bullet_ids:
                for bullet_id_used in request.bullet_ids["online"]:
                    pattern_manager.record_bullet_effectiveness(
                        pattern_id=pattern_id,
                        bullet_id=bullet_id_used,
                        node=request.node,
                        is_helpful=is_correct
                    )
        
        # Step 8: Update session run metrics (only if session_id and run_id provided)
        if request.session_id and request.run_id:
            from models import SessionRunMetrics
            
            # Track metrics for each evaluator
            for evaluator_name in evaluator_names:
                # Get the specific judge for this evaluator
                evaluator_judge = db.query(LLMJudge).filter(
                    LLMJudge.node == request.node,
                    LLMJudge.evaluator == evaluator_name,
                    LLMJudge.is_active == True
                ).first()
                
                # Determine if this evaluator was correct
                evaluator_is_correct = False
                if evaluator_judge:
                    # Evaluate with THIS specific evaluator's judge
                    client = OpenAI()
                    
                    if request.ground_truth:
                        evaluation_prompt = f"""{evaluator_judge.system_prompt}

Input: {request.input_text}

Output: {request.output}

Ground Truth: {request.ground_truth}

Evaluate the output based on these criteria:
{json.dumps(evaluator_judge.evaluation_criteria, indent=2) if evaluator_judge.evaluation_criteria else "Use your best judgment"}

Respond in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
                    else:
                        evaluation_prompt = f"""{evaluator_judge.system_prompt}

Input: {request.input_text}

Output: {request.output}

Evaluate the output based on these criteria:
{json.dumps(evaluator_judge.evaluation_criteria, indent=2) if evaluator_judge.evaluation_criteria else "Use your best judgment"}

Respond in JSON format:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
                    
                    response = client.chat.completions.create(
                        model=evaluator_judge.model,
                        messages=[
                            {"role": "system", "content": f"You are an LLM judge for {request.node}."},
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=evaluator_judge.temperature
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                    evaluator_is_correct = result.get("is_correct", False)
                    logger.info(f"Evaluator {evaluator_name} judge decision: {evaluator_is_correct}")
                elif request.ground_truth:
                    # If no judge for this evaluator but ground truth exists, compare
                    evaluator_is_correct = (request.output == request.ground_truth)
                else:
                    # If no judge and no ground truth, default to True
                    evaluator_is_correct = True
                
                metrics = db.query(SessionRunMetrics).filter(
                    SessionRunMetrics.session_id == request.session_id,
                    SessionRunMetrics.run_id == request.run_id,
                    SessionRunMetrics.node == request.node,
                    SessionRunMetrics.evaluator == evaluator_name,
                    SessionRunMetrics.mode == mode
                ).first()
                
                if not metrics:
                    metrics = SessionRunMetrics(
                        session_id=request.session_id,
                        run_id=request.run_id,
                        node=request.node,
                        evaluator=evaluator_name,
                        mode=mode,
                        correct_count=0,
                        total_count=0,
                        accuracy=0.0
                    )
                    db.add(metrics)
                
                metrics.total_count += 1
                if evaluator_is_correct:
                    metrics.correct_count += 1
                metrics.accuracy = metrics.correct_count / metrics.total_count if metrics.total_count > 0 else 0.0
                
                logger.info(f"Updated metrics for evaluator {evaluator_name}: is_correct={evaluator_is_correct}, accuracy={metrics.accuracy:.3f}")
            
            db.commit()
            logger.info(f"Updated metrics for {len(evaluator_names)} evaluators (mode={mode})")
        
        logger.info(f"Trace processing completed for transaction {txn.id}")
        
        return {
            "status": "success",
            "node": request.node,
            "transaction_id": txn.id,
            "pattern_id": pattern_id,
            "is_correct": is_correct,
            "message": "Processing completed"
        }
    
    except Exception as e:
        logger.error(f"Error in trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/judge-evaluations/{transaction_id}")
async def get_judge_evaluations(transaction_id: int, db: Session = Depends(get_db)):
    """
    Get judge evaluations for a specific transaction.
    
    Args:
        transaction_id: Transaction identifier
        db: Database session
    
    Returns:
        dict: Judge evaluations for the transaction
    """
    try:
        from models import JudgeEvaluation, LLMJudge
        
        # Get all judge evaluations for this transaction
        evaluations = db.query(JudgeEvaluation).filter(
            JudgeEvaluation.transaction_id == transaction_id
        ).all()
        
        if not evaluations:
            return {
                "status": "success",
                "transaction_id": transaction_id,
                "message": "No judge evaluations found for this transaction",
                "evaluations": []
            }
        
        # Get judge details for each evaluation
        result = []
        for eval in evaluations:
            judge = db.query(LLMJudge).filter(LLMJudge.id == eval.judge_id).first()
            
            result.append({
                "judge_id": eval.judge_id,
                "judge_node": judge.node if judge else None,
                "judge_evaluator": judge.evaluator if judge else None,
                "input_text": eval.input_text,
                "output_text": eval.output_text,
                "ground_truth": eval.ground_truth,
                "is_correct": eval.is_correct,
                "confidence": eval.confidence,
                "reasoning": eval.reasoning,
                "judge_was_correct": eval.judge_was_correct,
                "evaluated_at": eval.evaluated_at.isoformat() if eval.evaluated_at else None
            })
        
        return {
            "status": "success",
            "transaction_id": transaction_id,
            "evaluations": result
        }
    
    except Exception as e:
        logger.error(f"Error getting judge evaluations: {e}")
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
        
        # Initialize components with database
        pattern_manager = PatternManager(db_session=db)
        selector = HybridSelector(db_session=db, pattern_manager=pattern_manager)
        playbook = BulletPlaybook(db_session=db)
        
        # Get pattern classification
        pattern_id, confidence = pattern_manager.classify_input_to_category(
            input_summary=request.input_text,
            node=request.node
        )
        
        # Get all evaluators from LLM judges table
        from models import LLMJudge
        all_evaluators = db.query(LLMJudge.evaluator).filter(
            LLMJudge.node == request.node,
            LLMJudge.is_active == True
        ).distinct().all()
        
        evaluator_names = [e[0] for e in all_evaluators] if all_evaluators else []
        
        # If no evaluators, return empty context
        if not evaluator_names:
            logger.info(f"No evaluators found for node {request.node}. Returning empty context.")
            return {
                "status": "success",
                "node": request.node,
                "pattern_id": None,
                "bullet_ids": {
                    "full": [],
                    "online": []
                },
                "context": {
                    "full": "",
                    "online": ""
                }
            }
        
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


@app.get("/api/v1/metrics/{session_id}")
async def get_session_metrics(session_id: str, db: Session = Depends(get_db)):
    """
    Get metrics for a specific session.
    
    Returns metrics grouped by run_id and evaluator.
    
    Args:
        session_id: Session identifier
        db: Database session
    
    Returns:
        dict: Metrics organized by run_id and evaluator
    """
    try:
        from models import SessionRunMetrics
        
        # Get all metrics for this session
        metrics = db.query(SessionRunMetrics).filter(
            SessionRunMetrics.session_id == session_id
        ).all()
        
        if not metrics:
            return {
                "status": "success",
                "session_id": session_id,
                "message": "No metrics found for this session",
                "metrics": {}
            }
        
        # Organize metrics by run_id, evaluator, and mode (aggregate across all nodes)
        organized_metrics = {}
        
        for metric in metrics:
            run_id = metric.run_id
            evaluator = metric.evaluator
            mode = metric.mode
            
            if run_id not in organized_metrics:
                organized_metrics[run_id] = {}
            
            if evaluator not in organized_metrics[run_id]:
                organized_metrics[run_id][evaluator] = {}
            
            if mode not in organized_metrics[run_id][evaluator]:
                organized_metrics[run_id][evaluator][mode] = {
                    "correct_count": 0,
                    "total_count": 0,
                    "accuracy": 0.0
                }
            
            # Accumulate metrics for this mode (aggregate across all nodes)
            organized_metrics[run_id][evaluator][mode]["correct_count"] += metric.correct_count
            organized_metrics[run_id][evaluator][mode]["total_count"] += metric.total_count
        
        # Calculate accuracy for each mode
        for run_id, evaluators in organized_metrics.items():
            for evaluator, modes in evaluators.items():
                for mode, data in modes.items():
                    if data["total_count"] > 0:
                        data["accuracy"] = data["correct_count"] / data["total_count"]
                    else:
                        data["accuracy"] = 0.0
        
        return {
            "status": "success",
            "session_id": session_id,
            "metrics": organized_metrics
        }
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
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

