# LLM Judge Database Configuration

## Overview

LLM judges are now configurable per node via the database, allowing dynamic configuration without code changes.

---

## Database Models

### **LLMJudge Model**

Stores judge configuration per node:

```python
class LLMJudge(Base):
    node = Column(String(100), unique=True)  # Which agent node
    model = Column(String(50), default='gpt-4o-mini')  # Model to use
    temperature = Column(Float, default=0.0)  # Temperature for judging
    
    # Prompt configuration
    system_prompt = Column(Text)  # System prompt for judge
    evaluation_criteria = Column(JSONB)  # Specific criteria to evaluate
    
    # Domain context
    domain = Column(String(100), default='fraud detection')
    
    # Performance tracking
    total_evaluations = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    
    # Metadata
    is_active = Column(Boolean, default=True)
```

### **JudgeEvaluation Model**

Audit trail for all judge evaluations:

```python
class JudgeEvaluation(Base):
    judge_id = Column(Integer, ForeignKey("llm_judges.id"))
    transaction_id = Column(Integer, ForeignKey("transactions.id"))
    
    # Evaluation data
    input_text = Column(Text)
    output_text = Column(Text)
    ground_truth = Column(Text)
    
    # Judge results
    is_correct = Column(Boolean)
    confidence = Column(Float)
    reasoning = Column(Text)
    
    # Judge accuracy
    judge_was_correct = Column(Boolean)
```

---

## Features

### 1. **Per-Node Configuration**
Each agent node can have its own judge configuration:
- Different models (GPT-3.5-turbo, GPT-4o, GPT-4o-mini)
- Custom system prompts
- Evaluation criteria
- Temperature settings

### 2. **Dynamic Configuration**
- No code changes needed
- Update database to change judge behavior
- Cached for performance

### 3. **Audit Trail**
- All evaluations saved to database
- Track judge performance
- Analyze judge accuracy

### 4. **Fallback to Defaults**
- If no config in DB, uses sensible defaults
- Graceful degradation

---

## Usage Examples

### **Configure Judge for Node**

```python
from models import LLMJudge
from sqlalchemy.orm import Session

# Create judge configuration
judge = LLMJudge(
    node="fraud_detection",
    model="gpt-4o-mini",
    temperature=0.0,
    system_prompt="You are an expert fraud detection judge with 10 years of experience.",
    evaluation_criteria=[
        "Check if decision aligns with risk factors",
        "Verify reasoning is sound",
        "Ensure no false positives/negatives"
    ],
    domain="fraud detection",
    is_active=True
)

db_session.add(judge)
db_session.commit()
```

### **Use Judge**

```python
from ace.llm_judge import LLMJudge

# Initialize with database session
judge = LLMJudge(db_session=db)

# Judge an output
is_correct, confidence = await judge.judge(
    input_text="Transaction: User 'john' attempts $2500 purchase...",
    output="DECLINE",
    ground_truth="DECLINE",
    node="fraud_detection",
    save_to_db=True  # Save evaluation to database
)
```

### **Batch Judging**

```python
evaluations = [
    {"input": "query1", "output": "APPROVE", "ground_truth": "APPROVE"},
    {"input": "query2", "output": "DECLINE", "ground_truth": "APPROVE"},
]

results = await judge.judge_batch(evaluations, node="fraud_detection")
# Returns: [(True, 1.0), (False, 0.0)]
```

---

## Configuration Example

### **Strict Judge (Fewer False Positives)**

```python
judge = LLMJudge(
    node="fraud_detection",
    model="gpt-4o-mini",
    temperature=0.0,
    system_prompt="You are a conservative fraud detection judge. Only approve when very confident.",
    evaluation_criteria=[
        "Require strong evidence for fraud",
        "Prefer false negatives over false positives",
        "Question high-value transactions"
    ],
    domain="fraud detection"
)
```

### **Lenient Judge (Fewer False Negatives)**

```python
judge = LLMJudge(
    node="fraud_detection",
    model="gpt-3.5-turbo",
    temperature=0.3,
    system_prompt="You are a balanced fraud detection judge.",
    evaluation_criteria=[
        "Balance fraud prevention with user experience",
        "Consider transaction context",
        "Allow legitimate patterns"
    ],
    domain="fraud detection"
)
```

---

## How It Works

### **1. Judge Initialization**

```python
judge = LLMJudge(db_session=db_session)
```

The judge loads configurations from the database and caches them.

### **2. Configuration Loading**

```python
def _get_judge_config(self, node: str) -> dict:
    # Check cache first
    if node in self._judge_cache:
        return self._judge_cache[node]
    
    # Load from database
    judge_config = db_session.query(LLMJudge).filter(
        LLMJudge.node == node,
        LLMJudge.is_active == True
    ).first()
    
    # Return config or defaults
    return config or default_config
```

### **3. Evaluation**

```python
async def judge(self, input_text, output, ground_truth, node, save_to_db):
    # Get config for this node
    config = self._get_judge_config(node)
    
    # Use configured model, temperature, prompt
    response = self.client.chat.completions.create(
        model=config['model'],
        temperature=config['temperature'],
        messages=[
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": f"Input: {input_text}..."}
        ]
    )
    
    # Save to database if enabled
    if save_to_db:
        # Create JudgeEvaluation record
        # Update judge stats
```

---

## Benefits

### **1. Dynamic Configuration**
- Change judge behavior without code deployment
- A/B test different configurations
- Adjust based on performance

### **2. Per-Node Customization**
- Different nodes can have different judges
- Customize for specific use cases
- Optimize for node requirements

### **3. Audit Trail**
- Track all evaluations
- Analyze judge performance
- Debug issues

### **4. Performance Tracking**
- Monitor judge accuracy
- Identify improvements
- Compare configurations

---

## Migration

You'll need to create a migration for the new tables:

```bash
alembic revision --autogenerate -m "add_llm_judge_tables"
alembic upgrade head
```

---

## Default Configuration

If no configuration exists in the database, the system uses:

```python
default_config = {
    'model': 'gpt-4o-mini',
    'temperature': 0.0,
    'system_prompt': f"You are an expert judge evaluating fraud detection outputs for {node}.",
    'evaluation_criteria': None,
    'domain': 'fraud detection'
}
```

---

## Answer to Your Question

### **Does bullet generation use evaluation?**

**YES!** The Reflector uses correctness information:

```python
# In ace/reflector.py
async def reflect(self, query, predicted, correct, node, agent_reasoning):
    is_correct = (predicted == correct)  # ← Evaluates correctness
    
    prompt = f"""You are a fraud detection expert learning from {'success' if is_correct else 'failure'}.
    # ↑ Prompt changes based on if it was correct or not
```

**Flow**:
1. Agent makes prediction
2. Judge evaluates correctness
3. Reflector generates bullet based on success/failure
4. Curator adds bullet (if not duplicate)
5. Bullet gets used in future selections

**Success** → Reinforce good patterns  
**Failure** → Generate corrective bullets

