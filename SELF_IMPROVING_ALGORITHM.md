# ACE Self-Improving Algorithm Documentation

## Overview

The ACE (Adaptive Context Enhancement) system is a self-improving framework that learns from feedback to continuously enhance AI agent performance. It uses a combination of LLM judges, reflective learning, and evolutionary algorithms to generate actionable heuristics (called "bullets") that guide agent behavior.

## Core Philosophy

The system learns from mistakes and successes by:
1. **Evaluating** outputs with specialized LLM judges
2. **Reflecting** on failures to generate generalized heuristics
3. **Evolving** heuristics through crossover and fitness testing
4. **Applying** heuristics as context in future prompts

---

## System Architecture

### Components

```
┌─────────────────┐
│  LLM Judges     │  ← Evaluate outputs from multiple perspectives
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reflector     │  ← Generate bullets from judge feedback
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Curator       │  ← Deduplicate and quality control
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Darwin-Evolver  │  ← Evolve bullets through crossover
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Bullet Playbook │  ← Store and retrieve bullets
└─────────────────┘
```

---

## The Learning Loop

### Step 1: Transaction Processing

When a transaction is processed:

```python
POST /api/v1/trace
{
  "node": "decision_aggregator",
  "input_text": "...",
  "output": "...",
  "agent_reasoning": "...",
  "ground_truth": "..."
}
```

### Step 2: Multi-Evaluator Assessment

Each evaluator (specialized LLM judge) assesses the output:

**Example Evaluators:**
- `format_compliance`: Checks JSON structure, field types, case sensitivity
- `logic_consistency`: Verifies decision logic matches analyzer outputs
- `risk_synthesis`: Ensures holistic risk assessment

**Judge Response:**
```json
{
  "is_correct": false,
  "confidence": 0.85,
  "reasoning": "The output format is incorrect. Specifically: The final_decision field contains 'approve' which is invalid due to case mismatch."
}
```

### Step 3: Bullet Generation (Reflection)

For each evaluator, the system generates a bullet using the reflector:

**Input to Reflector:**
- Query: Transaction input
- Predicted: Agent's output
- Correct: Ground truth or judge's corrected output
- Agent Reasoning: Agent's explanation
- **Judge Reasoning: The specific issue identified by the judge**

**Reflector Prompt Structure:**
```
PRIMARY FOCUS - The LLM Judge has identified this specific issue:
{judge_reasoning}

CRITICAL CONTEXT: This bullet will be used as HEURISTIC/CONTEXT in prompts 
to guide the LLM's behavior. It must be written as a practical rule or 
instruction that tells the LLM how to behave when generating outputs.

CRITICAL REQUIREMENTS:
- The bullet MUST be written as a HEURISTIC/INSTRUCTION for the LLM agent
- The bullet MUST address the GENERAL PATTERN behind the judge's concern
- The bullet MUST be generalized to catch similar variations
- The bullet MUST be actionable guidance
```

**Example Generated Bullet:**
```
"When outputting final_decision field, ensure it uses EXACTLY uppercase 
versions ('APPROVE', 'REVIEW', 'DECLINE'), not lowercase or mixed case variants"
```

### Step 4: Curator (Deduplication)

The curator checks for duplicates:
- Compares semantic similarity (threshold: 0.85)
- Filters out near-duplicates
- Only adds truly novel bullets

### Step 5: Darwin-Gödel Evolution (Optional)

If enabled (`ENABLE_DARWIN_EVOLUTION=true`):

1. **Generate Candidates**: Create 4 new bullets via crossover
2. **Fitness Evaluation**: Test all bullets on saved transactions
3. **Selection**: Keep only the top-performing bullet

**Crossover Process:**
```python
Parent 1: "For DECLINE decisions, require 3/5 analyzers indicating high risk"
Parent 2: "For DECLINE decisions, cite specific analyzer findings"

↓ Crossover

Child: "For DECLINE decisions, require at least 3/5 analyzers indicating 
high risk AND cite specific analyzer findings in your reasoning"
```

### Step 6: Storage & Retrieval

Bullets are stored in the database with:
- `id`: Unique identifier
- `content`: The heuristic text
- `node`: Which agent node uses it
- `evaluator`: Which evaluator generated it
- `source`: 'offline', 'online', or 'evolution'
- `helpful_count`: Times it helped
- `harmful_count`: Times it hurt
- `times_selected`: Times it was used

---

## Bullet Usage as Context

### How Bullets Are Used

Bullets are retrieved and inserted into agent prompts as context:

```python
# Select relevant bullets
bullets = selector.select_bullets(
    query=transaction_input,
    node="decision_aggregator",
    n_bullets=5,
    evaluator="format_compliance"
)

# Insert into prompt
prompt = f"""
You are a fraud detection expert.

Heuristics to follow:
{chr(10).join([b.content for b in bullets])}

Transaction: {transaction_input}
...
"""
```

### Selection Strategy

The system uses **Hybrid Selection**:

1. **Semantic Search** (40%): Find bullets similar to the query
2. **Quality Score** (30%): Based on helpful_count/harmful_count ratio
3. **Thompson Sampling** (30%): Balance exploration vs exploitation
4. **Pattern Boost**: Additional boost if bullet is effective for this pattern

### Example Selection

For a transaction about "unknown merchant":

```python
Selected Bullets:
1. "When determining the final_decision field, ensure it strictly matches 
   the required values of 'APPROVE', 'REVIEW', or 'DECLINE'..." 
   (semantic similarity: 0.6, quality: 0.85, boost: 0.1)

2. "For emergency situations (hospitals, pharmacies), bias toward 
   APPROVE/REVIEW..." 
   (semantic similarity: 0.4, quality: 0.92, boost: 0.05)

3. "When outputting recommendations field, ensure it's a proper JSON array..."
   (semantic similarity: 0.3, quality: 0.78, boost: 0.0)
```

---

## Evaluator System

### Multiple Perspectives

Each node has multiple evaluators, each focusing on different aspects:

**Decision Aggregator Evaluators:**

1. **format_compliance**
   - Checks JSON structure
   - Validates field types
   - Ensures case sensitivity
   - Verifies array formats

2. **logic_consistency**
   - Validates decision logic
   - Checks analyzer consensus
   - Verifies weighted risk assessment
   - Ensures decision-reasoning alignment

3. **risk_synthesis**
   - Evaluates holistic risk synthesis
   - Checks emergency situation handling
   - Validates life transition recognition
   - Ensures business impact consideration

**Behavioral Analyzer Evaluators:**

1. **pattern_consistency**: Spending patterns, merchant categories, time patterns
2. **temporal_anomaly**: Time-of-day anomalies, velocity spikes, sequential patterns
3. **profile_evolution**: Life events vs fraud, gradual vs sudden changes

### Why Multiple Evaluators?

- **Specialized Feedback**: Each evaluator focuses on one aspect
- **Targeted Bullets**: Bullets address specific problems
- **Reduced Noise**: Format issues don't mix with logic issues
- **Better Tracking**: Can measure improvement per aspect

---

## Online vs Offline Learning

### Offline Learning

Pre-training on historical data:
- Bullets marked with `source='offline'`
- Generated from training sets
- Broad coverage of common patterns
- Static set (doesn't change during operation)

**Example Offline Bullets:**
```python
# Migration inserts 18 offline bullets
- Decision Aggregator: 9 bullets (3 per evaluator)
- Behavioral Analyzer: 9 bullets (3 per evaluator)
```

### Online Learning

Real-time learning from production:
- Bullets marked with `source='online'`
- Generated from actual transactions
- Adaptive to current issues
- Continuously updated

### Evolution

Advanced optimization:
- Bullets marked with `source='evolution'`
- Generated through Darwin-Gödel process
- Tested on fitness function
- Best performers selected

---

## Performance Tracking

### Bullet Effectiveness

Each bullet tracks:
- `helpful_count`: Times it led to correct output
- `harmful_count`: Times it led to incorrect output
- `times_selected`: How often it was chosen

**Success Rate Calculation:**
```python
def get_success_rate(self) -> float:
    total = self.helpful_count + self.harmful_count
    if total == 0:
        return 0.5  # Neutral prior
    return self.helpful_count / total
```

### Pattern-Bullet Association

Tracks which bullets work for which patterns:
- `bullet_input_effectiveness` table
- Links bullets to input patterns
- Used for pattern-specific boosting

---

## The Complete Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Transaction Input                                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Agent Processes with Current Bullets                │
│    - Semantic search for relevant bullets               │
│    - Quality score + Thompson sampling                  │
│    - Insert bullets as context into prompt              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Agent Generates Output                               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Multi-Evaluator Assessment                           │
│    - format_compliance: "approve should be APPROVE"    │
│    - logic_consistency: "needs analyzer support"       │
│    - risk_synthesis: "bias toward APPROVE for emergency"│
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 5. For Each Evaluator:                                  │
│    │                                                     │
│    ├─→ Reflector generates bullet                      │
│    │   "When outputting final_decision, use uppercase" │
│    │                                                     │
│    ├─→ Curator checks for duplicates                   │
│    │                                                     │
│    ├─→ Darwin-Evolver (optional)                        │
│    │   - Generate 4 candidates                         │
│    │   - Test on transactions                           │
│    │   - Keep best performer                            │
│    │                                                     │
│    └─→ Store in Bullet Playbook                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 6. Update Performance Metrics                           │
│    - helpful_count/harmful_count                        │
│    - pattern-bullet associations                       │
│    - evaluator-specific accuracy                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 7. Future Transactions Use Updated Bullets             │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Generalization Over Specificity

Bullets must be generalized patterns, not hardcoded values:

❌ **Bad**: "Ensure final_decision is set to 'REVIEW' exactly"
✅ **Good**: "When outputting final_decision, ensure it matches required values in uppercase"

### 2. Judge-Driven Generation

Bullets MUST address the judge's specific concern:

```
Judge: "format issue - final_decision is 'approve' instead of 'APPROVE'"
→ Bullet: "When outputting final_decision, use EXACTLY uppercase versions..."
```

### 3. Instruction Format

Bullets are instructions for the LLM:

❌ **Code**: `if value == 'review': return 'REVIEW'`
✅ **Instruction**: "When outputting final_decision, use uppercase versions..."

### 4. Evaluator Isolation

Each evaluator generates its own bullets:
- `format_compliance` → Format-related bullets
- `logic_consistency` → Logic-related bullets
- `risk_synthesis` → Risk-related bullets

### 5. Continuous Learning

Every transaction is a learning opportunity:
- Success → Track which bullets helped
- Failure → Generate new bullets
- Evolution → Optimize bullet effectiveness

---

## Example: Complete Learning Cycle

### Transaction

```json
{
  "input": "Transaction with unknown merchant",
  "output": "{'final_decision': 'approve', ...}",
  "ground_truth": "{'final_decision': 'APPROVE', ...}"
}
```

### Judge Evaluation

**format_compliance judge:**
```json
{
  "is_correct": false,
  "reasoning": "The final_decision field contains 'approve' which is invalid due to case mismatch"
}
```

### Bullet Generation

**Reflector Prompt:**
```
PRIMARY FOCUS - The LLM Judge has identified this specific issue:
The final_decision field contains 'approve' which is invalid due to case mismatch

Generate a heuristic that addresses the pattern behind this issue...
```

**Generated Bullet:**
```
"When outputting final_decision field, ensure it uses EXACTLY uppercase 
versions ('APPROVE', 'REVIEW', 'DECLINE'), not lowercase or mixed case variants"
```

### Storage

```python
{
  "bullet_id": "decision_aggregator_format_a1b2c3d4",
  "content": "When outputting final_decision field...",
  "node": "decision_aggregator",
  "evaluator": "format_compliance",
  "source": "online"
}
```

### Future Usage

Next similar transaction:
```python
# Retrieves bullet
bullet = "When outputting final_decision field, ensure it uses EXACTLY uppercase versions..."

# Inserts into prompt
prompt = """
Heuristics to follow:
- When outputting final_decision field, ensure it uses EXACTLY uppercase versions...

Transaction: {transaction_input}
"""
```

### Result

Agent now outputs: `{'final_decision': 'APPROVE', ...}` ✅

---

## Configuration

### Environment Variables

```bash
# Enable Darwin-Gödel evolution
ENABLE_DARWIN_EVOLUTION=true

# OpenAI API key
OPENAI_API_KEY=sk-...

# Database connection
DATABASE_URL=postgresql://...
```

### Database Schema

```sql
-- Bullets table
CREATE TABLE bullets (
    id VARCHAR(50) PRIMARY KEY,
    content TEXT NOT NULL,
    node VARCHAR(100) NOT NULL,
    evaluator VARCHAR(100),
    content_embedding ARRAY(FLOAT),
    helpful_count INTEGER DEFAULT 0,
    harmful_count INTEGER DEFAULT 0,
    times_selected INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    last_used TIMESTAMP,
    source VARCHAR(20) -- 'offline', 'online', 'evolution'
);

-- LLM Judges table
CREATE TABLE llm_judges (
    id SERIAL PRIMARY KEY,
    node VARCHAR(100) NOT NULL,
    evaluator VARCHAR(100) NOT NULL,
    model VARCHAR(50) DEFAULT 'gpt-4o-mini',
    temperature FLOAT DEFAULT 0.1,
    system_prompt TEXT NOT NULL,
    evaluation_criteria JSONB,
    domain VARCHAR(100),
    UNIQUE(node, evaluator)
);
```

---

## Key Flows

### Flow 1: Transaction Processing with Bullets

**Purpose**: How bullets are retrieved and used during transaction processing

**Steps**:
1. Transaction arrives with query text
2. Semantic search finds relevant bullets (based on query embedding)
3. Quality score calculated (helpful_count / harmful_count)
4. Thompson sampling balances exploration vs exploitation
5. Pattern boost applied if bullet is effective for this pattern type
6. Top N bullets selected (usually 5)
7. Bullets inserted as context into agent prompt
8. Agent generates output following bullet instructions

**Key Components**:
- `HybridSelector`: Selects bullets using hybrid scoring
- `BulletPlaybook`: Stores and retrieves bullets
- Agent prompt: Contains bullets as heuristics

### Flow 2: Multi-Evaluator Assessment

**Purpose**: How the system evaluates outputs from multiple perspectives

**Steps**:
1. Agent generates output
2. For each evaluator in the node:
   - Load evaluator's LLM judge (with system prompt and criteria)
   - Call LLM judge with output and ground truth (if available)
   - Judge returns: is_correct, confidence, reasoning
3. Collect all evaluator assessments
4. Determine overall correctness (use ground truth if available, else judge consensus)
5. Save judge evaluations to database

**Key Components**:
- `LLMJudge`: Specialized evaluator with system prompt
- `JudgeEvaluation`: Records evaluator's assessment
- Multiple evaluators per node (e.g., format_compliance, logic_consistency)

### Flow 3: Bullet Generation from Reflection

**Purpose**: How bullets are generated from judge feedback

**Steps**:
1. For each evaluator that provided feedback:
   - Extract judge's reasoning (the specific issue identified)
   - Call Reflector with:
     - Query: Original transaction input
     - Predicted: Agent's output
     - Correct: Ground truth or corrected output
     - Agent Reasoning: Agent's explanation
     - **Judge Reasoning**: The specific issue
   - Reflector generates bullet addressing the issue
2. Bullet is a generalized instruction (not hardcoded to one value)
3. Bullet is written as LLM directive (not code)
4. Bullet stored with evaluator identifier

**Key Components**:
- `Reflector`: Generates bullets from feedback
- Focus on judge's specific concern
- Generalization over specificity

### Flow 4: Darwin-Gödel Evolution (Optional)

**Purpose**: How bullets evolve through genetic programming

**Steps**:
1. New bullet generated from reflection
2. Check if evolution is enabled (`ENABLE_DARWIN_EVOLUTION=true`)
3. If enabled:
   - Load recent bullets from playbook (last 6)
   - Generate 4 candidate bullets via crossover
   - Crossover combines two parent bullets intelligently
   - Test all 5 bullets (original + 4 candidates) on saved transactions
   - Calculate fitness score for each bullet
   - Keep only the top-performing bullet
4. Store evolved bullet with `source='evolution'`

**Key Components**:
- `DarwinBulletEvolver`: Handles evolution process
- `_generate_candidates`: Creates crossover children
- `_evaluate_bullets`: Tests fitness on transactions
- Fitness = accuracy on test transactions

### Flow 5: Curator Deduplication

**Purpose**: Prevents duplicate bullets from being added

**Steps**:
1. New bullet generated
2. Compare with existing bullets for same node+evaluator
3. Calculate semantic similarity using SequenceMatcher
4. If similarity > threshold (0.85), skip adding
5. If novel, add to playbook

**Key Components**:
- `Curator`: Handles deduplication
- Similarity threshold: 0.85
- Per-node and per-evaluator deduplication

### Flow 6: Performance Tracking

**Purpose**: Tracks bullet effectiveness over time

**Steps**:
1. After transaction processing:
   - Determine if output was correct
   - For each bullet used:
     - If correct: increment helpful_count
     - If incorrect: increment harmful_count
     - Increment times_selected
2. Update pattern-bullet associations:
   - Track which bullets work for which input patterns
   - Build pattern-specific effectiveness metrics
3. Calculate success rates:
   - helpful_count / (helpful_count + harmful_count)
4. Use success rates for future bullet selection

**Key Components**:
- `Bullet`: Stores helpful_count, harmful_count, times_selected
- `BulletInputEffectiveness`: Tracks pattern-specific effectiveness
- Success rate used in quality scoring

### Flow 7: Pattern Classification

**Purpose**: Groups similar transactions for pattern-specific learning

**Steps**:
1. Transaction arrives with input text
2. Generate embedding for input (1536 dimensions)
3. Semantic search for similar past transactions
4. If similarity > threshold, assign to existing pattern
5. If novel, create new pattern category
6. Save input pattern record
7. Link transaction to pattern

**Key Components**:
- `PatternManager`: Handles pattern classification
- `InputPattern`: Stores pattern summaries and embeddings
- Semantic similarity for pattern matching

---

## Benefits

### 1. Continuous Improvement

System learns from every transaction, not just initial training.

### 2. Specialized Feedback

Multiple evaluators provide targeted feedback for different aspects.

### 3. Generalization

Bullets are generalized patterns that help with similar cases.

### 4. Performance Tracking

Quantify which bullets actually help vs hurt.

### 5. Evolution

Darwin-Gödel process optimizes bullets through testing.

### 6. Explainability

Clear heuristics show what the system learned.

---

## Best Practices

### For Bullet Generation

1. **Focus on judge feedback** - Address the specific issue
2. **Generalize** - Catch patterns, not single values
3. **Write as instructions** - LLM directives, not code
4. **Make measurable** - Include thresholds, conditions, patterns

### For System Operation

1. **Start with offline bullets** - Pre-populate with common patterns
2. **Enable evolution** - Let bullets optimize over time
3. **Monitor effectiveness** - Track helpful_count vs harmful_count
4. **Adjust evaluators** - Fine-tune judges for your domain

---

## Future Enhancements

### Planned Features

1. **Multi-bullet synthesis** - Combine multiple bullets into one
2. **Temporal decay** - Age out old bullets that are no longer relevant
3. **Cross-node learning** - Transfer bullets between similar nodes
4. **Human-in-the-loop** - Manual review and approval of bullets
5. **A/B testing** - Compare bullet effectiveness versions

---

## Conclusion

The ACE self-improving algorithm provides a robust framework for continuous learning:

- **Learns from feedback** through specialized evaluators
- **Generates generalizable** heuristics (bullets)
- **Evolves** through Darwin-Gödel optimization
- **Applies** heuristics as context in prompts
- **Tracks** effectiveness for continuous improvement

By combining reflection, curation, and evolution, the system continuously adapts to new patterns and improves performance over time.

