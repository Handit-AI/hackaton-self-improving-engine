# ACE (Adaptive Critical Experience) System Documentation

## Overview

The ACE system is a self-improving agent framework that learns from experience by generating, evaluating, and evolving actionable "bullets" (heuristics/rules) that improve agent performance. The system uses Darwin-Gödel evolutionary principles combined with LLM-based evaluation to continuously improve agent decision-making.

## Core Architecture

### Components

1. **BulletPlaybook** - Stores and manages bullets with database persistence
2. **HybridSelector** - Intelligent bullet selection using 5-stage algorithm
3. **Reflector** - Generates new bullets from agent failures
4. **Curator** - Deduplicates bullets using semantic similarity
5. **TrainingPipeline** - Orchestrates offline and online training
6. **DarwinBulletEvolver** - Evolves bullets using genetic programming
7. **PatternManager** - Classifies inputs and tracks bullet effectiveness per pattern
8. **LLMJudge** - Evaluates agent outputs using LLM-based judgment

## System Flow

### 1. Execution Phase

```
Input → Agent → Output
         ↓
    LLM Judge → Evaluation
```

**Agent Execution:**
1. Input is classified into a pattern category using `PatternManager`
2. Intelligent bullet selection using `HybridSelector`:
   - Gets bullets for each evaluator from database
   - Selects top bullets per evaluator using semantic similarity
   - Limits to max 10 bullets per evaluator
3. Agent executes with selected bullets
4. Output is evaluated by LLM Judge (or ground truth comparison)
5. Transaction is saved to database with pattern association

### 2. Bullet Generation Phase

```
Failure → Reflector → New Bullet → Curator → Playbook
                     ↓
              Darwin Evolution (if enabled)
```

**When bullets are generated:**
- Agent output is incorrect (wrong prediction)
- OR fewer than 5 bullets were used (coverage issue)

**Generation process:**
1. `Reflector` generates a new bullet from the failure
2. `Darwin-Gödel Evolution` triggers (if 2+ bullets exist):
   - Use last 6 bullets as parents (NOT tested)
   - Generate 4 candidates via crossover
   - Test ONLY the 5 newly generated bullets (new + 4 candidates) on transactions
   - Keep only top 2 bullets
3. `Curator` checks for duplicates (semantic similarity > 0.85)
4. If not duplicate, bullet is added to `Playbook`
5. Bullet is associated with evaluator and saved to database

### 3. Darwin-Gödel Evolution Phase

```
New Bullet → Crossover Parents (last 6 bullets) → Generate 4 Candidates
                                                    ↓
                                          Test 5 NEW Bullets Only
                                                    ↓
                                           Keep Top 2 Only
```

**Old bullets are NOT re-tested - only used as parents for crossover**

**Complete Evolution Flow:**

1. **Crossover Phase** (Genetic Generation):
   - Take last 6 bullets from playbook as parents
   - For each of 4 candidates:
     - Randomly select 2 different parents
     - Use LLM to intelligently merge them via crossover
     - Generate novel child bullet combining features from both parents
   
   Example:
   ```
   Parent 1: "New user (< 90 days) + large amount (> $1000) = 80% fraud"
   Parent 2: "VPN usage + crypto merchant = 95% fraud"
   ↓ (LLM crossover)
   Child: "New user (< 90 days) + VPN + crypto merchant + amount > $1000 = 97% fraud"
   ```

2. **Testing Phase** (Fitness Evaluation):
   - Take all 5 newly generated bullets (1 new + 4 candidates)
   - For each bullet:
     - Test on 5 random transactions from database
     - LLM Judge evaluates: "Would this bullet help with this transaction?"
     - Count how many transactions it would help with
     - Calculate fitness score (accuracy = helpful_count / 5)
   
   Example:
   ```
   Bullet: "New user + VPN + crypto = 97% fraud"
   
   Transaction 1: "New user from Nigeria using VPN, $500 crypto"
   → Judge: "YES, helpful" ✓
   
   Transaction 2: "Existing user buying groceries, $50"
   → Judge: "NO, not applicable" ✗
   
   Transaction 3: "New user + VPN + crypto, $2000"
   → Judge: "YES, helpful" ✓
   
   ... (2 more transactions)
   
   Fitness Score: 3/5 = 60%
   ```

3. **Selection Phase** (Survival of the Fittest):
   - Sort all 5 bullets by fitness score
   - Keep only top 2 bullets
   - Add them to playbook (if not duplicates)
   
   Example:
   ```
   Results:
   1. Bullet A: 80% fitness → Keep ✓
   2. Bullet B: 60% fitness → Keep ✓
   3. Bullet C: 40% fitness → Discard
   4. Bullet D: 20% fitness → Discard
   5. Bullet E: 20% fitness → Discard
   ```

**Evolution triggers:**
- EVERY time a new bullet is generated
- Requires at least 2 existing bullets for crossover
- Integrated into bullet generation process

**Evaluator-Specific Evolution:**
Bullets are generated **per evaluator** (not globally). This means:

1. **Get evaluator-specific parents**: Uses only bullets from the SAME evaluator
   ```python
   recent_bullets = playbook.get_bullets_for_node(node, evaluator=evaluator)
   ```

2. **Generate candidates**: Creates 4 candidates using only those evaluator-specific parents

3. **Test and select**: Tests all 5 bullets (new + 4 candidates) for this evaluator

4. **Add to evaluator**: Saves bullets with the same evaluator tag
   ```python
   self.curator.merge_bullet(
       content=bullet_content,
       node=node,
       playbook=playbook,
       source="evolution",
       evaluator=evaluator  # Same evaluator as parents
   )
   ```

**Example:**
```
Evaluator: "fraud_detection"
  ├─ Get last 6 bullets for "fraud_detection" evaluator
  ├─ Generate 4 candidates from those bullets
  ├─ Test all 5 bullets on transactions
  └─ Keep top 2 bullets → Add to "fraud_detection" evaluator

Evaluator: "risk_assessment"  
  ├─ Get last 6 bullets for "risk_assessment" evaluator
  ├─ Generate 4 candidates from those bullets
  ├─ Test all 5 bullets on transactions
  └─ Keep top 2 bullets → Add to "risk_assessment" evaluator
```

Each evaluator evolves independently!

**How Crossover Connects to Testing:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    CROSSOVER PHASE                               │
│  (Genetic Generation - Creates Novel Bullets)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Parent Pool (last 6 bullets)                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ Parent 1       │  │ Parent 2       │  │ ... (4 more)    │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  │
│           │                   │                    │           │
│           ├───────────────────┼────────────────────┤           │
│           │                   │                    │           │
│           ▼                   ▼                    ▼           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ Candidate 1    │  │ Candidate 2    │  │ Candidate 3    │  │
│  │ (LLM merged)  │  │ (LLM merged)  │  │ (LLM merged)  │  │
│  └────────────────┘  └────────────────┘  └────────────────┘  │
│                                                                    │
│  ┌────────────────┐                                             │
│  │ Candidate 4    │                                             │
│  │ (LLM merged)  │                                             │
│  └────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TESTING PHASE                                 │
│  (Fitness Evaluation - Tests Novel Bullets)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐                                             │
│  │ New Bullet     │──┐                                          │
│  └────────────────┘  │                                          │
│                      │                                          │
│  ┌────────────────┐  │     ┌────────────────────────────────┐  │
│  │ Candidate 1    │──┼─────│ Transaction 1: Nigeria + VPN   │  │
│  └────────────────┘  │     │ Judge: YES ✓                   │  │
│                      │     └────────────────────────────────┘  │
│  ┌────────────────┐  │                                          │
│  │ Candidate 2    │──┼─────│ Transaction 2: Groceries $50    │  │
│  └────────────────┘  │     │ Judge: NO ✗                     │  │
│                      │     └────────────────────────────────┘  │
│  ┌────────────────┐  │                                          │
│  │ Candidate 3    │──┼─────│ Transaction 3: Crypto $2000     │  │
│  └────────────────┘  │     │ Judge: YES ✓                   │  │
│                      │     └────────────────────────────────┘  │
│  ┌────────────────┐  │                                          │
│  │ Candidate 4    │──┼─────│ Transaction 4: ...             │  │
│  └────────────────┘  │     │ Judge: ...                      │  │
│                      │     └────────────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│     Fitness Scores: [80%, 60%, 40%, 20%, 20%]                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SELECTION PHASE                               │
│  (Survival of the Fittest)                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Sort by fitness → Keep top 2 only → Add to Playbook            │
│                                                                  │
│  ✓ Bullet A (80% fitness) → Added                                │
│  ✓ Bullet B (60% fitness) → Added                                │
│  ✗ Bullet C (40% fitness) → Discarded                            │
│  ✗ Bullet D (20% fitness) → Discarded                            │
│  ✗ Bullet E (20% fitness) → Discarded                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Evolution process:**
1. Generate new bullet from reflection
2. Get last 6 bullets from playbook (used ONLY as parents for crossover, NOT tested)
3. Generate 4 candidate bullets via crossover from those parents
4. Test ONLY the 5 newly generated bullets (new bullet + 4 candidates) on 5 random transactions from database
5. LLM Judge evaluates if each bullet would help with each transaction
6. Select top 2 bullets based on fitness scores
7. Add only the top 2 bullets to playbook

**Important:** Old bullets from the playbook are NOT re-tested. Evolution only tests newly generated bullets.

**Evolution parameters:**
- `n_samples`: 5 transactions
- `min_bullets_for_crossover`: 2
- `candidates_generated`: 4
- `bullets_kept`: 2
- `evaluator`: Uses LLM judge domain field for context

## Bullet Selection Algorithm (HybridSelector)

### 5-Stage Selection Process

#### Stage 1: Contextual Filtering
- Filter bullets by node (agent name)
- Filter by evaluator (perspective)
- Filter by source (offline/online)

#### Stage 2: Quality Filtering
- Minimum success rate threshold: 0.3
- Exclude bullets with poor performance

#### Stage 3: Semantic Filtering
- Generate embedding for input query
- Compute cosine similarity with bullet embeddings
- Minimum similarity threshold: 0.5
- Uses pre-calculated embeddings (cached in database)

#### Stage 4: Hybrid Scoring
Combines multiple factors:
- **Quality Score** (30%): Bullet's empirical success rate
- **Semantic Score** (40%): Cosine similarity to query
- **Thompson Sampling** (30%): Exploration-exploitation balance
- **Pattern Boost**: Additional boost if bullet is effective for this pattern

#### Stage 5: Diversity Promotion
- Promote diverse bullets
- Avoid redundant patterns
- Diversity weight: 0.15

## Database Schema

### Tables

**bullets**
- `id`: Unique bullet identifier
- `content`: Bullet text
- `node`: Agent node name
- `evaluator`: Evaluator/perspective name
- `content_embedding`: Pre-calculated 1536-dim embedding
- `helpful_count`: Times bullet helped
- `harmful_count`: Times bullet harmed
- `times_selected`: Times bullet was selected
- `source`: 'offline', 'online', or 'evolution'
- `created_at`, `last_used`: Timestamps

**transactions**
- `transaction_data`: JSON with systemprompt, userprompt, output, reasoning
- `mode`: 'vanilla', 'offline_online', 'online_only'
- `predicted_decision`: Agent's decision
- `correct_decision`: Ground truth
- `is_correct`: Boolean
- `input_pattern_id`: Pattern classification

**input_patterns**
- `pattern_summary`: LLM-generated summary
- `node`: Agent node
- `category`: Pattern category
- `features`: JSON with extracted features

**bullet_input_effectiveness**
- `pattern_id`: Input pattern
- `bullet_id`: Bullet
- `helpful_count`: Times bullet helped for this pattern
- `harmful_count`: Times bullet harmed for this pattern

**llm_judges**
- `node`: Agent node
- `model`: LLM model (default: gpt-4o-mini)
- `temperature`: 0.0
- `system_prompt`: Judge's system prompt
- `evaluation_criteria`: JSON array of criteria
- `domain`: Domain context (used for evaluator identification)

**judge_evaluations**
- `judge_id`: LLM judge
- `transaction_id`: Transaction evaluated
- `input_text`, `output_text`: Transaction data
- `ground_truth`: Ground truth (if available)
- `is_correct`: Judge's decision
- `confidence`: Confidence score
- `reasoning`: Judge's reasoning

**bullet_selections**
- `transaction_id`: Transaction
- `bullet_id`: Bullet selected
- `rank`: Selection rank
- `score`: Selection score

## Evaluator System

### Concept

Evaluators represent different perspectives or criteria for evaluating agent outputs:
- **Risk-focused**: Evaluates risk assessment correctness
- **Pattern-focused**: Evaluates pattern detection accuracy
- **Context-focused**: Evaluates contextual understanding

### Implementation

1. **LLM Judge Domain Field**: Identifies evaluator context
2. **Bullet Association**: Bullets are associated with evaluators
3. **Selection**: Bullets are selected per evaluator
4. **Evolution**: Evolution is evaluator-aware

### Prompt Structure

```
EVALUATOR1 Rules:
- bullet 1
- bullet 2

EVALUATOR2 Rules:
- bullet 1
- bullet 2
```

Each evaluator can have up to 10 bullets in the prompt.

## Training Modes

### 1. Vanilla Mode
- No bullets used
- Baseline performance
- Evaluated with ground truth

### 2. Offline + Online Mode
- Pre-train on training set (70%)
- Test on test set (30%)
- Evaluated with ground truth
- Bullets generated during training
- Bullets used during testing

### 3. Online Only Mode
- Real-time learning
- No pre-training
- Uses LLM Judge for evaluation
- Bullets generated on-the-fly

## Key Features

### 1. Pre-calculated Embeddings
- Embeddings generated when creating bullets
- Stored in database to avoid repeated API calls
- Used for semantic similarity search

### 2. Pattern-Based Effectiveness
- Inputs classified into patterns
- Bullet effectiveness tracked per pattern
- Pattern-specific bullet selection

### 3. Deduplication
- Semantic similarity threshold: 0.85
- Prevents duplicate bullets
- Saves context space

### 4. Database Persistence
- All bullets persisted to database
- Transactions saved for evolution
- Pattern classifications tracked
- Bullet effectiveness per pattern

### 5. Intelligent Bullet Selection
- Contextual filtering
- Quality filtering
- Semantic filtering
- Hybrid scoring
- Diversity promotion

## Performance Optimizations

### 1. Fast Evolution
- **Crossover**: 4 candidates generated via LLM crossover (4 API calls)
- **Testing**: 5 bullets × 5 transactions = 25 LLM judge calls
- **Selection**: Keep top 2 bullets only
- **Total**: ~29 API calls per evolution cycle
- Reduced from 60+ to ~29 API calls with same quality

### 2. Embedding Caching
- Pre-calculated embeddings in database
- No on-the-fly generation
- Faster semantic similarity

### 3. Pattern-Based Caching
- Pattern classifications cached
- Bullet effectiveness per pattern cached
- Faster retrieval

### 4. Limited Bullet Selection
- Max 10 bullets per evaluator
- Hyperparameter: `max_bullets` (default: 5)
- Prevents context rot

## Example Workflow

### Scenario: Fraud Detection Agent

1. **Input**: "New user from Nigeria using VPN, $500 crypto purchase"
2. **Pattern Classification**: `pattern_id=5, confidence=0.92`
3. **Bullet Selection**: 
   - Evaluator: "fraud_detection"
   - Selected 5 bullets matching input
4. **Agent Execution**: Returns "DECLINE"
5. **Evaluation**: Ground truth says "DECLINE" → Correct
6. **Pattern Tracking**: Update bullet effectiveness for pattern_id=5
7. **Transaction Saved**: Full transaction data persisted

### Scenario: Bullet Generation

1. **Agent Error**: Predicted "APPROVE" but should be "DECLINE"
2. **Reflection**: Reflector generates new bullet
3. **Evolution**: 
   - Generate 4 candidates via crossover from last 6 bullets
   - Test all 5 bullets on 5 random transactions
   - LLM Judge evaluates fitness
   - Keep top 2 bullets
4. **Curator Check**: Not duplicate (similarity 0.72)
5. **Add to Playbook**: Top 2 bullets saved with evaluator="fraud_detection"

## API Endpoints

### POST /api/v1/analyze
Analyze transaction with selected bullets

### POST /api/v1/evaluate
Evaluate agent output with LLM Judge

### POST /api/v1/get-bullets
Get bullets for a node (with optional query for intelligent selection)

### GET /api/v1/playbook/stats
Get playbook statistics

### GET /api/v1/playbook/{node}
Get bullets for a specific node

### POST /api/v1/test/comprehensive
Run comprehensive test suite

## Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `OPENAI_API_KEY`: OpenAI API key
- `DEBUG`: Debug mode (default: False)
- `LOG_LEVEL`: Logging level (default: INFO)

### Hyperparameters

- `max_bullets`: Maximum bullets per evaluator (default: 10)
- `quality_threshold`: Minimum success rate (default: 0.3)
- `semantic_threshold`: Minimum similarity (default: 0.5)
- `diversity_weight`: Diversity promotion weight (default: 0.15)
- `similarity_threshold`: Deduplication threshold (default: 0.85)

## Design Decisions

### 1. Why Darwin-Gödel Evolution?

- Genetic programming allows novel combinations
- Testing on real transactions ensures fitness
- Top performers survive, weak ones die

### 2. Why Per-Evaluator Bullets?

- Different perspectives need different rules
- Prevents confusion between criteria
- More targeted improvements

### 3. Why Pre-calculated Embeddings?

- Performance: No repeated API calls
- Cost: Reduce API usage
- Consistency: Same embeddings over time

### 4. Why Pattern-Based Selection?

- Different inputs need different bullets
- Pattern-specific effectiveness tracking
- More intelligent selection

### 5. Why Evolution on Every Generation?

- Tests new bullets immediately against real transactions
- Ensures only the best bullets survive and get added
- Prevents accumulation of weak bullets
- Creates competitive pressure for improvement
- Fast enough (5 transactions × 5 bullets = 25 API calls)

## Future Improvements

1. **Multi-Objective Evolution**: Evolve for multiple criteria simultaneously
2. **Adaptive Thresholds**: Automatically adjust thresholds based on performance
3. **Bullet Hierarchies**: Organize bullets into hierarchies
4. **Explainability**: Track why bullets were selected
5. **A/B Testing**: Compare different bullet strategies

## References

- Darwin-Gödel Machines: https://arxiv.org/abs/1509.08784
- Thompson Sampling: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
- Semantic Similarity: https://en.wikipedia.org/wiki/Semantic_similarity

