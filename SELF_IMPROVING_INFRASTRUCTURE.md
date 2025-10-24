# Self-Improving Process Infrastructure

## Overview

The Self-Improving Engine implements a dual-mode learning system that operates in **Offline** and **Online** modes. The system continuously generates, evolves, and applies "bullets" (heuristic rules) to improve agent performance.

---

## Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Core Infrastructure                      │
├─────────────────────────────────────────────────────────────┤
│  • PatternManager: Input classification & pattern matching   │
│  • BulletPlaybook: Bullet storage & retrieval               │
│  • HybridSelector: 5-stage intelligent bullet selection      │
│  • Reflector: Generate bullets from failures                │
│  • Curator: Deduplication & quality control                 │
│  • DarwinBulletEvolver: Genetic optimization                │
│  • LLMJudge: Multi-evaluator assessment                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Mode Overview

### Mode 1: Vanilla Mode
- **No bullets** - Baseline performance
- Agent makes decisions without any learned rules
- Evaluated with ground truth

### Mode 2: Offline + Online Mode
- **Pre-trained** on historical data (bullets with `source='offline'`)
- **Combines** offline + online bullets during execution
- Uses ground truth for evaluation

### Mode 3: Online Only Mode
- **Real-time learning** from production transactions
- Only uses bullets generated during operation (`source='online'`)
- Uses LLM Judge for evaluation

---

## Data Entry Flow

### Input Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   1. Data Entry Point                        │
│                         POST /api/v1/trace                   │
├─────────────────────────────────────────────────────────────┤
│  Input Data:                                                  │
│  • input_text: Transaction query                             │
│  • node: Agent node identifier                               │
│  • output: Agent's decision                                  │
│  • ground_truth: Correct answer (optional)                    │
│  • agent_reasoning: Agent's explanation                      │
│  • model_type: Mode identifier ("vanilla", "full", "online")│
│  • session_id: Session identifier                            │
│  • run_id: Run identifier                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   2. Pattern Classification                   │
├─────────────────────────────────────────────────────────────┤
│  • PatternManager extracts features from input               │
│  • Generates embedding for semantic matching                 │
│  • Finds existing similar pattern OR creates new pattern     │
│  • Returns: pattern_id, confidence                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   3. Transaction Storage                      │
├─────────────────────────────────────────────────────────────┤
│  • Save transaction to database with pattern_id               │
│  • Mode: vanilla, offline_online, or online_only              │
│  • Initial is_correct = False (evaluated later)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   4. Evaluator Retrieval                      │
├─────────────────────────────────────────────────────────────┤
│  • Query LLM judges table for this node                      │
│  • Get all active evaluators (e.g., format_compliance,       │
│    logic_consistency, risk_synthesis)                        │
│  • Each evaluator = specialized LLM judge                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌────────────────────────────────┐
         │  For Each Evaluator:           │
         └───────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│             5. Multi-Evaluator Assessment (LLM Judge)         │
├─────────────────────────────────────────────────────────────┤
│  LLM Judge Evaluation:                                        │
│  • Load evaluator's system prompt & criteria                │
│  • Call LLM with: input_text, output, ground_truth           │
│  • Judge returns:                                             │
│    - is_correct: boolean                                      │
│    - confidence: 0.0-1.0                                      │
│    - reasoning: specific issue identified                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│        6. Bullet Generation (Mode-Dependent)                 │
├─────────────────────────────────────────────────────────────┤
│  IF mode != "vanilla":                                        │
│    For each evaluator:                                        │
│      • Reflector generates bullet from judge reasoning        │
│      • Darwin evolution (if enabled):                        │
│        - Generate 4 candidates via crossover                  │
│        - Test all 5 bullets on transactions                   │
│        - Keep only best bullet                                │
│      • Curator checks for duplicates                          │
│      • Add to playbook (source='online')                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             7. Bullet Effectiveness Tracking                   │
├─────────────────────────────────────────────────────────────┤
│  • Update helpful_count/harmful_count for used bullets        │
│  • Track pattern-bullet associations                         │
│  • Record in bullet_input_effectiveness table                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             8. Session Metrics Update                         │
├─────────────────────────────────────────────────────────────┤
│  • Update SessionRunMetrics for each evaluator               │
│  • Track accuracy per evaluator per mode                     │
│  • Increment correct_count/total_count                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     9. Response                               │
├─────────────────────────────────────────────────────────────┤
│  Return:                                                      │
│  • transaction_id: Database ID                                │
│  • pattern_id: Pattern classification                         │
│  • is_correct: Evaluation result                             │
│  • generated_bullets: List of new bullets                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Offline Mode Infrastructure

### Purpose
Pre-train bullets on historical data before production deployment.

### Endpoint
`POST /api/v1/train`

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Offline Training Infrastructure                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Dataset with historical transactions                 │
│                                                               │
│  For each training example:                                  │
│                                                               │
│  1. Pattern Classification                                    │
│     • PatternManager.classify_input_to_category()            │
│     • Find or create pattern                                 │
│                                                               │
│  2. Bullet Generation                                        │
│     • TrainingPipeline.add_bullet_from_reflection()          │
│     • Reflector generates bullet from example                │
│     • Darwin evolution (if enabled):                         │
│       - Use last 6 bullets as parents                        │
│       - Generate 4 candidates via crossover                   │
│       - Test all 5 bullets on 4 transactions                 │
│       - Keep only best bullet                                │
│     • Curator checks duplicates                               │
│     • Add with source='offline'                               │
│                                                               │
│  3. Bullet Storage                                            │
│     • Save to bullets table                                   │
│     • Generate embedding (cached)                            │
│     • Associate with evaluator                               │
│                                                               │
│  Output: Playbook populated with offline bullets              │
└─────────────────────────────────────────────────────────────┘
```

### Key Characteristics
- **Source**: All bullets marked `source='offline'`
- **Timing**: One-time pre-training phase
- **Data**: Historical/existing dataset
- **Evolution**: Darwin-Gödel enabled by default
- **Output**: Static bullet set ready for production

---

## Online Mode Infrastructure

### Purpose
Real-time learning from production transactions.

### Endpoint
`POST /api/v1/trace`

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│               Online Learning Infrastructure                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  For each production transaction:                             │
│                                                               │
│  1. Input Processing                                          │
│     • Receive transaction data                               │
│     • Extract input_text, output, ground_truth               │
│                                                               │
│  2. Pattern Classification                                    │
│     • Classify to existing pattern OR create new             │
│     • Pattern embedding for similarity                        │
│                                                               │
│  3. Transaction Storage                                       │
│     • Save transaction with pattern_id                        │
│     • Mode: offline_online or online_only                    │
│                                                               │
│  4. Multi-Evaluator Assessment                               │
│     • For each evaluator:                                     │
│       - Load LLM judge configuration                          │
│       - Evaluate output against criteria                      │
│       - Get judge reasoning                                  │
│                                                               │
│  5. Bullet Generation                                         │
│     • For each evaluator with feedback:                      │
│       - Reflector generates bullet                            │
│       - Input: query, predicted, correct, judge_reasoning    │
│       - Darwin evolution (if enabled):                       │
│         * Generate 4 candidates                              │
│         * Test on 4 transactions                             │
│         * Keep best bullet                                   │
│       - Curator deduplicates                                  │
│       - Save with source='online'                             │
│                                                               │
│  6. Effectiveness Tracking                                    │
│     • Update helpful_count/harmful_count                      │
│     • Track pattern-bullet associations                      │
│                                                               │
│  7. Metrics Update                                            │
│     • Update SessionRunMetrics                               │
│     • Track per-evaluator accuracy                           │
│                                                               │
│  Output: New bullets added to playbook                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Characteristics
- **Source**: Bullets marked `source='online'`
- **Timing**: Continuous real-time learning
- **Data**: Live production transactions
- **Evolution**: Darwin-Gödel enabled by configuration
- **Output**: Continuously growing bullet set

---

## Hybrid Mode Infrastructure (Offline + Online)

### Purpose
Combine pre-trained offline bullets with online learned bullets.

### Context Retrieval
`POST /api/v1/context`

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│           Hybrid Mode Context Retrieval                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: input_text, node, max_bullets_per_evaluator          │
│                                                               │
│  1. Pattern Classification                                    │
│     • Classify input to pattern                              │
│                                                               │
│  2. Evaluator Retrieval                                       │
│     • Get all evaluators for this node                       │
│                                                               │
│  3. Bullet Selection (Per Evaluator)                         │
│     • Full Context (offline + online):                       │
│       - Get ALL bullets for evaluator                        │
│       - HybridSelector intelligently selects                 │
│       - 5-stage selection process                             │
│     • Online Context (online only):                          │
│       - Filter to source='online'                            │
│       - Same selection process                               │
│                                                               │
│  4. Context Formatting                                         │
│     • Organize by evaluator                                   │
│     • Format: "EVALUATOR_NAME Rules:\n-bullet1\n-bullet2"    │
│                                                               │
│  Output:                                                      │
│  • full: All bullets (offline + online)                     │
│  • online: Only online bullets                              │
│  • bullet_ids: Track which bullets were used                │
└─────────────────────────────────────────────────────────────┘
```

### Usage in Agent

```
Agent Prompt Structure:
┌─────────────────────────────────────────────────────────────┐
│ You are a fraud detection expert.                            │
│                                                               │
│ [FORMAT_COMPLIANCE Rules:                                    │
│ - bullet 1                                                    │
│ - bullet 2]                                                   │
│                                                               │
│ [LOGIC_CONSISTENCY Rules:                                     │
│ - bullet 1                                                    │
│ - bullet 2]                                                   │
│                                                               │
│ Transaction: {input_text}                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Context Retrieval Flow (`POST /api/v1/context`)

```
┌─────────────────────────────────────────────────────────────┐
│            Context Retrieval Infrastructure                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Step 1: Initialize Components                               │
│  • PatternManager for classification                        │
│  • HybridSelector for intelligent selection                  │
│  • BulletPlaybook for retrieval                              │
│                                                               │
│  Step 2: Pattern Classification                              │
│  • Classify input_text to pattern                           │
│  • Get pattern_id and confidence                             │
│                                                               │
│  Step 3: Get Evaluators                                      │
│  • Query LLM judges table                                    │
│  • Get all active evaluators for node                        │
│                                                               │
│  Step 4: Select Bullets (Per Evaluator)                     │
│  For each evaluator:                                         │
│  • Get all bullets for this evaluator                        │
│  • Apply HybridSelector:                                     │
│    - Stage 1: Contextual filtering (node, evaluator)        │
│    - Stage 2: Quality filtering (success rate)              │
│    - Stage 3: Semantic filtering (similarity)                │
│    - Stage 4: Hybrid scoring (quality + semantic +         │
│                Thompson sampling)                            │
│    - Stage 5: Diversity promotion                            │
│  • Limit to max_bullets_per_evaluator                       │
│                                                               │
│  Step 5: Build Context Strings                               │
│  • Full context: All bullets                                 │
│  • Online context: Filter to source='online'                 │
│  • Format: "EVALUATOR Rules:\n-bullet1\n-bullet2"            │
│                                                               │
│  Step 6: Return Results                                      │
│  • context.full: Complete context with all bullets           │
│  • context.online: Only online bullets                      │
│  • bullet_ids: Track which bullets were selected             │
└─────────────────────────────────────────────────────────────┘
```

---

## Darwin-Gödel Evolution Infrastructure

### Purpose
Optimize bullets through genetic programming.

### Trigger
**EVERY** time a new bullet is generated (offline or online).

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│           Darwin-Gödel Evolution Process                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Prerequisites:                                              │
│  • At least 2 existing bullets in playbook                  │
│  • Darwin-Gödel evolution enabled                            │
│                                                               │
│  Phase 1: Bullet Generation                                  │
│  • Reflector generates new bullet from failure               │
│                                                               │
│  Phase 2: Crossover (Genetic Generation)                    │
│  • Get last 6 bullets from playbook (used as parents ONLY)   │
│  • Generate 4 candidate bullets via LLM crossover:           │
│    - Randomly select 2 different parents                     │
│    - LLM intelligently merges features                      │
│    - Create novel child bullet                               │
│  • Parents are NOT tested - only used for breeding            │
│                                                               │
│  Phase 3: Testing (Fitness Evaluation)                       │
│  • Test Set: 5 NEW bullets (1 new + 4 candidates)           │
│  • For each bullet:                                          │
│    - Test on 4 random transactions from database              │
│    - LLM Judge evaluates: "Would this bullet help?"          │
│    - Count helpful transactions                              │
│    - Calculate fitness = helpful_count / 4                   │
│                                                               │
│  Phase 4: Selection (Survival of the Fittest)               │
│  • Sort all 5 bullets by fitness score                       │
│  • Keep ONLY the best bullet                                 │
│  • Discard the other 4                                       │
│                                                               │
│  Phase 5: Deduplication                                       │
│  • Curator checks for duplicates (similarity > 0.85)        │
│  • If not duplicate, add to playbook                         │
│  • Mark with source='evolution'                              │
│                                                               │
│  Result: Best-performing bullet added to playbook            │
└─────────────────────────────────────────────────────────────┘
```

### Key Points
- **Old bullets are NOT re-tested** - only used as parents
- **Only newly generated bullets are tested** - reduces API calls
- **Top 1 bullet kept** - strict survival criteria
- **Per-evaluator evolution** - each evaluator evolves independently

---

## Bullet Selection Infrastructure (HybridSelector)

### Purpose
Intelligently select the best bullets for each transaction.

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│          HybridSelector 5-Stage Selection Process            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Stage 1: Contextual Filtering                              │
│  • Filter by node (agent identifier)                         │
│  • Filter by evaluator (perspective)                        │
│  • Optional: Filter by source (offline/online)               │
│                                                               │
│  Stage 2: Quality Filtering                                  │
│  • Calculate success_rate = helpful_count /                  │
│                              (helpful_count + harmful_count) │
│  • Filter: success_rate >= 0.3 (minimum threshold)          │
│  • Relax threshold if not enough bullets                     │
│                                                               │
│  Stage 3: Semantic Filtering                                 │
│  • Generate embedding for query                              │
│  • Calculate cosine similarity with bullet embeddings        │
│  • Filter: similarity >= 0.5 (threshold)                    │
│  • Uses pre-calculated embeddings (cached)                    │
│                                                               │
│  Stage 4: Hybrid Scoring                                     │
│  • Quality Score (30%): success_rate                          │
│  • Semantic Score (40%): cosine similarity                    │
│  • Thompson Sampling (30%): Beta distribution                │
│  • Pattern Boost: Extra boost if bullet effective           │
│                     for this pattern                         │
│                                                               │
│  Stage 5: Diversity Promotion                                │
│  • Calculate average similarity to selected bullets           │
│  • Diversity bonus = (1 - avg_similarity) * 0.15             │
│  • Promote diverse bullets                                  │
│                                                               │
│  Result: Top N bullets ranked by final score                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Pattern-Based Effectiveness Tracking

### Purpose
Track which bullets work best for which input patterns.

### Flow

```
┌─────────────────────────────────────────────────────────────┐
│         Pattern-Bullet Effectiveness Tracking                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  When bullet is used:                                         │
│                                                               │
│  1. Pattern Classification                                    │
│     • Classify input to pattern                              │
│     • Get pattern_id                                         │
│                                                               │
│  2. Record Effectiveness                                     │
│     • Query bullet_input_effectiveness table                 │
│     • If pattern-bullet combination exists:                   │
│       - Update helpful_count or harmful_count                │
│     • If new:                                                  │
│       - Create new record                                    │
│       - Set counts based on is_helpful                        │
│                                                               │
│  3. Use for Future Selection                                  │
│     • Pattern boost in HybridSelector                        │
│     • Prioritize bullets that work for this pattern         │
│                                                               │
│  Database Table: bullet_input_effectiveness                  │
│  • pattern_id: Input pattern                                  │
│  • bullet_id: Bullet                                          │
│  • helpful_count: Times helpful for this pattern             │
│  • harmful_count: Times harmful for this pattern              │
│  • success_rate: helpful_count / (helpful_count +            │
│                 harmful_count)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Data Lifecycle

### Transaction Lifecycle

```
1. Input Arrives
   ↓
2. Pattern Classification
   ↓
3. Bullet Selection (if mode != vanilla)
   ↓
4. Agent Execution (with or without bullets)
   ↓
5. Output Generated
   ↓
6. Multi-Evaluator Assessment
   ↓
7. Bullet Generation (if mode != vanilla)
   ↓
8. Effectiveness Tracking
   ↓
9. Metrics Update
   ↓
10. Transaction Saved
```

### Bullet Lifecycle

```
1. Generation (Offline or Online)
   ↓
2. Darwin Evolution (if enabled)
   ↓
3. Deduplication Check
   ↓
4. Storage in Playbook
   ↓
5. Selection for Transactions
   ↓
6. Effectiveness Tracking
   ↓
7. Quality Metrics Update
   ↓
8. Re-selection Based on Quality
```

---

## Summary

### Key Infrastructure Elements

1. **Multi-Mode Operation**: Vanilla, Offline+Online, Online Only
2. **Pattern-Based Learning**: Classify inputs and track bullet effectiveness per pattern
3. **Multi-Evaluator System**: Specialized LLM judges for different perspectives
4. **Darwin-Gödel Evolution**: Genetic optimization of bullets
5. **Hybrid Selection**: 5-stage intelligent bullet selection
6. **Continuous Learning**: Real-time adaptation from production data

### Data Flow Summary

- **Offline**: Historical data → Pattern classification → Bullet generation → Playbook population
- **Online**: Live transaction → Pattern classification → Bullet generation → Playbook update
- **Hybrid**: Input → Pattern classification → Intelligent selection → Agent execution → Learning

### Key Tables

- `bullets`: Bullet storage with embeddings
- `transactions`: Transaction history with pattern associations
- `input_patterns`: Pattern classifications
- `bullet_input_effectiveness`: Pattern-bullet effectiveness tracking
- `llm_judges`: Evaluator configurations
- `judge_evaluations`: Evaluation results
- `session_run_metrics`: Performance metrics per evaluator per mode

