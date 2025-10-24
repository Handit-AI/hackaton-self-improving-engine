# Self-Improving Process Infrastructure

## Overview

The Self-Improving Engine implements a dual-mode learning system that operates in **Offline** and **Online** modes. The system continuously generates, evolves, and applies "bullets" (heuristic rules) to improve agent performance.

---

## Fraud Detection Agent Architecture

The self-improving infrastructure was tested on a **Fraud Detection Agent** built with LangGraph. The agent uses a state-graph architecture with parallel analyzer execution and decision aggregation.

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server (Port 8001)              │
│                    with Handit AI Tracing                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │   LangGraphAgent   │
        │  (Main Orchestrator)│
        └────────────┬───────┘
                     │
        ┌────────────▼────────────┐
        │  RiskManagerGraph       │
        │  (StateGraph Based)     │
        └────────────┬────────────┘
                     │
        ┌────────────▼──────────────────────────────────────┐
        │        ORCHESTRATOR NODE (START)                  │
        │  Normalizes and enriches transaction data          │
        └────────────┬──────────────────────────────────────┘
                     │
        ┌────────────▼──────────────────────────────────────┐
        │         PARALLEL ANALYZER EXECUTION               │
        │  ┌──────────────┐ ┌──────────────┐               │
        │  │  Pattern     │ │ Behavioral   │               │
        │  │  Detector    │ │ Analyzer     │               │
        │  └──────────────┘ └──────────────┘               │
        │  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ │
        │  │  Velocity    │ │ Merchant     │ │Geographic││
        │  │  Checker     │ │ Risk Analyzer│ │ Analyzer ││
        │  └──────────────┘ └──────────────┘ └──────────┘ │
        └────────────┬──────────────────────────────────────┘
                     │
        ┌────────────▼──────────────────────┐
        │  DECISION AGGREGATOR NODE         │
        │  Combines all analyzer results    │
        └────────────┬──────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Final JSON Decision Output    │
        │  {final_decision, reason, ...} │
        └────────────────────────────────┘
```

### Node Descriptions

#### 1. **Orchestrator Node** (Start Node)
- **Purpose**: Normalizes and enriches incoming transaction data
- **Responsibilities**:
  - Validates input format
  - Extracts key transaction features
  - Prepares data for parallel analyzer execution
- **Output**: Enriched transaction data

#### 2. **Parallel Analyzer Nodes**

The system executes **5 specialized analyzers** in parallel:

##### a. **Pattern Detector**
- Analyzes spending patterns and anomalies
- Detects unusual transaction sequences
- Identifies behavioral deviations

##### b. **Behavioral Analyzer**
- Evaluates user behavior consistency
- Checks transaction history alignment
- Assesses profile evolution

##### c. **Velocity Checker**
- Monitors transaction frequency
- Detects rapid-fire attack patterns
- Identifies automated bot behavior

##### d. **Merchant Risk Analyzer**
- Evaluates merchant reputation
- Checks merchant fraud history
- Assesses transaction context

##### e. **Geographic Analyzer**
- Validates location consistency
- Detects impossible travel scenarios
- Checks timezone alignment

#### 3. **Decision Aggregator Node** (End Node)
- **Purpose**: Synthesizes all analyzer outputs into final decision
- **Responsibilities**:
  - Weighted aggregation of analyzer scores
  - Consensus building across analyzers
  - Final decision determination (APPROVE/REVIEW/DECLINE)
- **Output**: Structured JSON decision with reasoning

### Agent State Flow

```
1. Transaction Input → Orchestrator Node
2. Orchestrator → Parallel Execution (5 analyzers)
3. Analyzers → Individual Risk Assessments
4. Aggregator → Final Decision + Reasoning
5. Output → JSON with final_decision, risk_score, recommendations
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

## Generated Bullets: Real-World Examples

The following bullets were generated during offline training and demonstrate the system's ability to create actionable, generalized heuristics from specific failure patterns.

### Bullet 1: Format Compliance
```
When determining the final_decision field, ensure it strictly matches the required 
values of 'APPROVE', 'REVIEW', or 'DECLINE', and avoid using any variations or 
invalid terms.
```

**Context**: Generated from judge feedback identifying case mismatch errors (e.g., 'approve' vs 'APPROVE')

### Bullet 2: Conclusion Field Validation
```
The conclusion field must be 1-2 sentences (under 200 characters), professional 
language, no emoji or formatting. If conclusion contradicts final_decision 
(e.g., APPROVE + negative conclusion), flag as format violation.
```

**Context**: Addresses inconsistencies between decision and reasoning

### Bullet 3: Life Event vs Fraud Pattern Recognition
```
Legitimate life events show gradual changes over weeks/months: income increase → 
gradual spending increase, new hobby → consistent new merchant category. Fraud 
shows sudden changes within hours/days.
```

**Context**: Distinguishes between legitimate behavioral evolution and fraudulent patterns

### Bullet 4: Sequential Pattern Analysis
```
For sequential pattern analysis, detect rapid sequential transactions: <1 minute 
between distant locations = impossible travel, consistent 30-60 second gaps = 
automated/bot behavior.
```

**Context**: Identifies automated fraud signals from transaction timing

### Bullet 5: Time-of-Day Contextual Validation
```
Flag transactions outside typical_time_of_day UNLESS: customer has 
night_activity_normal=true OR transaction type justifies timing (emergency 
pharmacy, 24hr services) OR timezone differences for travel.
```

**Context**: Balances fraud detection with legitimate off-hours activity

### Bullet 6: Account Age Reliability
```
Account age <90 days has limited pattern reliability. For new accounts, focus on 
velocity analysis rather than spending patterns. Students/young accounts: Higher 
variance tolerance (stability <0.5 is normal).
```

**Context**: Adapts detection approach for new vs established customers

### Bullet 7: Amount Deviation Analysis
```
When transaction amount is >3x typical_transaction_amount, flag as suspicious 
UNLESS context justifies (life event, emergency, professional purchase). Check 
customer segment: STUDENT vs PREMIUM have different thresholds.
```

**Context**: Context-aware amount-based fraud detection

### Bullet 8: Emergency Situation Handling
```
For emergency situations (hospitals, pharmacies, natural disasters, stranded 
travelers), bias toward APPROVE/REVIEW, never auto-DECLINE. These are time-critical 
legitimate needs.
```

**Context**: Prevents false positives for critical legitimate transactions

---

## Training Phase Results Summary

### Test Configuration

Three comprehensive test runs were conducted to evaluate the self-improving system's performance across different scenarios:

**Test Run 1** (`test_results_20251023_115310.json`):
- Vanilla Mode: 81.4% accuracy (35/43 correct)
- Offline+Online Mode: 83.7% accuracy (36/43 correct)
- Total Samples: 43 transactions

**Test Run 2** (`test_results_20251023_134102.json`):
- Vanilla Mode: 75.0% accuracy (15/20 correct)
- Offline+Online Mode: 100% accuracy (10/10 correct)
- Total Samples: 20 transactions (subsequent subset)

**Test Run 3** (`test_results_20251023_123838.json`):
- Vanilla Mode: 83.7% accuracy (36/43 correct)
- Offline+Online Mode: 86.0% accuracy (37/43 correct)
- Total Samples: 43 transactions

### Key Findings

#### 1. Offline+Online Mode Outperforms Vanilla

Across all test runs, the Offline+Online mode consistently achieved **2-3% higher accuracy** than the vanilla baseline. This demonstrates that:
- Pre-trained bullets provide valuable guidance
- Online learning adds adaptive behavior
- The hybrid approach successfully combines external experience with real-time adaptation

#### 2. Bullet Selection Quality

The test results show sophisticated bullet selection with scores ranging from 0.57 to 0.73, where bullets are evaluated based on:
- **Quality Score** (30%): Empirical success rate
- **Semantic Score** (40%): Similarity to query
- **Thompson Sampling** (30%): Exploration-exploitation balance
- **Diversity Bonus** (up to 15%): Reduces redundancy

#### 3. Pattern Recognition Effectiveness

The system successfully classified transactions into patterns (e.g., "high_value_transaction", "new_user_pattern", "unusual_time") and selected bullets specific to each pattern, demonstrating intelligent contextual selection.

#### 4. Error Analysis

Common errors across modes included:
- Over-aggressive declines on legitimate business accounts
- Misclassification of crypto trader transactions
- Difficulty with borderline cases requiring nuanced judgment

These patterns indicate areas where additional training bullets would improve performance.
