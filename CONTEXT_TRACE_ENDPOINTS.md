# Context & Trace Endpoints Documentation

## Overview

These two endpoints work together to provide context to your agent and track bullet effectiveness for self-improvement.

---

## 1. Get Context

**POST** `/api/v1/context`

### Purpose

Returns full text context with bullets organized by evaluator. The agent uses this context in its prompt to make decisions.

### Request

```json
{
  "input_text": "New user from Nigeria using VPN to purchase $500 crypto",
  "node": "fraud_detection",
  "max_bullets_per_evaluator": 10
}
```

**Fields:**
- `input_text` (required): The transaction/input to analyze
- `node` (required): Agent node name (e.g., "fraud_detection")
- `max_bullets_per_evaluator` (optional): Max bullets per evaluator, default: 10

### Response

```json
{
  "status": "success",
  "node": "fraud_detection",
  "pattern_id": 5,
  "bullet_ids": {
    "full": ["fraud_detection_b99ac880", "fraud_detection_a1b2c3d4", "fraud_detection_e5f6g7h8"],
    "online": ["fraud_detection_a1b2c3d4", "fraud_detection_e5f6g7h8"]
  },
  "context": {
    "full": "FRAUD_DETECTION Rules:\n- New user (< 90 days) + VPN + crypto merchant + amount > $1000 = 97% fraud\n- VPN usage + suspicious location = 95% fraud\n- New user + large amount (> $1000) = 80% fraud\n\nRISK_ASSESSMENT Rules:\n- Multiple failed attempts = 90% fraud\n- High velocity transactions = 85% fraud",
    "online": "FRAUD_DETECTION Rules:\n- New user (< 90 days) + VPN + crypto merchant + amount > $1000 = 97% fraud\n- VPN usage + suspicious location = 95% fraud"
  }
}
```

**Fields:**
- `status`: "success" or "error"
- `node`: Agent node name
- `pattern_id`: Input pattern classification ID
- `bullet_ids.full`: Array of bullet IDs used in offline + online context
- `bullet_ids.online`: Array of bullet IDs used in online-only context
- `context.full`: Full text context with offline + online bullets
- `context.online`: Full text context with only online bullets

### How It Works

1. **Pattern Classification**: Input is classified into a pattern category
2. **Evaluator Detection**: Gets all evaluators for this node
3. **Bullet Selection**: For each evaluator, intelligently selects top bullets based on:
   - Semantic similarity to input
   - Quality score (helpful_count / harmful_count)
   - Thompson sampling
   - Pattern-specific effectiveness
4. **Context Building**: Organizes bullets by evaluator into ready-to-use text

### Context Format

```
EVALUATOR_NAME Rules:
- bullet 1
- bullet 2
- ...

EVALUATOR_NAME Rules:
- bullet 1
- bullet 2
- ...
```

### Usage in Agent Prompt

```python
# Get context
response = requests.post(f"{BASE_URL}/api/v1/context", json={
    "input_text": "New user from Nigeria using VPN",
    "node": "fraud_detection"
})
data = response.json()

# Build agent prompt
system_prompt = f"You are a fraud detection expert.{data['context']['full']}"
user_prompt = "New user from Nigeria using VPN"

# Save bullet IDs for later tracking
bullet_ids = data['bullet_ids']
```

---

## 2. Trace Transaction

**POST** `/api/v1/trace`

### Purpose

Saves transaction to database and executes online learning (bullet generation and effectiveness tracking).

### Request

```json
{
  "input_text": "New user from Nigeria using VPN to purchase $500 crypto",
  "node": "fraud_detection",
  "output": "DECLINE",
  "ground_truth": "DECLINE",
  "agent_reasoning": "High risk factors: new user, VPN usage, crypto purchase",
  "bullet_ids": {
    "full": ["fraud_detection_b99ac880", "fraud_detection_a1b2c3d4", "fraud_detection_e5f6g7h8"],
    "online": ["fraud_detection_a1b2c3d4", "fraud_detection_e5f6g7h8"]
  }
}
```

**Fields:**
- `input_text` (required): The transaction/input that was analyzed
- `node` (required): Agent node name
- `output` (required): Agent's decision/output (unlimited length)
- `ground_truth` (optional): Correct answer (unlimited length, defaults to output if not provided)
- `agent_reasoning` (optional): Agent's reasoning for the decision
- `bullet_ids` (optional): Object with bullet IDs that were used
  - `full`: Bullet IDs used in offline + online context
  - `online`: Bullet IDs used in online-only context

### Response

```json
{
  "status": "success",
  "node": "fraud_detection",
  "transaction_id": 123,
  "pattern_id": 5,
  "is_correct": true,
  "message": "Processing in background"
}
```

**Fields:**
- `status`: "success" or "error"
- `node`: Agent node name
- `transaction_id`: Database ID of saved transaction
- `pattern_id`: Input pattern classification ID
- `is_correct`: Whether output matches ground truth
- `message`: "Processing in background" (indicates async processing)

**Note:** This endpoint returns immediately. Bullet generation and effectiveness tracking happen in the background.

### How It Works

1. **Pattern Classification**: Input is classified into a pattern category
2. **Correctness Check**: 
   - If `ground_truth` provided: Direct comparison
   - If `ground_truth` NOT provided: Uses LLM judge from database (per node)
3. **Transaction Saving**: Saves transaction to database with pattern association
4. **Return Immediately**: Returns transaction ID without waiting
5. **Background Processing** (async):
   - Evaluates with LLM judge if no ground truth (queries LLM judge by node from database)
   - Records effectiveness for each bullet ID provided
   - If correct: `helpful_count += 1`
   - If incorrect: `harmful_count += 1`
   - Always: `times_selected += 1`
   - Generates new bullet if needed (with Darwin-Gödel evolution)

### Effectiveness Tracking

When `bullet_ids` are provided:

```python
# For each bullet in bullet_ids["full"]:
# Record effectiveness in bullet_input_effectiveness table
if is_correct:
    helpful_count += 1
else:
    harmful_count += 1
times_selected += 1
```

This tracking is done **per pattern**, so the same bullet can have different effectiveness scores for different patterns.

---

## Complete Flow

```python
# 1. Get context for agent
context_response = requests.post(f"{BASE_URL}/api/v1/context", json={
    "input_text": "New user from Nigeria using VPN",
    "node": "fraud_detection",
    "max_bullets_per_evaluator": 10
})
context_data = context_response.json()

# Extract context and bullet IDs
system_prompt = f"You are a fraud detection expert.{context_data['context']['full']}"
bullet_ids = context_data['bullet_ids']

# 2. Agent executes with context
user_prompt = "New user from Nigeria using VPN"
agent_output = "DECLINE"  # Agent's decision
agent_reasoning = "High risk factors detected"

# 3. Trace transaction (returns immediately, processes in background)
trace_response = requests.post(f"{BASE_URL}/api/v1/trace", json={
    "input_text": "New user from Nigeria using VPN",
    "node": "fraud_detection",
    "output": agent_output,
    "ground_truth": "DECLINE",
    "agent_reasoning": agent_reasoning,
    "bullet_ids": bullet_ids  # Pass bullet IDs from context
})
trace_data = trace_response.json()

# Response received immediately:
# {
#   "status": "success",
#   "transaction_id": 123,
#   "message": "Processing in background"
# }

# Bullet effectiveness tracking and generation happen in background
```

---

## Key Points

### Context Endpoint
- **Returns**: Ready-to-use text context + bullet IDs
- **Bullet IDs**: Track which bullets were used for each context type
- **Pattern ID**: Used for pattern-specific bullet selection

### Trace Endpoint
- **Records**: Transaction and bullet effectiveness
- **Tracks**: Which bullets were helpful/harmful per pattern
- **Learns**: Generates new bullets with Darwin-Gödel evolution
- **Evolution**: Tests 5 bullets, keeps top 2
- **Async**: Returns immediately, processing happens in background

### Bullet IDs Structure
```json
{
  "full": ["bullet1", "bullet2", "bullet3"],    // All bullets (offline + online)
  "online": ["bullet2", "bullet3"]              // Only online bullets
}
```

### Effectiveness Tracking
- Tracked per pattern (not globally)
- Updated in `bullet_input_effectiveness` table
- Used for selection scoring in future queries
- Better bullets get selected more often

---

## Notes

1. **Always pass bullet_ids from context to trace**: This enables effectiveness tracking
2. **Bullet IDs are optional**: Endpoint works without them, just won't track effectiveness
3. **Pattern-specific**: Same bullet can be good for one pattern, bad for another
4. **Automatic evolution**: Darwin-Gödel evolution happens automatically when generating bullets
5. **Database persistence**: All transactions and effectiveness records are saved
6. **Async processing**: Trace endpoint returns immediately, heavy processing happens in background
7. **Background tasks**: FastAPI handles background task execution automatically
8. **LLM Judge per node**: If no ground_truth provided, system queries LLM judge from database for that node
9. **Judge configuration**: LLM judges are configured per node with custom system prompts and evaluation criteria
10. **No judge available**: If no LLM judge exists for a node, defaults to False (marked as incorrect)

