# Self-Improving Engine API Documentation

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication is required.

---

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the service is healthy and database is connected.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected"
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service is unhealthy (database disconnected)

---

### 2. Train Offline

**POST** `/api/v1/train`

Train bullets on a provided dataset with Darwin-Gödel evolution.

**Request Body:**
```json
{
  "dataset": [
    {
      "query": "New user from Nigeria using VPN to purchase $500 crypto",
      "predicted": "DECLINE",  // Optional: predicted output
      "answer": "DECLINE"      // Required: ground truth
    },
    {
      "query": "Long-time customer buying groceries for $50",
      "predicted": "APPROVE",  // Optional
      "answer": "APPROVE"      // Required
    }
  ],
  "node": "fraud_detection",    // Required: agent node name
  "max_samples": 10              // Optional: max samples to process (default: 10)
}
```

**Response:**
```json
{
  "status": "success",
  "node": "fraud_detection",
  "samples_processed": 10,
  "bullets_generated": 8,
  "total_bullets": 12,
  "unique_bullets": 10
}
```

**Fields:**
- `status`: "success" or "error"
- `node`: Agent node name
- `samples_processed`: Number of samples processed
- `bullets_generated`: Number of bullets generated
- `total_bullets`: Total bullets in playbook for this node
- `unique_bullets`: Number of unique bullets (after deduplication)

**Notes:**
- Each item generates bullets with Darwin-Gödel evolution
- Evolution tests 5 bullets (1 new + 4 candidates) and keeps top 2
- Bullets are deduplicated using semantic similarity (threshold: 0.85)
- Evolution happens ONLY if there are 2+ existing bullets

**Status Codes:**
- `200`: Training completed successfully
- `500`: Internal error

---

### 3. Trace Transaction

**POST** `/api/v1/trace`

Save transaction and execute online learning (synchronous).

**Request Body:**
```json
{
  "input_text": "New user from Nigeria using VPN",
  "node": "fraud_detection",
  "output": "DECLINE",                    // Required: agent's output (unlimited length)
  "model_type": "online",                 // Optional: mode type (vanilla, offline_online, online, or full) - defaults to "online"
  "session_id": "session-123",            // Optional: session identifier for metrics tracking
  "run_id": "run-1",                      // Optional: run identifier within session
  "ground_truth": "DECLINE",              // Optional: ground truth (unlimited length, defaults to output)
  "agent_reasoning": "High risk factors", // Optional: agent's reasoning
  "bullet_ids": {                         // Optional: bullet IDs used
    "full": ["bullet1", "bullet2"],
    "online": ["bullet2"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "node": "fraud_detection",
  "transaction_id": 123,
  "pattern_id": 5,
  "is_correct": true,
  "message": "Processing completed"
}
```

**Response Fields:**
- `status`: "success" or "error"
- `node`: Agent node name
- `transaction_id`: Database ID of saved transaction
- `pattern_id`: Input pattern classification ID (set after processing)
- `is_correct`: Whether output matches ground truth
- `message`: "Processing completed" (indicates all processing is done)

**Fields:**
- `input_text` (required): The transaction/input that was analyzed
- `node` (required): Agent node name
- `output` (required): Agent's decision/output (unlimited length)
- `model_type` (optional): Mode type - "vanilla", "offline_online", "online", or "full" (maps "full" to "offline_online") - defaults to "online"
- `session_id` (optional): Session identifier for tracking metrics across multiple runs
- `run_id` (optional): Run identifier within a session for tracking different runs
- `ground_truth` (optional): Correct answer (unlimited length, defaults to output if not provided)
- `agent_reasoning` (optional): Agent's reasoning for the decision
- `bullet_ids` (optional): Object with bullet IDs that were used
  - `full`: Bullet IDs used in offline + online context
  - `online`: Bullet IDs used in online-only context

**Notes:**
- **Returns only after all processing is complete** - waits for bullet generation, evolution, and effectiveness tracking
- Transaction is **always saved** to database regardless of evaluators
- Pattern classification happens before saving transaction
- **Only runs improvement and metrics if evaluators exist** for the node
- Evaluators are **retrieved from `llm_judges` table** - no evaluators are created
- If no evaluators exist, transaction is saved but no bullet generation or metrics tracking occurs
- Online learning generates bullet with Darwin-Gödel evolution in background
- Bullet is evaluated against previous transactions from the same node
- Evolution tests 5 bullets and keeps top 2
- If `session_id` and `run_id` are provided, metrics are automatically tracked
- Response includes `pattern_id` and `is_correct` after processing
- **Metrics are tracked per evaluator** - each evaluator has separate metrics

**Status Codes:**
- `200`: Trace completed successfully
- `500`: Internal error

---

### 4. Get Context

**POST** `/api/v1/context`

Get full text context for agent prompt with bullets organized by evaluator.

**Request Body:**
```json
{
  "input_text": "New user from Nigeria using VPN",
  "node": "fraud_detection",
  "max_bullets_per_evaluator": 10  // Optional: max bullets per evaluator (default: 10)
}
```

**Response:**
```json
{
  "status": "success",
  "node": "fraud_detection",
  "pattern_id": 5,
  "bullet_ids": {
    "full": ["bullet1", "bullet2", "bullet3"],
    "online": ["bullet2", "bullet3"]
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
- `bullet_ids.full`: Bullet IDs for offline + online context
- `bullet_ids.online`: Bullet IDs for online-only context
- `context.full`: Full context with offline + online bullets
- `context.online`: Only online bullets (real-time learning)

**Context Format:**
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

**Notes:**
- Uses intelligent bullet selection based on semantic similarity
- Organizes bullets by evaluator (perspective)
- Returns ready-to-use text for agent prompts
- Can replace placeholder in system prompt: `"You are a fraud detection expert.{context}"`
- **Evaluators are retrieved from `llm_judges` table** - no evaluators are created
- **Returns empty context if no evaluators exist** for the node

**Status Codes:**
- `200`: Context retrieved successfully
- `500`: Internal error

---

### 5. Get Session Metrics

**GET** `/api/v1/metrics/{session_id}`

Get metrics for a specific session, grouped by run_id and evaluator.

**Path Parameters:**
- `session_id` (str): Session identifier

**Response:**
```json
{
  "status": "success",
  "session_id": "session-123",
  "metrics": {
    "run-1": {
      "fraud_detection": {
        "online": {
          "correct_count": 45,
          "total_count": 50,
          "accuracy": 0.9,
          "node": "fraud_detection"
        }
      }
    },
    "run-2": {
      "fraud_detection": {
        "online": {
          "correct_count": 48,
          "total_count": 50,
          "accuracy": 0.96,
          "node": "fraud_detection"
        }
      }
    }
  }
}
```

**Fields:**
- `status`: "success" or "error"
- `session_id`: Session identifier
- `metrics`: Object organized by run_id, then evaluator, then mode
  - `run_id`: Object containing evaluators
    - `evaluator`: Object containing modes
      - `mode`: Object with metrics (vanilla, offline_online, or online)
        - `correct_count`: Number of correct predictions
        - `total_count`: Total number of predictions
        - `accuracy`: Calculated accuracy (correct_count / total_count)
        - `node`: Agent node name

**Notes:**
- Returns metrics aggregated by run_id, evaluator, and mode
- Accuracy is automatically calculated
- Useful for tracking performance across different runs and modes in a session
- Each evaluator has separate metrics for each mode

**Status Codes:**
- `200`: Metrics retrieved successfully
- `500`: Internal error

---

### 7. Get Judge Evaluations

**GET** `/api/v1/judge-evaluations/{transaction_id}`

Get judge evaluations for a specific transaction.

**Path Parameters:**
- `transaction_id` (int): Transaction identifier

**Response:**
```json
{
  "status": "success",
  "transaction_id": 123,
  "evaluations": [
    {
      "judge_id": 1,
      "judge_node": "fraud_detection",
      "judge_evaluator": "pattern_consistency",
      "input_text": "New user from Nigeria using VPN",
      "output_text": "DECLINE",
      "ground_truth": null,
      "is_correct": true,
      "confidence": 0.95,
      "reasoning": "The output correctly identifies the high-risk pattern",
      "judge_was_correct": null,
      "evaluated_at": "2025-10-23T15:30:00"
    }
  ]
}
```

**Fields:**
- `status`: "success" or "error"
- `transaction_id`: Transaction identifier
- `evaluations`: Array of judge evaluations
  - `judge_id`: ID of the judge used
  - `judge_node`: Node name of the judge
  - `judge_evaluator`: Evaluator name of the judge
  - `input_text`: Input text that was evaluated
  - `output_text`: Output that was evaluated
  - `ground_truth`: Ground truth if available
  - `is_correct`: Judge's assessment (true/false)
  - `confidence`: Judge's confidence (0.0-1.0)
  - `reasoning`: Judge's reasoning
  - `judge_was_correct`: Whether judge was correct (if ground truth available)
  - `evaluated_at`: Timestamp of evaluation

**Notes:**
- Returns all judge evaluations for the transaction
- Useful for debugging and auditing judge performance
- `judge_was_correct` is only set when ground truth is available

**Status Codes:**
- `200`: Judge evaluations retrieved successfully
- `500`: Internal error

---

### 8. Analyze Transaction (Legacy)

**POST** `/api/v1/analyze`

Analyze transaction with specified mode (legacy endpoint).

**Request Body:**
```json
{
  "transaction": {
    "user": "John Doe",
    "amount": 500,
    "merchant": "Crypto Exchange"
  },
  "mode": "vanilla"  // "vanilla", "offline_online", or "online_only"
}
```

**Response:**
```json
{
  "mode": "vanilla",
  "decision": "DECLINE",
  "risk_score": 0.85
}
```

**Status Codes:**
- `200`: Analysis completed
- `400`: Invalid mode
- `500`: Internal error

---

### 9. Evaluate and Generate Bullets (Legacy)

**POST** `/api/v1/evaluate`

Evaluate agent output and generate bullets using LLM as judge (legacy endpoint).

**Request Body:**
```json
{
  "input_text": "Transaction details",
  "node": "fraud_detection",
  "output": "DECLINE",
  "ground_truth": "DECLINE"
}
```

**Response:**
```json
{
  "node": "fraud_detection",
  "is_correct": true,
  "bullet_id": "fraud_detection_b99ac880",
  "bullet_added": true
}
```

**Status Codes:**
- `200`: Evaluation completed
- `500`: Internal error

---

### 10. Get Bullets (Legacy)

**POST** `/api/v1/get-bullets`

Get bullets for a query and node based on mode (legacy endpoint).

**Request Body:**
```json
{
  "query": "Transaction details",
  "node": "fraud_detection",
  "mode": "hybrid"  // "vanilla", "offline_online", or "online_only"
}
```

**Response:**
```json
{
  "mode": "offline_online",
  "bullets": {
    "offline": [...],
    "online": [...]
  }
}
```

**Status Codes:**
- `200`: Bullets retrieved
- `400`: Invalid mode
- `500`: Internal error

---

### 11. Get Playbook Stats

**GET** `/api/v1/playbook/stats`

Get statistics for the playbook.

**Response:**
```json
{
  "stats": {
    "total_bullets": 100,
    "bullets_per_node": {
      "fraud_detection": 100
    }
  },
  "total_bullets": 100
}
```

**Status Codes:**
- `200`: Stats retrieved successfully

---

### 12. Get Node Playbook

**GET** `/api/v1/playbook/{node}`

Get bullets for a specific node.

**Query Parameters:**
- `limit`: Maximum number of bullets (default: 10)
- `query`: Optional query text for intelligent selection

**Examples:**
```
GET /api/v1/playbook/fraud_detection?limit=20
GET /api/v1/playbook/fraud_detection?query=VPN+usage&limit=5
```

**Response:**
```json
{
  "node": "fraud_detection",
  "bullets": [
    {
      "id": "fraud_detection_b99ac880",
      "content": "New user + VPN + crypto = 97% fraud",
      "node": "fraud_detection",
      "evaluator": "fraud_detection",
      "source": "offline",
      "helpful_count": 10,
      "harmful_count": 2,
      "times_selected": 15
    }
  ],
  "selection_method": "intelligent"  // or "all"
}
```

**Status Codes:**
- `200`: Playbook retrieved successfully

---

## Integration Example

### Complete Flow

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Train offline
train_data = {
    "dataset": [
        {"query": "New user from Nigeria using VPN", "answer": "DECLINE"},
        {"query": "Long-time customer buying groceries", "answer": "APPROVE"}
    ],
    "node": "fraud_detection",
    "max_samples": 10
}
response = requests.post(f"{BASE_URL}/api/v1/train", json=train_data)
print(response.json())

# 2. Get context for agent
context_data = {
    "input_text": "New user from Nigeria using VPN",
    "node": "fraud_detection",
    "max_bullets_per_evaluator": 10
}
response = requests.post(f"{BASE_URL}/api/v1/context", json=context_data)
context = response.json()["context"]

# Build agent prompt
system_prompt = f"You are a fraud detection expert.{context['full']}"
user_prompt = "New user from Nigeria using VPN"

# 3. Agent executes and returns output
agent_output = "DECLINE"

# 4. Trace transaction
trace_data = {
    "input_text": "New user from Nigeria using VPN",
    "node": "fraud_detection",
    "output": agent_output,
    "ground_truth": "DECLINE",
    "agent_reasoning": "High risk factors detected"
}
response = requests.post(f"{BASE_URL}/api/v1/trace", json=trace_data)
print(response.json())
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found
- `500`: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Notes

1. **Darwin-Gödel Evolution**: Automatically runs when 2+ bullets exist
2. **Bullet Deduplication**: Uses semantic similarity threshold of 0.85
3. **Evaluator Support**: Bullets are organized by evaluator (perspective)
4. **Pattern Classification**: Transactions are classified into patterns automatically
5. **Database Persistence**: All transactions and bullets are persisted to PostgreSQL
6. **Node-Specific**: Evolution tests bullets only on transactions from the same node

---

## Architecture

```
┌─────────────┐
│   Agent     │ ──► Execute transaction
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Context   │ ◄─── GET /api/v1/context
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Output    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Trace     │ ──► POST /api/v1/trace
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Learning  │ (Darwin-Gödel Evolution)
└─────────────┘
```

---

## Support

For issues or questions, check the logs or contact the development team.

