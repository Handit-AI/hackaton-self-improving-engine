# ACE System Flow Documentation

## Overview

The **Adaptive Critical Experience (ACE)** system is a self-improving fraud detection system that learns from both historical data (offline) and real-time transactions (online). It uses a playbook of "bullets" (heuristic rules) that are continuously refined through a 5-stage selection process.

---

## 🎯 Core Components

### 1. **BulletPlaybook** - Knowledge Storage
- Stores all bullets (heuristic rules) organized by agent node
- Tracks performance metrics: `helpful_count`, `harmful_count`, `times_selected`
- Supports filtering by `source` (offline vs online)
- Maintains embeddings for semantic search

### 2. **HybridSelector** - 5-Stage Selection Algorithm
- Selects the best bullets for each transaction
- Combines quality, semantic similarity, and exploration

### 3. **Reflector** - Bullet Generation
- Generates new bullets from successes/failures
- Uses GPT-4o-mini to extract actionable heuristics

### 4. **Curator** - Deduplication
- Prevents duplicate bullets
- Uses text similarity checking

### 5. **TrainingPipeline** - Orchestration
- Coordinates offline and online training
- Manages bullet lifecycle

---

## 📊 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Transaction Input                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   1. Vanilla Mode (Baseline)                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Agent (GPT-3.5-turbo) → Decision                          │ │
│  │  Judge (GPT-4o-mini) → Correctness                          │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              2. Offline + Online Mode (Pre-trained)              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Step 1: Offline Training                                   │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │ Train Set → Reflector → Bullets (source='offline')  │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  │                                                              │ │
│  │  Step 2: Test with Offline Bullets                          │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │ Test Set → HybridSelector → Select Bullets        │   │ │
│  │  │           → Agent (with bullets) → Decision        │   │ │
│  │  │           → Judge → Correctness                    │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              3. Online Only Mode (Real-time Learning)           │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  For each transaction:                                     │ │
│  │  1. Agent (GPT-3.5-turbo) → Decision                      │ │
│  │  2. Judge (GPT-4o-mini) → Correctness                      │ │
│  │  3. Reflector → Generate Bullet (source='online')         │ │
│  │  4. Curator → Check Duplicates → Add to Playbook          │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Component Flows

### **Flow 1: Bullet Generation (Reflector)**

**Purpose**: Extract actionable heuristics from transaction outcomes

**Inputs**:
- `query`: Transaction text
- `predicted`: Agent's decision
- `correct`: Ground truth
- `node`: Agent node name
- `agent_reasoning`: Agent's explanation

**Process**:
1. Determine if prediction was correct
2. Call GPT-4o-mini with structured prompt
3. Extract JSON with:
   - `new_bullet`: Specific heuristic
   - `problem_types`: Categories
   - `confidence`: Confidence score

**Example Output**:
```json
{
  "new_bullet": "VPN from high-risk countries (Nigeria, Romania) + crypto merchant = 95% fraud",
  "problem_types": ["location_fraud", "merchant_risk"],
  "confidence": 0.92
}
```

**Key Features**:
- 🎯 Actionable: Includes thresholds and specific conditions
- 📊 Measurable: Can be quantified
- ✅ Specific: Avoids vague advice

---

### **Flow 2: Bullet Selection (HybridSelector)**

**Purpose**: Select the best bullets for a given transaction

**5-Stage Process**:

#### **Stage 1: Contextual Filtering**
- Filter bullets by `node` (which agent uses them)
- Optional: Filter by `source` (offline vs online)
- **Output**: Node-specific bullets

#### **Stage 2: Quality Filtering**
```python
# Filter by success rate
quality_threshold = 0.3  # Minimum success rate
filtered = [b for b in bullets if b.get_success_rate() >= threshold]

# Relax threshold if not enough bullets
if len(filtered) < n_bullets:
    relaxed_threshold = threshold * 0.8
```

**Metrics**:
- `helpful_count`: Times bullet led to correct decision
- `harmful_count`: Times bullet led to wrong decision
- `get_success_rate()`: `helpful_count / (helpful_count + harmful_count)`

#### **Stage 3: Semantic Filtering**
```python
# Get query embedding
query_embedding = get_embedding(query)

# Calculate similarity for each bullet
for bullet in bullets:
    similarity = cosine_similarity(query_embedding, bullet.embedding)
    semantic_scores[bullet.id] = similarity

# Filter by threshold
semantic_threshold = 0.5
filtered = [b for b in bullets if semantic_scores[b.id] >= threshold]
```

**Features**:
- Uses OpenAI `text-embedding-3-small` model
- Caches embeddings for performance
- Cosine similarity for ranking

#### **Stage 4: Hybrid Scoring**
```python
for bullet in bullets:
    # Component scores
    quality_score = bullet.get_success_rate()  # 0-1
    semantic_score = semantic_scores[bullet.id]  # 0-1
    thompson_score = thompson_sample(bullet)  # Beta distribution
    
    # Combined score
    combined = (
        0.3 * quality_score +      # 30% weight
        0.4 * semantic_score +     # 40% weight
        0.3 * thompson_score       # 30% weight
    )
```

**Thompson Sampling**:
- Balances exploration vs exploitation
- Uses Beta distribution: `Beta(helpful_count + 1, harmful_count + 1)`
- Allows underperforming bullets to be tried occasionally

#### **Stage 5: Diversity Promotion**
```python
for bullet in scored_bullets:
    # Calculate diversity bonus
    diversity_bonus = 0.0
    if selected_bullets:
        avg_similarity = avg_similarity_to_selected(bullet)
        diversity_bonus = (1 - avg_similarity) * 0.15
    
    final_score = combined_score + diversity_bonus
```

**Purpose**: Avoid redundant bullets (promote variety)

---

### **Flow 3: Deduplication (Curator)**

**Purpose**: Prevent duplicate bullets

**Process**:
```python
def merge_bullet(content, node, playbook):
    # Check for duplicates
    for existing in playbook.get_bullets_for_node(node):
        similarity = text_similarity(content, existing.content)
        if similarity > 0.85:  # 85% threshold
            return None  # Duplicate!
    
    # Add new bullet
    return playbook.add_bullet(content, node)
```

**Similarity Check**:
- Uses Python's `SequenceMatcher`
- Case-insensitive comparison
- Threshold: 85% similarity = duplicate

---

### **Flow 4: Evaluation (Judge)**

**Purpose**: Determine if agent's decision was correct

**Inputs**:
- `input_text`: Transaction query
- `output`: Agent's decision
- `ground_truth`: Correct answer (optional)

**Process**:
```python
async def judge(input_text, output, ground_truth):
    # If ground truth provided, exact match
    if ground_truth:
        return output.strip().lower() == ground_truth.strip().lower()
    
    # Otherwise use LLM judge
    response = GPT4oMini.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Expert judge..."},
            {"role": "user", "content": f"Input: {input_text}\nOutput: {output}\nCorrect?"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    result = json.loads(response.choices[0].message.content)
    return result["is_correct"], result["confidence"]
```

**Output**:
```json
{
  "is_correct": true,
  "confidence": 0.92,
  "reasoning": "Output matches expected fraud detection pattern"
}
```

---

## 🎬 Complete Training Flow

### **Offline Training**

```
1. Load Training Dataset
   ↓
2. For each training example:
   a. Call Reflector to generate bullet
   b. Mark source='offline'
   c. Add to playbook via Curator
   ↓
3. Playbook now contains offline bullets
```

### **Online Training**

```
1. For each transaction:
   ↓
2. Agent analyzes transaction
   ↓
3. Judge evaluates correctness
   ↓
4. If wrong or periodically:
   a. Reflector generates new bullet
   b. Curator checks for duplicates
   c. If not duplicate, add to playbook (source='online')
   ↓
5. Update bullet stats (helpful/harmful)
```

---

## 🔍 Key Algorithms

### **Thompson Sampling**

```python
def thompson_sample(bullet):
    alpha = bullet.helpful_count + 1  # Successes + 1
    beta = bullet.harmful_count + 1   # Failures + 1
    return beta_distribution.rvs(alpha, beta)
```

**Purpose**: Balance exploration vs exploitation
- High-performing bullets get selected more often
- Low-performing bullets occasionally get selected (exploration)
- Prevents getting stuck in local optima

### **Cosine Similarity**

```python
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

**Purpose**: Measure semantic similarity between query and bullets

---

## 📈 Performance Tracking

### **Bullet Metrics**

Each bullet tracks:
- `helpful_count`: Times it led to correct decision
- `harmful_count`: Times it led to wrong decision
- `times_selected`: Total times selected
- `success_rate`: `helpful_count / (helpful_count + harmful_count)`

### **Update Stats**

```python
def update_stats(bullet_id, is_helpful):
    bullet = playbook.get_bullet(bullet_id)
    if is_helpful:
        bullet.helpful_count += 1
    else:
        bullet.harmful_count += 1
    bullet.times_selected += 1
```

---

## 🎯 Model Assignments

### **Agent (SimpleFraudAgent)**
- **Model**: GPT-3.5-turbo
- **Purpose**: Fast transaction analysis
- **Input**: Transaction text + optional bullets
- **Output**: Decision (APPROVE/DECLINE)

### **Judge (SimpleJudge & LLMJudge)**
- **Model**: GPT-4o-mini
- **Purpose**: Evaluate correctness
- **Input**: Query, output, ground_truth
- **Output**: `{is_correct: bool, confidence: float}`

### **Reflector (Bullet Generation)**
- **Model**: GPT-4o-mini
- **Purpose**: Generate actionable heuristics
- **Input**: Transaction, predicted, correct, reasoning
- **Output**: `{new_bullet: str, problem_types: [], confidence: float}`

### **Embeddings**
- **Model**: text-embedding-3-small
- **Purpose**: Semantic similarity
- **Dimensions**: 1536

---

## 🚀 Usage Examples

### **Offline Training**

```python
# Load training data
train_set = load_dataset("agents/complex_fraud_detection.json")

# Initialize components
playbook = BulletPlaybook()
reflector = Reflector()
curator = Curator()

# Generate bullets
for item in train_set:
    bullet = await reflector.reflect(
        query=item['query'],
        predicted=item['answer'],
        correct=item['answer'],
        node="fraud_detection"
    )
    
    if bullet:
        curator.merge_bullet(
            content=bullet['new_bullet'],
            node="fraud_detection",
            playbook=playbook,
            source="offline"
        )
```

### **Online Selection**

```python
# Get bullets for transaction
selector = HybridSelector()
bullets, scores = selector.select_bullets(
    query="Transaction: User 'john' attempts $2500 purchase...",
    node="fraud_detection",
    playbook=playbook,
    n_bullets=5,
    source="offline"  # or "online" or None for all
)

# Use bullets as context
agent = SimpleFraudAgent(bullets=[b.content for b in bullets])
decision = agent.analyze(transaction)
```

### **Online Learning**

```python
# Agent makes decision
decision = agent.analyze(transaction)

# Judge evaluates
is_correct, confidence = await judge.judge(
    input_text=transaction,
    output=decision,
    ground_truth=ground_truth
)

# Learn from result
if not is_correct:
    bullet = await reflector.reflect(
        query=transaction,
        predicted=decision,
        correct=ground_truth,
        node="fraud_detection"
    )
    
    if bullet:
        curator.merge_bullet(
            content=bullet['new_bullet'],
            node="fraud_detection",
            playbook=playbook,
            source="online"
        )
```

---

## 📊 Expected Results

### **Vanilla Mode (Baseline)**
- **Accuracy**: ~65-70%
- **Model**: GPT-3.5-turbo
- **Improvements**: None

### **Offline + Online Mode**
- **Accuracy**: ~75-80%
- **Model**: GPT-3.5-turbo + bullets
- **Improvements**: Pre-trained bullets from historical data

### **Online Only Mode**
- **Accuracy**: ~70-75% (improves over time)
- **Model**: GPT-3.5-turbo
- **Improvements**: Real-time learning

---

## 💡 Key Design Decisions

### **Why 5-Stage Selection?**
- **Stage 1**: Ensures bullets are relevant to the agent node
- **Stage 2**: Filters out low-quality bullets
- **Stage 3**: Ensures semantic relevance
- **Stage 4**: Balances multiple factors
- **Stage 5**: Promotes diversity

### **Why Thompson Sampling?**
- Explores promising bullets
- Exploits known good bullets
- Prevents local optima

### **Why Separate Offline/Online Sources?**
- Allows users to choose mode
- Offline: Stable, pre-trained
- Online: Adaptive, real-time

### **Why GPT-3.5-turbo for Agent?**
- 10x faster than GPT-4o
- Sufficient quality for fraud detection
- Lower cost

### **Why GPT-4o-mini for Judge/Reflector?**
- Higher quality than GPT-3.5-turbo
- Cheaper than GPT-4o
- Good balance of cost/quality

---

## 🔧 Configuration

### **HybridSelector Parameters**

```python
selector = HybridSelector(
    quality_threshold=0.3,      # Min success rate
    semantic_threshold=0.5,     # Min similarity
    diversity_weight=0.15,      # Diversity bonus weight
    weights={
        'quality': 0.3,
        'semantic': 0.4,
        'thompson': 0.3
    },
    embedding_model="text-embedding-3-small"
)
```

### **Curator Parameters**

```python
curator = Curator(
    similarity_threshold=0.85  # Above this = duplicate
)
```

---

## 📚 Summary

The ACE system is a **self-improving fraud detection system** that:

1. **Generates** actionable heuristics from successes/failures
2. **Selects** the best bullets using a 5-stage hybrid algorithm
3. **Evaluates** correctness using LLM judge
4. **Deduplicates** bullets to prevent redundancy
5. **Tracks** performance metrics for continuous improvement

**Key Innovation**: Combines quality metrics, semantic similarity, Thompson sampling, and diversity promotion to select the most effective heuristic rules for each transaction.

