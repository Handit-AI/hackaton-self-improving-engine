# Database Schema Design for ACE System (Revised)

## Overview
Learning approach: **Input Patterns → Bullet Associations**
- Store input patterns with embeddings
- Track which bullets work best for each pattern
- Use similarity search to find similar inputs
- Select bullets based on historical effectiveness for that pattern

---

## Core Concept

```
Input Query (with embedding)
    ↓
Find similar historical inputs
    ↓
Look up bullets that worked best for those inputs
    ↓
Use those bullets for this query
```

---

## Revised Tables

### 1. **input_patterns** - Store input patterns

Store normalized input patterns/queries with embeddings for similarity matching.

```sql
CREATE TABLE input_patterns (
    id SERIAL PRIMARY KEY,
    
    -- Input data
    query_text TEXT NOT NULL,
    query_embedding REAL[] NOT NULL,  -- 1536 dimensions
    
    -- Normalized features
    normalized_features JSONB NULL,  -- Extracted features (amount ranges, merchant types, etc.)
    
    -- Metadata
    frequency INTEGER DEFAULT 1,  -- How often we see this pattern
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Classification
    problem_type VARCHAR(100) NULL,  -- Categorized pattern type
    
    -- Constraints
    CONSTRAINT valid_embedding CHECK (array_length(query_embedding, 1) = 1536)
);

-- Indexes
CREATE INDEX idx_input_patterns_embedding ON input_patterns USING GIST (
    (query_embedding::vector)  -- Vector similarity search (if using pgvector)
    -- OR use array indexing for simple approach
);
CREATE INDEX idx_input_patterns_frequency ON input_patterns(frequency DESC);
CREATE INDEX idx_input_patterns_problem_type ON input_patterns(problem_type);
```

---

### 2. **bullets** - Store learned bullets

```sql
CREATE TABLE bullets (
    id VARCHAR(50) PRIMARY KEY,
    
    -- Content
    content TEXT NOT NULL,
    node VARCHAR(100) NOT NULL,  -- Which agent node
    
    -- Performance tracking (global)
    helpful_count INTEGER DEFAULT 0,
    harmful_count INTEGER DEFAULT 0,
    times_selected INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP NULL,
    
    -- Constraints
    CONSTRAINT valid_performance CHECK (helpful_count >= 0 AND harmful_count >= 0)
);

-- Indexes
CREATE INDEX idx_bullets_node ON bullets(node);
CREATE INDEX idx_bullets_performance ON bullets((helpful_count + harmful_count));
```

---

### 3. **bullet_input_effectiveness** - Association table

**This is the key table!** Links bullets to input patterns with performance metrics.

```sql
CREATE TABLE bullet_input_effectiveness (
    id SERIAL PRIMARY KEY,
    
    -- Links
    input_pattern_id INTEGER REFERENCES input_patterns(id) ON DELETE CASCADE,
    bullet_id VARCHAR(50) REFERENCES bullets(id) ON DELETE CASCADE,
    node VARCHAR(100) NOT NULL,  -- Which node this bullet is for
    
    -- Performance for THIS specific input pattern
    helpful_count INTEGER DEFAULT 0,
    harmful_count INTEGER DEFAULT 0,
    times_selected INTEGER DEFAULT 0,
    
    -- Metrics
    first_tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_tested_at TIMESTAMP NULL,
    
    -- Unique constraint: one effectiveness record per bullet-input pair
    UNIQUE(input_pattern_id, bullet_id),
    
    -- Constraints
    CONSTRAINT valid_performance CHECK (helpful_count >= 0 AND harmful_count >= 0)
);

-- Indexes
CREATE INDEX idx_bullet_input_pattern ON bullet_input_effectiveness(input_pattern_id);
CREATE INDEX idx_bullet_input_bullet ON bullet_input_effectiveness(bullet_id);
CREATE INDEX idx_bullet_input_node ON bullet_input_effectiveness(node);
CREATE INDEX idx_bullet_input_success_rate ON bullet_input_effectiveness(
    (helpful_count::float / NULLIF(helpful_count + harmful_count, 0))
);

-- Composite index for fast lookups
CREATE INDEX idx_bullet_input_lookup ON bullet_input_effectiveness(
    input_pattern_id, node, 
    (helpful_count::float / NULLIF(helpful_count + harmful_count, 0))
);
```

---

### 4. **transactions** - Audit trail

```sql
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    
    -- Transaction data
    transaction_data JSONB NOT NULL,
    
    -- Analysis metadata
    mode VARCHAR(50) NOT NULL,  -- 'vanilla', 'offline_ace', 'online_ace'
    
    -- Results
    predicted_decision VARCHAR(20) NOT NULL,
    correct_decision VARCHAR(20) NOT NULL,
    is_correct BOOLEAN GENERATED ALWAYS AS (predicted_decision = correct_decision) STORED,
    
    -- Links
    input_pattern_id INTEGER REFERENCES input_patterns(id) NULL,
    
    -- Timestamps
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transactions_mode ON transactions(mode);
CREATE INDEX idx_transactions_pattern ON transactions(input_pattern_id);
```

---

### 5. **bullet_selections** - Usage tracking

```sql
CREATE TABLE bullet_selections (
    id SERIAL PRIMARY KEY,
    
    -- Links
    transaction_id INTEGER REFERENCES transactions(id) ON DELETE CASCADE,
    bullet_id VARCHAR(50) REFERENCES bullets(id) ON DELETE CASCADE,
    bullet_input_effectiveness_id INTEGER REFERENCES bullet_input_effectiveness(id) ON DELETE SET NULL,
    
    -- Selection details
    node VARCHAR(100) NOT NULL,
    selection_score NUMERIC(10,6) NULL,
    
    -- Timestamps
    selected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_bullet_selections_transaction ON bullet_selections(transaction_id);
CREATE INDEX idx_bullet_selections_bullet ON bullet_selections(bullet_id);
```

---

## How It Works

### **1. New Input Comes In**

```python
# Get input embedding
query = "Transaction: User 'sketchy_alice' attempts $7,500 purchase..."
query_embedding = get_embedding(query)

# Find similar historical inputs
similar_inputs = find_similar_inputs(query_embedding, threshold=0.85)
# Returns: [input_pattern_id_123, input_pattern_id_456, ...]
```

### **2. Find Best Bullets for Those Similar Inputs**

```sql
-- Get bullets that worked best for similar inputs
SELECT 
    b.id,
    b.content,
    b.node,
    AVG(bie.helpful_count::float / NULLIF(bie.helpful_count + bie.harmful_count, 0)) as avg_success_rate,
    SUM(bie.times_selected) as total_times_selected
FROM bullets b
JOIN bullet_input_effectiveness bie ON b.id = bie.bullet_id
WHERE bie.input_pattern_id IN (123, 456, 789)  -- Similar inputs
  AND bie.node = 'behavior_analyzer'
GROUP BY b.id, b.content, b.node
HAVING SUM(bie.helpful_count + bie.harmful_count) > 2  -- At least some evidence
ORDER BY avg_success_rate DESC, total_times_selected DESC
LIMIT 5;
```

### **3. Use Those Bullets**

```python
# Select top bullets
selected_bullets = [bullet1, bullet2, bullet3]

# Analyze with those bullets
result = await agent.analyze_with_bullets(transaction, selected_bullets)

# Record effectiveness
if result['is_correct']:
    update_bullet_input_effectiveness(
        input_pattern_id=current_input_id,
        bullet_id=bullet1.id,
        is_helpful=True
    )
```

### **4. Update Pattern Association**

```python
# When we learn a bullet works well for this input pattern
record_bullet_effectiveness(
    input_pattern_id=current_input_id,
    bullet_id=bullet1.id,
    node='behavior_analyzer',
    helpful=True
)
```

---

## Key Benefits

✅ **Pattern-based learning**: Learn which bullets work for specific input patterns  
✅ **Similarity matching**: Reuse bullets for similar inputs  
✅ **Better targeting**: Bullets selected based on historical effectiveness for that pattern  
✅ **Adaptive**: Better bullets naturally rise to the top for each pattern  
✅ **Granular tracking**: Track effectiveness at bullet-input pattern level  

---

## Example Use Case

```
Input: "VPN + crypto merchant + large amount + new user"
    ↓
Find similar inputs: [pattern_123, pattern_456]
    ↓
Look up bullets that worked for those patterns:
  - bullet_789: "VPN from high-risk countries + crypto = fraud" (success_rate: 0.92)
  - bullet_790: "New users + large amounts = suspicious" (success_rate: 0.88)
    ↓
Use those bullets for this input
    ↓
Track: Did they work? Update bullet_input_effectiveness
```

---

## Schema Summary

**Core Tables:**
1. **input_patterns** - Store input patterns with embeddings
2. **bullets** - Store learned bullets (simplified)
3. **bullet_input_effectiveness** - KEY TABLE: Links bullets to inputs with metrics
4. **transactions** - Audit trail
5. **bullet_selections** - Usage tracking

**Key Relationship:**
```
input_patterns ←→ bullet_input_effectiveness ←→ bullets
                      (with effectiveness metrics)
```

Does this approach make more sense? We're learning "which bullets work best for which input patterns"!
