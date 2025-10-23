# ACE Self-Improving Engine - Implementation Plan

## Overview
Building a fraud detection system with 3 operating modes:
1. **Vanilla Agent** - No learning, baseline performance (55-65% accuracy)
2. **Offline ACE** - Pre-trained playbook (75-85% accuracy)
3. **Online ACE** - Real-time learning (70% → 90% accuracy improvement)

## System Architecture

```
[Dataset] → [FastAPI Orchestration] → [Jose's Multi-Agent System] → [ACE System] → [Oliver's UI]
```

## Implementation Phases

### Phase 1: Foundation & Data Layer (2-3 hours)
**Goal**: Create the dataset and core storage infrastructure

#### Tasks:
1. **Dataset Creation** (1.5 hours)
   - Create `data/fraud_transactions.json`
   - 40-50 labeled transactions
   - Mix: 20 APPROVE (40%), 30 DECLINE (60%)
   - Difficulty levels: easy, medium, hard, very_hard
   - Categories: obvious fraud, obvious legit, ambiguous, edge cases

2. **Database Schema** (1 hour)
   - Create Alembic migration for bullets table
   - Schema: id, content, node, problem_types, performance metrics
   - Schema: timestamps, embeddings, source tracking
   - Update `database.py` with models

3. **BulletPlaybook Class** (30 minutes)
   - Create `ace/bullet_playbook.py`
   - Per-node bullet organization
   - CRUD operations
   - Performance tracking
   - JSON serialization

**Deliverable**: Functional dataset + database schema + playbook storage

---

### Phase 2: ACE Core Components (2-3 hours)
**Goal**: Implement the learning engine

#### Tasks:
1. **HybridSelector** (1.5 hours)
   - Create `ace/hybrid_selector.py`
   - 5-stage algorithm:
     - Stage 1: Contextual Filter
     - Stage 2: Quality Filter
     - Stage 3: Semantic Filter
     - Stage 4: Hybrid Scoring (Thompson Sampling)
     - Stage 5: Diversity Promotion
   - Per-node support
   - OpenAI embeddings integration

2. **Reflector** (30 minutes)
   - Create `ace/reflector.py`
   - Generate new bullets from successes/failures
   - Per-node bullet generation
   - GPT-4 integration

3. **Curator** (30 minutes)
   - Create `ace/curator.py`
   - Deduplication logic
   - Similarity checking
   - Quality control

**Deliverable**: Complete ACE learning system

---

### Phase 3: Training Pipeline (1-2 hours)
**Goal**: Implement offline and online training modes

#### Tasks:
1. **Training Pipeline** (1.5 hours)
   - Create `ace/training_pipeline.py`
   - Offline training (pre-train on dataset)
   - Online training (real-time updates)
   - Metrics tracking
   - Per-node playbook management

2. **Integration with Jose's Agents** (30 minutes)
   - Create wrapper for Jose's fraud detection agent
   - Multi-agent result aggregation
   - Node-specific result handling

**Deliverable**: Training pipeline + agent integration

---

### Phase 4: FastAPI Orchestration (2-3 hours)
**Goal**: Build the API layer with 3-mode support

#### Tasks:
1. **API Endpoints** (2 hours)
   - Update `main.py` with ACE endpoints:
     - `POST /api/v1/analyze` - Analyze transaction (3 modes)
     - `POST /api/v1/train` - Run training experiment
     - `POST /api/v1/train-offline` - Pre-train offline playbook
     - `GET /api/v1/playbook/stats` - Get statistics
     - `GET /api/v1/playbook/{node}` - Get node-specific bullets
   - Mode routing logic (vanilla/offline/online)
   - Error handling

2. **Mode Implementation** (1 hour)
   - Vanilla mode: Direct agent call
   - Offline ACE: Use pre-trained playbook
   - Online ACE: Real-time learning
   - State management

**Deliverable**: Complete API with 3-mode support

---

### Phase 5: Testing & Evaluation (1-2 hours)
**Goal**: Validate all modes and measure performance

#### Tasks:
1. **Integration Tests** (1 hour)
   - Create `test_3_modes.py`
   - Test vanilla mode
   - Test offline ACE mode
   - Test online ACE mode
   - Compare results

2. **Pre-training** (30 minutes)
   - Run offline training
   - Generate `offline_playbook.json`
   - Verify playbook quality

3. **Documentation** (30 minutes)
   - Create `INTEGRATION.md`
   - API documentation
   - Usage examples

**Deliverable**: Tested system + documentation

---

### Phase 6: Cloud Deployment (1 hour)
**Goal**: Deploy to Google Cloud Run

#### Tasks:
1. **Update Dockerfile** (15 minutes)
   - Add ACE dependencies
   - Environment variable setup
   - Data volume handling

2. **Cloud Configuration** (15 minutes)
   - Update `cloudbuild.yaml` with new secrets
   - Add OpenAI API key secret
   - Update environment variables

3. **Deployment** (30 minutes)
   - Test locally
   - Deploy to Cloud Run
   - Verify all endpoints

**Deliverable**: Production-ready deployment

---

## Project Structure

```
hackaton-self-improving-engine/
├── alembic/                    # Database migrations
├── data/                       # Dataset and playbooks
│   ├── fraud_transactions.json
│   ├── offline_playbook.json
│   └── online_playbook.json
├── ace/                        # ACE System
│   ├── __init__.py
│   ├── bullet_playbook.py
│   ├── hybrid_selector.py
│   ├── reflector.py
│   ├── curator.py
│   └── training_pipeline.py
├── agents/                     # Agent integration
│   └── fraud_detection_agent.py  # Jose's agent wrapper
├── tests/                      # Test files
│   └── test_3_modes.py
├── main.py                     # FastAPI app
├── config.py
├── database.py
├── requirements.txt
├── Dockerfile
├── cloudbuild.yaml
├── INTEGRATION.md
└── README.md
```

---

## Dependencies to Add

```python
# Update requirements.txt
openai>=1.0.0          # For LLM and embeddings
scipy>=1.11.0          # For Thompson Sampling
numpy>=1.24.0          # For numerical operations
```

---

## Success Criteria

- [ ] Dataset: 40-50 transactions created
- [ ] Database: Bullets table with migrations
- [ ] ACE Components: All 5 components implemented
- [ ] 3 Modes: Vanilla, Offline, Online working
- [ ] API: All endpoints functional
- [ ] Training: Offline playbook pre-trained
- [ ] Integration: Jose's agent integrated
- [ ] Testing: All modes tested
- [ ] Deployment: Cloud Run deployed
- [ ] Documentation: Complete and clear

---

## Timeline Estimate

- **Phase 1**: 2-3 hours
- **Phase 2**: 2-3 hours
- **Phase 3**: 1-2 hours
- **Phase 4**: 2-3 hours
- **Phase 5**: 1-2 hours
- **Phase 6**: 1 hour

**Total**: 9-14 hours (planning for 7.5 hours of focused work)

---

## Next Steps

1. Start with Phase 1: Create dataset and database schema
2. Implement core ACE components
3. Build training pipeline
4. Create API endpoints
5. Test and deploy

Ready to begin? Let's start with Phase 1!

