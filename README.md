# Failure Propagation Prediction Pattern

### A Graph-Learning-Based Proactive Failure Management Mechanism for Cloud Microservices

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Django 6.0](https://img.shields.io/badge/django-6.0-green.svg)](https://www.djangoproject.com/)
[![License: Patent Pending](https://img.shields.io/badge/license-patent%20pending-orange.svg)](#)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Architecture Diagram](#3-architecture-diagram)
4. [Data Flow Deep Dive](#4-data-flow-deep-dive)
5. [Machine Learning Design](#5-machine-learning-design)
6. [Failure Propagation Logic](#6-failure-propagation-logic)
7. [Preventive Action Strategy](#7-preventive-action-strategy)
8. [Simulation & Testing Framework](#8-simulation--testing-framework)
9. [Technology Stack](#9-technology-stack)
10. [Code Structure Mapping](#10-code-structure-mapping)
11. [Design Decisions & Tradeoffs](#11-design-decisions--tradeoffs)
12. [Future Improvements](#12-future-improvements)

---

## 1. Project Overview

### The Problem

In modern cloud-native architectures, microservices are interconnected through complex dependency graphs. When a single service degrades — due to CPU saturation, latency spikes, or elevated error rates — the failure doesn't stay contained. It propagates along dependency edges to upstream consumers, triggering **cascading failures** that can bring down entire systems within seconds.

**Existing solutions are reactive.** Traditional monitoring stacks (Prometheus + Grafana, PagerDuty, etc.) detect failures *after* they have already occurred and propagated. By the time an alert fires, the cascade is already in motion. Circuit breakers and retry policies mitigate damage but do not *predict* it.

### The Solution

This project implements the **Failure Propagation Prediction Pattern** — a proactive system that:

1. **Models** microservice topologies as directed dependency graphs
2. **Collects** runtime health metrics (CPU, latency, error rate) at each simulation tick
3. **Predicts** per-node failure probabilities using a Graph Learning Agent (MLP with graph-aware embeddings)
4. **Analyzes** cascading propagation risk across the topology
5. **Triggers** preventive actions (isolation, throttling) *before* cascades materialize
6. **Learns** from prediction-vs-actual outcomes to improve accuracy over time

### Why Proactive > Reactive

| Aspect | Reactive Systems | This System (Proactive) |
|---|---|---|
| **Detection** | After failure occurs | Before failure propagates |
| **Scope** | Single-node alerts | System-wide cascade prediction |
| **Response** | Human-triggered | Automated preventive action |
| **Accuracy** | Static thresholds | Learned, adaptive thresholds |
| **Learning** | None | Continuous feedback loop |

---

## 2. System Architecture

The system is composed of **eight core components** organized across a Django REST backend, a pure-Python simulation engine, and a React visualization frontend.

---

### 2.1 API Layer (`backend/api/`)

**Responsibility:** Exposes the simulation engine to external consumers via RESTful HTTP endpoints. Acts as the entry point for all operations — graph initialization, fault injection, simulation execution, and result retrieval.

**Key Files:**
- `views.py` — Six Django REST Framework `APIView` classes handling HTTP request/response
- `services.py` — `SimulationService` singleton that orchestrates all internal components
- `serializers.py` — Input validation for graph topology and fault injection payloads
- `models.py` — Django ORM models for persisting simulation runs and evaluation reports to PostgreSQL/SQLite
- `urls.py` — Route definitions mapping URL paths to view handlers

**Endpoints:**

| Method | Endpoint | Handler | Purpose |
|--------|----------|---------|---------|
| `POST` | `/api/initialize-graph` | `InitializeGraphView` | Load a service topology (manual or synthetic) |
| `POST` | `/api/inject-fault` | `InjectFaultView` | Inject a fault event into the simulation |
| `POST` | `/api/run-simulation` | `RunSimulationView` | Advance the simulation by one tick |
| `GET` | `/api/system-risk` | `SystemRiskView` | Retrieve current system-level risk intelligence |
| `GET` | `/api/evaluation-report` | `EvaluationReportView` | Get prediction-vs-actual evaluation metrics |
| `GET` | `/api/graph-state` | `GraphStateView` | Retrieve the full topology with risk scores |
| `POST` | `/api/run-benchmark` | `RunBenchmarkView` | Execute a full multi-trial benchmark suite |

**Design Pattern:** The `SimulationService` uses the **Singleton pattern** (`__new__` override) to ensure all views share the same simulation state — graph engine, metrics generator, predictor, and logger instances persist across HTTP requests within the same server process.

---

### 2.2 Service Dependency Graph Builder (`backend/simulation/graph_engine.py`)

**Responsibility:** Manages the microservice topology as a `NetworkX.DiGraph`. Provides a clean abstraction over the raw graph, exposing typed operations for adding nodes/edges, querying neighbors, and traversing dependencies.

**Core Class:** `GraphEngine`

**Internal Representation:**
- **Nodes** are stored as `ServiceNode` Pydantic models with attributes: `id`, `service_name`, `criticality_score`, `current_metrics`, `calculated_risk_score`
- **Directed Edges** represent dependencies: an edge `A → B` means "A depends on B." If B fails, A is at risk.
- Each edge carries a `DependencyEdge` model with an `amplification_factor` that governs how strongly a failure at the target amplifies risk at the source.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `build_from_definitions(nodes, edges)` | Clears and rebuilds the graph from lists of `ServiceNode` and `DependencyEdge` |
| `get_upstream_dependents(node_id)` | Returns predecessors — nodes that depend *on* the given node (risk receivers) |
| `get_downstream_dependencies(node_id)` | Returns successors — nodes the given node depends *on* (risk sources) |
| `update_node(node)` | Updates the node's data object in-place within the graph |

**Directionality Convention:**
```
Edge: Web → API  (means: Web depends on API)
If API fails → Web is at risk
get_upstream_dependents("API") returns ["Web"]
get_downstream_dependencies("Web") returns ["API"]
```

---

### 2.3 Runtime Metric Collector (`backend/simulation/metrics_generator.py`)

**Responsibility:** Generates synthetic per-node health metrics at each simulation tick. Simulates both healthy baseline behavior and injected fault conditions.

**Core Classes:**
- `MetricsGenerator` — Produces `MetricTick` objects per node per tick
- `FaultEvent` — Defines a fault injection: target node, fault type, magnitude, and duration

**How Metrics Are Generated:**

1. **Healthy Baseline:** Each node receives randomized nominal values:
   - CPU: `uniform(0.1, 0.4)` with 10% chance of noise bump up to `+0.5`
   - Latency: `uniform(10, 50)ms` plus noise
   - Error Rate: `uniform(0.001, 0.01)` plus noise

2. **Fault Application:** Active `FaultEvent` objects modify metrics:
   - `cpu_spike` → Adds `magnitude` to CPU (capped at 1.0)
   - `latency_spike` → Adds `magnitude` to latency (in ms)
   - `error_spike` → Adds `magnitude` to error rate (capped at 1.0)

3. **Fault Expiry:** Each tick decrements `ticks_remaining`; expired faults are purged.

**Noise Design:** The 10% random noise probability intentionally produces false-positive-inducing spikes, forcing the ML model to learn to distinguish genuine faults from noise — a critical real-world challenge.

---

### 2.4 Graph-Based ML Predictor (`backend/simulation/graph_learning_model.py`)

**Responsibility:** The core intelligence engine. Predicts per-node failure probabilities `P(failure) ∈ [0, 1]` using a graph-aware MLP model that learns from historical outcomes. Incorporates **dynamic attention scoring** to modulate edge weights based on real-time telemetry correlation between connected services.

This is the **Graph Learning Agent** — the centerpiece implementation of the patent.

> **Replaces:** The legacy `RiskPropagationEngine` (BFS + hop decay) and `DeterministicFeedbackLoop` (rule-based parameter tuning).

**Core Class:** `GraphFailurePredictor`

**Dual-Mode Operation:**

| Mode | Trigger | Method |
|------|---------|--------|
| **Cold Start** | `< 10` training samples accumulated | Sigmoid-based heuristic scoring function |
| **Learned** | `≥ 10` samples, model trained | Calibrated MLP neural network |

**Key Capabilities:**
- **Dynamic Attention:** Edge weights are dynamically modulated via Pearson correlation of CPU telemetry over a rolling 10-tick window, amplifying edges between services that exhibit synchronized stress patterns
- **Metric History:** Maintains a per-node rolling window of CPU and latency metrics (`_metric_history`) for temporal correlation analysis

**Public API:**

| Method | Input | Output |
|--------|-------|--------|
| `predict_failure_probabilities(graph_engine)` | Current graph state | Dict with per-node probabilities, system risk, severity, cascade info |
| `train_on_batch(predicted, actual, graph_engine)` | Predictions + ground-truth labels | Sample count + readiness status |
| `update_model()` | *(uses internal buffer)* | Training status, iteration count, accuracy score |
| `save_model(path)` / `load_model(path)` | File path | Persists/restores full model state (including metric history) via `joblib` |

**Detailed operation is covered in [Section 5: Machine Learning Design](#5-machine-learning-design).**

---

### 2.5 Propagation Risk Analyzer (`backend/simulation/risk_aggregation.py`)

**Responsibility:** Given per-node risk scores, computes system-level predictive intelligence: overall system risk, risk concentration, reachable cascade nodes, critical propagation paths, and severity classification.

**Core Class:** `CascadeSeverityScorer`

**Key Computations:**

| Metric | Formula / Method |
|--------|-----------------|
| **System Risk Score** | Criticality-weighted mean of node risk scores |
| **Risk Concentration** | `max(risks) / sum(risks)` — measures localization vs. distribution |
| **Cascade Size** | BFS traversal from high-risk nodes through upstream dependents |
| **Critical Paths** | DFS-based search for longest dependency chains from high-risk origins |
| **Severity Level** | Rule-based classification (see below) |

**Severity Classification Thresholds:**

| Level | Condition |
|-------|-----------|
| `CRITICAL` | `system_score ≥ 0.7` OR `cascade_size ≥ 4` OR `high_risk_count ≥ 3` |
| `HIGH` | `system_score ≥ 0.4` OR `cascade_size ≥ 2` OR `high_risk_count ≥ 2` |
| `MODERATE` | `system_score ≥ 0.1` OR `high_risk_count ≥ 1` |
| `LOW` | All below thresholds |

> **Note:** The `GraphFailurePredictor` internalizes these same severity thresholds in `_classify_severity()` for backward compatibility, so the `CascadeSeverityScorer` serves as the reference implementation.

---

### 2.6 Preventive Action Engine (`backend/simulation/action_engine.py`)

**Responsibility:** Evaluates whether preventive intervention is needed and determines the optimal mitigation strategy by simulating multiple actions on cloned graphs and comparing projected outcomes.

**Core Class:** `PreventiveActionEngine`

**Supported Actions:**

| Action | Graph Modification | Real-World Analog |
|--------|-------------------|-------------------|
| `ISOLATE` | Removes all incoming edges to the target node | Circuit breaker open |
| `THROTTLE` | Halves the `amplification_factor` on incoming edges | Rate limiting / fallback |

**Decision Logic:**
1. Check severity — only act when `HIGH` or `CRITICAL`
2. Identify high-risk nodes from the prediction output
3. Clone the graph and apply `THROTTLE` on a copy
4. Clone the graph and apply `ISOLATE` on another copy
5. Run `predict_failure_probabilities()` on both modified graphs
6. Compare `post_action_cascade_size` and `risk_reduction_percent`
7. Select the action that reduces cascade the most, preferring `THROTTLE` (less destructive) when tied

**Detailed operation is covered in [Section 7: Preventive Action Strategy](#7-preventive-action-strategy).**

---

### 2.7 Logging & Feedback System

Two components handle observability and learning:

#### 2.7.1 Counterfactual Logger (`backend/simulation/counterfactual_logger.py`)

**Responsibility:** Captures what the system *predicted* would happen, tracks what *actually* happened across subsequent simulation ticks, and computes accuracy metrics.

**Lifecycle:**
1. `snapshot_prediction()` — Records predicted severity, cascade size, and high-risk nodes when a non-LOW event is detected
2. `track_actual_tick()` — Called each subsequent tick; records actual node failures and max system risk
3. Tracking ends when severity returns to `LOW` (system stabilizes)
4. `evaluate()` — Compares predicted vs. actual and computes:
   - **Precision:** `TP / (TP + FP)` — of predicted failures, how many actually failed?
   - **Recall:** `TP / (TP + FN)` — of actual failures, how many were predicted?
   - **Cascade Size Error:** `|predicted_size - actual_size|`
   - **Binary Accuracy:** Did we correctly predict cascade vs. no-cascade?
   - **Severity Match:** Did predicted severity match actual inferred severity?

#### 2.7.2 Feedback Learning Loop (`backend/simulation/feedback_learning.py`)

> **DEPRECATED** — Superseded by `GraphFailurePredictor.train_on_batch()` + `update_model()`

The legacy `DeterministicFeedbackLoop` used rule-based parameter tuning (adjusting `amplification_multiplier`, `hop_decay`, and `high_risk_threshold`). It is retained for backward compatibility testing but is no longer used in the active pipeline.

The new feedback mechanism is fundamentally different: instead of tuning propagation parameters, it **retrains the MLP model weights** on accumulated prediction-vs-actual outcome data, directly optimizing the model's ability to predict failures.

---

### 2.8 Simulation / Benchmark Engine (`backend/simulation/benchmark_runner.py`)

**Responsibility:** Executes structured experimental trials comparing three operational modes and produces quantitative resilience metrics for patent validation.

**Core Class:** `BenchmarkRunner`

**Three Modes Compared:**
1. **No Mitigation (Baseline):** Fault injected, no preventive action taken. Measures raw cascade damage.
2. **Static Mitigation:** Fresh `GraphFailurePredictor` per scenario (cold-start only). Measures action engine effectiveness without learning.
3. **Adaptive Learning:** Shared `GraphFailurePredictor` across cycles. Model trains between scenarios. Measures learning improvement over time.

**Detailed operation is covered in [Section 8: Simulation & Testing Framework](#8-simulation--testing-framework).**

---

## 3. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React + Vite)                            │
│                                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │  SimulationPage  │  │  BenchmarkPage   │  │       Components             │   │
│  │  (Interactive)   │  │  (Batch Trials)  │  │  GraphView (Cytoscape.js)    │   │
│  └────────┬─────────┘  └────────┬─────────┘  │  MetricsDashboard (Recharts) │   │
│           │                     │             │  SimulationControls          │   │
│           └─────────┬───────────┘             └──────────────────────────────┘   │
│                     │ HTTP (axios)                                               │
└─────────────────────┼───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DJANGO REST API LAYER                                   │
│                                                                                 │
│  POST /initialize-graph    POST /inject-fault    POST /run-simulation           │
│  GET  /system-risk         GET  /evaluation-report                              │
│  GET  /graph-state         POST /run-benchmark                                  │
│                                                                                 │
│  ┌─────────────────────────────────────────────┐                                │
│  │      SimulationService (Singleton)          │ ◄── Orchestrates all below     │
│  └──────────────────┬──────────────────────────┘                                │
└─────────────────────┼───────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION ENGINE (Pure Python)                           │
│                                                                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────────────┐ │
│  │   Graph     │    │    Metrics       │    │     Graph Learning Agent        │ │
│  │   Engine    │───▶│    Generator     │───▶│     (GraphFailurePredictor)     │ │
│  │ (NetworkX)  │    │ (Fault Inject)   │    │                                 │ │
│  └─────────────┘    └──────────────────┘    │  ┌───────────┐ ┌─────────────┐ │ │
│         │                                    │  │ Feature   │ │ Graph-Aware │ │ │
│         │           ┌──────────────────┐    │  │ Extractor │→│ Embedding   │ │ │
│         │           │  Counterfactual  │    │  └───────────┘ └──────┬──────┘ │ │
│         │           │  Logger          │◄───│                      │        │ │
│         │           │ (Pred vs Actual) │    │  ┌───────────┐ ┌─────▼──────┐ │ │
│         │           └───────┬──────────┘    │  │ Temporal  │→│  MLP /     │ │ │
│         │                   │               │  │ Signals   │ │  Heuristic │ │ │
│         │                   ▼               │  └───────────┘ └─────┬──────┘ │ │
│         │           ┌──────────────────┐    │                      │        │ │
│         │           │ Train on Batch   │◄───│──────────────────────┘        │ │
│         │           │ + Update Model   │    └─────────────────────────────────┘ │
│         │           └──────────────────┘                                        │
│         │                                                                       │
│         │           ┌──────────────────┐    ┌──────────────────┐                │
│         └──────────▶│ Preventive       │    │  Benchmark       │                │
│                     │ Action Engine    │    │  Runner          │                │
│                     │ (ISOLATE/THRTL)  │    │ (Multi-Trial)    │                │
│                     └──────────────────┘    └──────────────────┘                │
│                                                                                 │
└────────────────────────────────────────────────────────────┬────────────────────┘
                                                             │
                                                             ▼
                                                   ┌──────────────────┐
                                                   │  PostgreSQL /    │
                                                   │  SQLite          │
                                                   │  (Django ORM)    │
                                                   └──────────────────┘
```

**Data Flow Summary:**
```
Metric Collection → Graph Update → Failure Prediction → Risk Analysis
    → Preventive Action → Counterfactual Logging → Model Training → Loop
```

---

## 4. Data Flow Deep Dive

### Step-by-Step Lifecycle of a Simulation Tick

The following sequence executes every time `POST /api/run-simulation` is called (or each iteration inside `BenchmarkRunner`):

#### Step 1: Tick Advancement & Metric Generation
```
SimulationService.run_simulation()
  └─▶ self.current_tick += 1
  └─▶ MetricsGenerator.generate_metrics(tick_id, nodes)
       ├── Generate healthy baselines (randomized per node)
       ├── Apply active FaultEvents (cpu_spike / latency_spike / error_spike)
       └── Expire faults with ticks_remaining == 0
```
**Output:** `Dict[node_id → MetricTick]` containing `cpu_utilization`, `latency_ms`, `error_rate`.

#### Step 2: Graph State Update
```
For each node:
  └─▶ node.current_metrics = metrics[node.id]
  └─▶ graph_engine.update_node(node)
```
The graph now reflects the latest health snapshot.

#### Step 3: Failure Probability Prediction
```
GraphFailurePredictor.predict_failure_probabilities(graph_engine)
  ├── 1. Build adjacency list + edge weight map
  ├── 2. Extract 6-dim raw features per node:
  │        [cpu, log_latency, error_rate, criticality, degree, dep_count]
  ├── 3. Compute 12-dim graph-aware embeddings with DYNAMIC ATTENTION:
  │        concat(node_features, dynamic_weighted_mean(neighbor_features))
  │        Edge weights modulated by Pearson correlation of 10-tick CPU history
  ├── 4. Append 2-dim temporal signals:
  │        [previous_cpu, delta_cpu]
  ├── 5. Predict (14-dim input vector):
  │        IF trained → MLP.predict_proba() (learned mode)
  │        ELSE       → sigmoid heuristic   (cold_start mode)
  ├── 6. Update temporal state + metric history (10-tick rolling window)
  └── 7. Build output: system_risk, severity, cascade analysis
```

#### Step 4: Cascade Analysis
```
GraphFailurePredictor.predict_cascade(graph_engine, probabilities)
  ├── Filter nodes where P(failure) ≥ high_risk_threshold (0.7)
  ├── Trace propagation paths through upstream dependents
  ├── Compute propagation_risk_score using nx.descendants()
  └── Compute system_failure_probability = affected_count / total_count
```

#### Step 5: Counterfactual Logging
```
IF severity != LOW AND not already tracking:
  └─▶ CounterfactualLogger.snapshot_prediction(tick, intelligence, risks)
       Records: predicted severity, cascade size, high-risk node set

ELSE IF currently tracking:
  └─▶ CounterfactualLogger.track_actual_tick(tick, intelligence, risks)
       Records: actual failed nodes, max system risk observed
       IF severity returns to LOW → stop tracking → trigger training
```

#### Step 6: Model Training (triggered on tracking completion)
```
SimulationService._train_and_persist()
  ├── Build actual_labels: {node_id: True/False} from logger
  ├── predictor.train_on_batch(predicted, actual, graph_engine)
  │    └── Accumulates (feature_vector, label) pairs into buffer
  ├── IF buffer_size ≥ 10:
  │    └── predictor.update_model()
  │         └── Trains calibrated MLP on accumulated data
  └── Persist evaluation metrics to database
```

#### Step 7: Response Assembly
```
Return {
  tick: current_tick,
  intelligence: { system_risk_score, severity_level, cascade_size, ... },
  risks: { node_id: failure_probability, ... }
}
```

---

## 5. Machine Learning Design

### Model Architecture

The Graph Learning Agent uses a **two-layer MLP (Multi-Layer Perceptron)** with probability calibration:

```
Input Layer (14 features)
    │
    ▼
Hidden Layer 1 (16 neurons, ReLU activation)
    │
    ▼
Hidden Layer 2 (8 neurons, ReLU activation)
    │
    ▼
Output Layer (2 classes: [P(healthy), P(failure)])
    │
    ▼
Platt Scaling (Sigmoid Calibration via CalibratedClassifierCV)
    │
    ▼
Calibrated P(failure) ∈ [0, 1]
```

**Configuration:** `MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=300, warm_start=True)`

### Input Feature Vector (14 dimensions)

| Index | Feature | Source | Dimension |
|-------|---------|--------|-----------|
| 0 | `cpu_utilization` | `MetricTick` | Node |
| 1 | `latency_normalized` | `log1p(latency) / log1p(1000)` | Node |
| 2 | `error_rate` | `MetricTick` | Node |
| 3 | `criticality_score` | `ServiceNode` | Node |
| 4 | `degree` | upstream + downstream count | Structural |
| 5 | `dep_count` | downstream dependency count | Structural |
| 6–11 | `weighted_mean(neighbor_features)` | Edge-weighted aggregation | Graph |
| 12 | `previous_cpu` | Prior tick CPU cache | Temporal |
| 13 | `delta_cpu` | `current_cpu - previous_cpu` | Temporal |

### Graph-Aware Embedding Computation with Dynamic Attention

For each node, the embedding is the concatenation of its own features and a **dynamically weighted** aggregation of its neighbors' features:

```
embedding(v) = concat( features(v), Σ(w̃_e · features(u)) / Σ(w̃_e) )
                                     u ∈ N(v)                u ∈ N(v)

where:
  N(v)  = bidirectional neighborhood (upstream dependents + downstream deps)
  w̃_e  = dynamic_attention(v, u, amplification_factor)
```

#### Dynamic Attention Scoring (`_calculate_dynamic_attention`)

Edge weights are no longer static `amplification_factor` values. Instead, they are **dynamically modulated** based on real-time telemetry correlation between connected services:

```python
# 1. Retrieve 10-tick rolling CPU history for both nodes
cpu_a = metric_history[node_a]["cpu"]  # last 10 ticks
cpu_b = metric_history[node_b]["cpu"]  # last 10 ticks

# 2. Compute Pearson correlation coefficient
r = np.corrcoef(cpu_a, cpu_b)[0, 1]

# 3. Dynamic multiplier: amplify edges between correlated services
dynamic_weight = base_weight * (1.0 + max(0.0, correlation))
```

**Behavior:**

| Correlation (r) | Multiplier | Interpretation |
|-----------------|------------|----------------|
| `r ≈ 1.0` (high positive) | `base × 2.0` | Services degrade in sync → doubled attention (likely causal link) |
| `r ≈ 0.0` (uncorrelated) | `base × 1.0` | No telemetry correlation → static amplification factor |
| `r < 0.0` (negative) | `base × 1.0` | Anti-correlated → clamped to base (no reduction) |
| `< 5 ticks history` | `base × 1.0` | Insufficient data → fallback to static weight |

**Why This Matters:** In real microservice deployments, a database under CPU stress often causes correlated CPU spikes in its API consumers due to blocked threads and connection pool exhaustion. Dynamic attention automatically discovers these runtime coupling patterns and amplifies the corresponding edges, making cascade predictions more accurate than static amplification factors alone.

This replaces explicit BFS propagation with a learned structural context enhanced by runtime telemetry correlation — the model learns *how* failures propagate through the graph structure rather than following hand-coded decay rules.

### Output Dictionary

```python
{
    "node_failure_probabilities": {"web": 0.12, "api": 0.45, "db": 0.89},
    "system_risk_score": 0.52,           # Criticality-weighted mean
    "high_risk_nodes": ["db"],           # P(failure) ≥ 0.7
    "predicted_affected_nodes": ["db"],  # Cascade-reachable high-risk
    "cascade_size": 1,
    "propagation_paths": [["db", "api"]],
    "propagation_risk_score": 0.34,
    "system_failure_probability": 0.33,
    "severity_level": "MODERATE",
    "prediction_mode": "cold_start"      # or "learned"
}
```

### Cold-Start Heuristic (Fallback)

Before `≥ 10` training samples are collected, the agent uses a sigmoid-based scoring function:

```python
raw_score = 0.0
if cpu > 0.8:       raw_score += (cpu - 0.8) * 2.0
if latency > 0.5:   raw_score += (latency - 0.5) * 0.6
if error_rate > 0.05: raw_score += min(0.4, (error_rate - 0.05) * 4.0)

raw_score *= criticality_score
P(failure) = 1 / (1 + exp(-5 * (raw_score - 0.5)))
```

### Training Pipeline

```
1. CounterfactualLogger finishes tracking an event
2. Extract {node_id: bool} actual failure labels
3. predictor.train_on_batch(predicted_probs, actual_labels, graph_engine)
   └── Re-extracts features, builds embeddings, appends to buffer
4. IF buffer_size ≥ 10:
   └── predictor.update_model()
        ├── Ensure both classes represented (synthetic minority if needed)
        ├── Train MLPClassifier with warm_start=True
        ├── Calibrate via CalibratedClassifierCV (StratifiedKFold, sigmoid)
        └── Set _is_trained = True → switches to "learned" mode
```

---

## 6. Failure Propagation Logic

### How Failures Spread Across Nodes

The system models failure propagation through the dependency graph structure:

1. **Dependency Direction:** An edge `A → B` means A depends on B. If B fails, A *receives* risk.
2. **Amplification Factors:** Each edge carries a static `amplification_factor` (range: `0.5–2.5` in benchmarks), which is then **dynamically modulated** at runtime via Pearson-correlation-based attention scoring.
3. **Dynamic Attention:** When two connected services exhibit correlated CPU degradation over a 10-tick window, the edge weight between them is amplified up to 2× the base value. This allows the system to automatically discover and strengthen causal propagation paths based on live telemetry.
4. **Bidirectional Neighborhood:** For embedding computation, both upstream (dependents) and downstream (dependencies) neighbors are aggregated. This gives each node contextual awareness of stress in *both* directions.

### Risk Scoring Mechanism

**System Risk Score** (implemented in `_build_output()`):
```
system_risk = Σ(P(failure_i) × criticality_i) / Σ(criticality_i)
```
This weights failure probabilities by service importance — a critical database with `P(failure) = 0.8` contributes more than a non-critical frontend with the same probability.

**Propagation Risk Score** (implemented in `predict_cascade()`):
```
For each node v:
  reachable = |descendants in reversed graph|  (upstream services at risk)
  contribution = P(failure_v) × reachable

propagation_risk = Σ(contribution_v) / total_nodes
```

### Legacy vs. Current Propagation

| Aspect | Legacy (`propagation.py`) | Current (`graph_learning_model.py`) |
|--------|--------------------------|--------------------------------------|
| Method | BFS traversal with hop decay | Learned embeddings + MLP |
| Edge Weights | Static `amplification_factor` only | Dynamic attention (Pearson correlation × static factor) |
| Risk decay | `risk × amp × decay_factor` per hop (max 5 hops) | Implicitly learned via dynamic neighbor aggregation |
| Adaptability | Static parameters | Model weights + dynamic edge weights adapt over time |
| Temporal Awareness | None | 10-tick rolling CPU/latency history per node |
| Status | Deprecated (retained for testing) | Active |

---

## 7. Preventive Action Strategy

### Action Types

| Action | Implementation | Graph Modification | Real-World Analog |
|--------|---------------|-------------------|-------------------|
| **ISOLATE** | `graph.remove_edge(src, dst)` for all incoming edges | Completely disconnects the failing node from all dependents | Opening a circuit breaker — dependents immediately stop calling the failing service |
| **THROTTLE** | `edge.amplification_factor *= 0.5` for all incoming edges | Reduces the failure amplification by 50% | Rate limiting — reducing traffic to the failing service, using fallback responses |

### Decision Logic (`trigger_proactive_defense`)

```
1. IF severity ∉ {HIGH, CRITICAL} → take no action (severity below threshold)
2. Identify high_risk_nodes from prediction output
3. IF no high-risk nodes identified → take no action

4. Clone graph → apply THROTTLE → predict on modified graph
5. Clone graph → apply ISOLATE  → predict on modified graph

6. Compare:
   IF isolate.cascade_size < throttle.cascade_size → select ISOLATE
   ELSE IF throttle.risk_reduction > 0% → select THROTTLE (less destructive)
   ELSE → select ISOLATE (fallback)

7. Return evaluation with:
   - baseline vs. post-action cascade size
   - risk_reduction_percent
   - severity_before vs. severity_after
```

### Tradeoff Analysis

| Factor | THROTTLE | ISOLATE |
|--------|----------|---------|
| **Disruption** | Low — service still partially reachable | High — service completely unreachable |
| **Risk Reduction** | Moderate — reduces amplification | Maximum — eliminates propagation path |
| **Availability** | Preserved (degraded performance) | Sacrificed for protection |
| **Use Case** | Gradual degradation, load spikes | Critical failures, security events |

---

## 8. Simulation & Testing Framework

### Benchmark Runner Architecture

The `BenchmarkRunner` executes structured experiments across three tracks to produce comparative resilience metrics.

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_services` | 12 | Nodes in synthetic graph |
| `density` | 0.4 | Edge probability (Erdős–Rényi model) |
| `fault_type` | `cpu_spike` | Injected fault category |
| `fault_duration` | 4 ticks | How long the fault persists |
| `learning_cycles` | 15 | Scenarios per trial (adaptive track) |
| `repeated_trials` | 2 | Independent trial repetitions |

### How Failures Are Injected

1. A random `target_node` is selected per scenario
2. A `FaultEvent` with `magnitude=1.0` is injected (maximum severity)
3. The simulation runs for up to 15 ticks
4. Tracking begins when severity exceeds `LOW`; stops when severity returns to `LOW`

### Metrics Evaluated

| Metric | Computation | Purpose |
|--------|-------------|---------|
| **Average Cascade Size** | Mean of actual cascaded node count | Measures damage scope |
| **Average Precision** | Mean of `TP / (TP + FP)` per scenario | Are predictions reliable? |
| **Average Recall** | Mean of `TP / (TP + FN)` per scenario | Are all failures caught? |
| **Cascade Reduction %** | Cross-track: `(baseline_cascade - mitigated_cascade) / baseline_cascade × 100` | Action engine effectiveness (computed across tracks, not per-scenario) |
| **Stabilization Cycles** | Ticks until severity returns to `LOW` | Mean Time To Recovery (MTTR proxy) |
| **Severity Distribution** | Count of `CRITICAL / HIGH / MODERATE / LOW` outcomes | Overall system resilience profile |
| **Convergence Trend** | Per-cycle precision averaged across trials | Does the model improve over time? |

### Test Suite

| Test File | Scope | Test Count |
|-----------|-------|------------|
| `tests/test_graph_learning_model.py` | ML model: features, embeddings, cold-start, training, persistence, temporal, end-to-end | 16 tests |
| `tests/test_simulation.py` | Core simulation: topology integrity, risk propagation, fault injection | 3 tests |
| `tests/test_action_engine.py` | Preventive actions: isolation, throttling, mitigation evaluation | ~5 tests |
| `tests/test_counterfactual_logger.py` | Logging: snapshot, tracking, evaluation metrics | ~5 tests |
| `tests/test_risk_aggregation.py` | Risk analysis: system risk, concentration, cascade paths, severity | ~5 tests |
| `backend/tests/test_benchmark_runner.py` | Benchmark: multi-trial execution, metric aggregation | ~3 tests |
| `backend/tests/test_feedback_learning.py` | Legacy feedback loop (deprecated) | ~5 tests |

---

## 9. Technology Stack

### Backend

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core language |
| **Django** | 6.0+ | Web framework, ORM, admin |
| **Django REST Framework** | 3.15+ | RESTful API layer |
| **django-cors-headers** | 4.4+ | Cross-origin requests for frontend |
| **PostgreSQL** / **SQLite** | — | Persistent storage (configurable via env vars) |
| **psycopg2-binary** | 2.9+ | PostgreSQL adapter |
| **Pydantic** | 2.8+ | Data validation for simulation models |

### Machine Learning & Graph Processing

| Technology | Version | Purpose |
|------------|---------|---------|
| **NetworkX** | 3.3+ | Directed graph modeling, BFS/DFS traversal, descendant computation |
| **NumPy** | 1.26+ | Feature vector computation, matrix operations |
| **scikit-learn** | 1.5+ | `MLPClassifier`, `CalibratedClassifierCV`, `StratifiedKFold` |
| **joblib** | 1.4+ | Model serialization / deserialization |

### Frontend

| Technology | Version | Purpose |
|------------|---------|---------|
| **React** | 18.2+ | UI framework |
| **Vite** | 5.0+ | Build tool / dev server |
| **Cytoscape.js** | 3.28+ | Interactive graph topology visualization |
| **Recharts** | 2.10+ | Metrics charts and dashboards |
| **Axios** | 1.6+ | HTTP client for API communication |
| **Tailwind CSS** | 3.4+ | Utility-first styling |
| **Lucide React** | 0.300+ | Icon library |
| **React Router** | 6.20+ | Client-side routing |

### Testing

| Technology | Purpose |
|------------|---------|
| **pytest** | Test runner and assertion framework |

---

## 10. Code Structure Mapping

```
microservice-failure-prediction/
│
├── backend/                          # Django project root
│   ├── core/                         # Django project configuration
│   │   ├── settings.py               # DB config, CORS, installed apps
│   │   ├── urls.py                   # Root URL routing → api/urls.py
│   │   ├── wsgi.py                   # WSGI entry point
│   │   └── asgi.py                   # ASGI entry point
│   │
│   ├── api/                          # REST API Layer
│   │   ├── views.py                  # 6 APIView classes (entry points)
│   │   ├── services.py               # SimulationService (orchestrator singleton)
│   │   ├── serializers.py            # Input validation (DRF serializers)
│   │   ├── models.py                 # Django ORM: ServiceNode, SimulationRun, etc.
│   │   ├── urls.py                   # API route definitions
│   │   └── admin.py                  # Django admin registration
│   │
│   ├── simulation/                   # Pure Python Simulation Engine
│   │   ├── models.py                 # Pydantic: MetricTick, ServiceNode, DependencyEdge
│   │   ├── graph_engine.py           # NetworkX DiGraph wrapper
│   │   ├── graph_learning_model.py   # ★ GraphFailurePredictor (Graph Learning Agent)
│   │   ├── metrics_generator.py      # Synthetic metric + fault injection
│   │   ├── action_engine.py          # PreventiveActionEngine (ISOLATE/THROTTLE)
│   │   ├── risk_aggregation.py       # CascadeSeverityScorer
│   │   ├── counterfactual_logger.py  # Prediction vs. actual tracking
│   │   ├── benchmark_runner.py       # Multi-trial benchmark orchestrator
│   │   ├── propagation.py            # [DEPRECATED] BFS risk propagation
│   │   └── feedback_learning.py      # [DEPRECATED] Rule-based parameter tuning
│   │
│   ├── tests/                        # Backend-specific tests
│   │   ├── test_benchmark_runner.py
│   │   └── test_feedback_learning.py
│   │
│   ├── manage.py                     # Django management script
│   ├── run_benchmark.py              # CLI benchmark runner
│   └── requirements.txt              # Python dependencies
│
├── frontend/                         # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx                   # Root component with routing
│   │   ├── main.jsx                  # Vite entry point
│   │   ├── pages/
│   │   │   ├── SimulationPage.jsx    # Interactive simulation UI
│   │   │   └── BenchmarkPage.jsx     # Batch benchmark execution UI
│   │   ├── components/
│   │   │   ├── GraphView.jsx         # Cytoscape.js topology visualization
│   │   │   ├── MetricsDashboard.jsx  # Recharts risk/metric charts
│   │   │   └── SimulationControls.jsx # Fault injection + tick controls
│   │   └── services/
│   │       └── api.js                # Axios HTTP client wrapper
│   ├── package.json
│   └── vite.config.js
│
├── tests/                            # Cross-cutting test suite
│   ├── test_graph_learning_model.py  # ★ Comprehensive ML model tests (16 cases)
│   ├── test_simulation.py            # Core simulation integration tests
│   ├── test_action_engine.py         # Preventive action tests
│   ├── test_counterfactual_logger.py # Logging + evaluation tests
│   └── test_risk_aggregation.py      # Risk analysis tests
│
├── package.json                      # Root JS dependencies
└── README.md                         # ← You are here
```

### Component → File Mapping

| Architecture Component | Primary File(s) |
|----------------------|-----------------|
| API Entry Points | `api/views.py`, `api/urls.py` |
| Orchestration Layer | `api/services.py` (SimulationService) |
| Dependency Graph | `simulation/graph_engine.py`, `simulation/models.py` |
| Metric Collection | `simulation/metrics_generator.py` |
| ML Prediction Engine | `simulation/graph_learning_model.py` |
| Risk Aggregation | `simulation/risk_aggregation.py` |
| Preventive Actions | `simulation/action_engine.py` |
| Observation & Learning | `simulation/counterfactual_logger.py` |
| Benchmark System | `simulation/benchmark_runner.py`, `run_benchmark.py` |
| Data Persistence | `api/models.py` (Django ORM) |
| Frontend Visualization | `frontend/src/` (React + Cytoscape.js + Recharts) |

---

## 11. Design Decisions & Tradeoffs

### Why Graph-Based Approach?

**Decision:** Model the microservice topology as a directed graph and embed structural information into the ML feature space.

**Rationale:**
- Microservice dependencies naturally form directed graphs
- Failure propagation follows graph edges — structure is *the* causal mechanism
- Graph-aware embeddings allow the model to learn propagation patterns implicitly rather than relying on hand-coded BFS traversal rules
- Edge amplification factors capture heterogeneous dependency strengths

**Tradeoff:** The current weighted-mean neighbor aggregation is a single-hop operation. However, the **dynamic attention scoring** mechanism partially compensates by amplifying edges where runtime telemetry indicates synchronized degradation — effectively surfacing multi-hop causal chains through correlation rather than explicit graph traversal. A full Graph Neural Network would capture deeper structural patterns natively but at higher computational cost.

### Why Proactive Prediction?

**Decision:** Predict failures *before* they cascade, rather than react to them after detection.

**Rationale:**
- Reactive systems (circuit breakers, retries) activate *during* failure — by then, cascade damage is already in progress
- Proactive prediction allows preventive action (throttle/isolate) *before* the cascade reaches critical mass
- The prediction-action loop operates at simulation-tick granularity, enabling sub-second intervention in production

**Tradeoff:** Proactive systems risk false positives — taking unnecessary preventive action on healthy services. The counterfactual logging and precision/recall metrics directly measure and mitigate this risk.

### Why MLP Over GNN?

**Decision:** Use a scikit-learn `MLPClassifier` with hand-crafted graph embeddings rather than a full Graph Neural Network (PyTorch Geometric, DGL).

**Rationale:**
- Minimal dependency footprint (no PyTorch/CUDA required)
- Fast training on small datasets (tens to hundreds of samples)
- Probability calibration via `CalibratedClassifierCV` produces well-calibrated P(failure) outputs
- Sufficient for TRL-4 (simulated environment) validation
- `warm_start=True` enables incremental learning without full retraining

**Tradeoff:** GNNs would capture multi-hop propagation, graph isomorphism, and attention-weighted neighborhoods more powerfully. This is planned for future work.

### Why Simulated Environment?

**Decision:** Validate the pattern in a fully simulated environment (synthetic graphs, synthetic metrics, injected faults) rather than deploying to live infrastructure.

**Rationale:**
- Controlled experiments with reproducible results (seeded random generation)
- Ability to run hundreds of scenarios in seconds
- No risk to production systems during research validation
- Patent claims require demonstrable results, not production deployment

**Limitations:**
- Synthetic metrics may not capture real-world failure distributions
- Synthetic graphs may not reflect production topology complexity
- No network latency, containerization overhead, or resource contention effects

---

## 12. Future Improvements

### Short-Term Enhancements

- **Graph Neural Networks:** Replace MLP + hand-crafted embeddings with GCN/GAT/GraphSAGE for multi-hop structural learning
- **Multi-Variate Dynamic Attention:** Extend the current CPU-based Pearson correlation to incorporate latency and error rate correlation (the `_metric_history` already stores normalized latency for this purpose)
- **Richer Feature Space:** Add memory utilization, disk I/O, network throughput, request queue depth, and container restart counts
- **Ensemble Models:** Combine probabilistic (MLP), graph-structural (GNN), and time-series (LSTM) predictions

### Medium-Term Integration

- **Kubernetes Integration:** Deploy as a sidecar or operator that reads metrics from Prometheus/OpenTelemetry and topology from service mesh (Istio/Linkerd)
- **Service Mesh Hooks:** Implement ISOLATE as Istio `VirtualService` fault injection and THROTTLE as `DestinationRule` circuit breakers
- **Real-Time Streaming:** Replace tick-based simulation with Kafka/NATS event streams for continuous monitoring
- **Multi-Cluster Support:** Extend the graph model to span multiple Kubernetes clusters with cross-cluster dependency edges

### Long-Term Vision

- **Reinforcement Learning:** Train the action engine via RL (reward = cascade prevention, penalty = unnecessary isolation) instead of rule-based heuristics
- **Explainable AI:** Provide human-readable explanations for predictions ("DB failure probability is 89% because CPU has risen 60% in 3 ticks and 2 upstream services show correlated latency spikes")
- **Digital Twin:** Mirror production topology and metrics in real-time, running continuous what-if simulations alongside the live system
- **Federated Learning:** Enable multiple organizations to collaboratively train failure prediction models without sharing proprietary topology or metric data

---

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Run Benchmarks (CLI)

```bash
cd backend
python run_benchmark.py
```

### Run Tests

```bash
cd backend
pytest ../tests/ -v
pytest tests/ -v
```

---

## License

Patent Pending — *"Failure Propagation Prediction Pattern: A Graph-Learning-Based Proactive Failure Management Mechanism for Cloud Microservices"*

---

<p align="center"><sub>Built for patent validation and research demonstration at TRL-4 (simulated environment).</sub></p>
