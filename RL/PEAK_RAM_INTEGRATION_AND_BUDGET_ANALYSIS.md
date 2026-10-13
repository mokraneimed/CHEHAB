# Technical Report: Peak Memory Estimator Integration & Budget Selection

## 1. Executive Summary & Objective

The objective of this work is to integrate a **Peak RAM Memory Constraint** into the CHEHAB Reinforcement Learning (RL) circuit rewriting framework, replacing the legacy **Remaining Noise Budget** constraint as the primary optimization constraint while preserving backwards compatibility so that either constraint can be selected.

During this integration, two fundamental issues have emerged:
1. **The Mapping Issue**: The empirical profiling dataset provided for memory (`memory_estimator_stats_dataset.csv`) contains measured peak memory values starting around ~800 MB and extending to ~3 GB. However, it does not include the raw S-expressions corresponding to each row (only operation counts and depth). Consequently, exact row-by-row verification between the expressions in the training dataset (`final_llm_dataset.txt`) and the measured memory values is impossible.
2. **The Scale & Budget Discrepancy**: The analytical closed-form estimator provided for CKKS (`ckks_memory_estimator.py`) outputs peak memory values in the range of ~20 MB to ~200 MB (median ~67 MB) on expressions from `final_llm_dataset.txt`. Because of the gap between the measured dataset (~800 MB - 3 GB) and the analytical model (~20 MB - 200 MB), setting appropriate budget thresholds for Lagrangian RL training is ambiguous.

This document details the background, technical implementation, proposed memory metrics, data evidence, the exact step-by-step integration blueprint, and open architectural questions for external review.

---

## 2. Codebase & Branch Context

- **Base Branch**: `peakModel-integration`
- **Upstream Origin**: Forked directly from `mokraneimed/CHEHAB:unified-RL-eval` (Imed's branch with unified evaluation and training).
- **Target Remote**: `origin` (`git@github.com:rayaneSMR/chehab-unified.git`).
- **Core Requirement**: The legacy noise constraint implementation must **not** be overwritten or deleted. The framework must support a configurable constraint metric (default: `memory`, optional: `noise`).

---

## 3. The Analytical Estimator (`ckks_memory_estimator.py`)

### Origin
Provided by Mounir Melzi from branch `dev` of `mounirmelzi/chehab`.

### Formulation
The estimator is a closed-form analytical model. It does not load external datasets or train regression models at runtime. It performs an Abstract Syntax Tree (AST) Directed Acyclic Graph (DAG) traversal to compute active live ciphertexts and Galois rotation keys, and applies the following formula:

```
Peak Memory (in bytes) =
    (1 + G) * SWK
    + sum_over_live_ciphertexts [ 2 * N * (level + 1) * 8 ]
    + (3 + a + c * L + r * [G > 0]) * N * (L + K) * 8
```

Where:
- `N`: Polynomial ring dimension (e.g., 16384, 32768, 65536), selected based on multiplicative depth and total modulus bits.
- `L`: Number of Q primes (levels).
- `K`: Number of P primes (special modulus for key switching, default 2).
- `G`: Number of distinct Galois rotation steps required by the circuit (plus 1 on GPU).
- `SWK`: Size of one key-switching key:
  `SWK = ceil(L / K) * 2 * N * (L + K) * 8 bytes`
- `(a, c, r)`: Overhead coefficients calibrated for two backends:
  - `gpu` (HEonGPU on NVIDIA A100): `a = 8.01788211, c = 1.13505002, r = 5.97810219`
  - `cpu` (Lattigo v5.0.7): `a = 26.50443709, c = 0.72034038, r = 1.49652300`

### Output Unit & Interface
- `CKKSMemoryEstimator(backend="gpu" | "cpu").estimate(expr)` -> returns a single `float` representing peak memory in **Megabytes (MB)**.
- `estimate_expression_memory(expr, backend)` -> returns detailed breakdown: `CKKSParams(N, L, K)`, `rotation_steps`, `galois_keys`, `peak_live_ciphertexts`, and individual terms in bytes.

---

## 4. The Profiled Dataset (`memory_estimator_stats_dataset.csv`)

### Structure & Content
The dataset contains 20,699 profiled rows with the following schema:
- `add`: integer count of addition operations
- `sub`: integer count of subtraction operations
- `multiply_plain`: integer count of plaintext multiplications
- `rotate_rows`: integer count of rotation operations
- `negate`: integer count of negation operations
- `multiply`: integer count of ciphertext multiplications
- `Depth`: total tree depth
- `Multiplicative Depth`: multiplicative depth
- `execution_time (s)`: execution time in seconds
- `Remaining_noise_budget`: legacy noise metric
- `peak_ram_usage (KB)`: measured peak RAM in Kilobytes

### Statistical Distribution of `peak_ram_usage (KB)`
Converting the 20,699 entries to MB and GB:

| Percentile / Metric | Value in KB | Value in MB | Value in GB |
| :--- | :--- | :--- | :--- |
| **Minimum** | 780,364 KB | 762.07 MB | 0.74 GB |
| **p10** | 848,972 KB | 829.07 MB | 0.81 GB |
| **p25** | 876,540 KB | 856.00 MB | 0.84 GB |
| **p50 (Median)** | 951,920 KB | 929.61 MB | 0.91 GB |
| **p75** | 1,089,862 KB | 1,064.32 MB | 1.04 GB |
| **p90** | 1,527,138 KB | 1,491.35 MB | 1.46 GB |
| **p95** | 2,010,461 KB | 1,963.34 MB | 1.92 GB |
| **p99** | 2,408,853 KB | 2,352.40 MB | 2.30 GB |
| **Maximum** | 3,021,872 KB | 2,951.05 MB | 2.88 GB |

### Key Observation on Small Circuits in the CSV
- **Row 0**: `add=1, sub=0, multiply=2, Depth=2, MultDepth=1` -> Peak RAM = **819,824 KB (~800.6 MB)**
- **Row 6**: `add=2, sub=1, multiply=1, Depth=3, MultDepth=1` -> Peak RAM = **823,644 KB (~804.3 MB)**
- **Row 2**: `add=13, sub=10, multiply=9, Depth=6, MultDepth=2` -> Peak RAM = **1,017,256 KB (~993.4 MB)**

Notice that even an almost-empty expression (1 addition, 2 multiplications) records ~800 MB, and the increase across operations is an additive delta above this ~780-800 MB baseline.

---

## 5. The Core Dilemma: Mapping and Scale Discrepancy

### Dilemma 1: Lack of 1-to-1 Mapping to Expressions
`memory_estimator_stats_dataset.csv` contains numerical summaries of operations, but does **not** contain the expression strings or line indices corresponding to `final_llm_dataset.txt` (which has 44,656 expressions).
- Line 0 of `final_llm_dataset.txt` has 16 additions and 32 multiplications.
- Row 0 of `memory_estimator_stats_dataset.csv` has 1 addition and 2 multiplications.
Because the two files are not aligned row-by-row, we cannot take an expression from `final_llm_dataset.txt`, evaluate it with `ckks_memory_estimator.py`, and compare it directly to a known ground-truth row in `memory_estimator_stats_dataset.csv`.

### Dilemma 2: Numerical Scale Difference
When evaluating expressions using `ckks_memory_estimator.py`:
- For an expression with `add=1, multiply=2`:
  - `ckks_memory_estimator.py (CPU)`: **32.0 MB**
  - `ckks_memory_estimator.py (GPU)`: **20.2 MB**
  - `memory_estimator_stats_dataset.csv (Row 0)`: **800.6 MB**
- Distribution on a sample of 500 expressions from `final_llm_dataset.txt`:
  - Min: 17.7 MB
  - p25: 41.1 MB
  - p50 (Median): 67.2 MB
  - p75: 140.7 MB
  - p95: 239.7 MB
  - Max: 1,721.1 MB

### Two Interpretations of the Discrepancy

#### Interpretation A: Net FHE Working Set vs. Gross Operating System RSS
- **The CSV measured Gross Process RSS**: In `src/fheco/code_gen/gen_func_lattigo.cpp`, memory was sampled via `/proc/self/statm` (Linux process Resident Set Size). A Go runtime process running Lattigo with large polynomial ring parameters allocates an initial runtime heap and OS page buffer of ~750 to 800 MB before circuit execution begins.
- **The Analytical Estimator measures Net FHE Memory**: Mounir's probes (`gpu_mem_probe.cu` via NVIDIA RMM pool, and `cpu_mem_probe` via forced `runtime.GC()`) intentionally isolate the memory of cryptographic objects (keys, ciphertexts, evaluator scratchpad) and exclude container/runtime OS allocations.
- Under this view:
  `Gross Measured RSS ≈ 770 MB (OS/Runtime Base) + Net Analytical FHE Memory`.

#### Interpretation B: Analytical Estimator Underestimation
- The analytical model may be underestimating the actual physical footprint experienced by a compiler or runtime executor because it does not model memory fragmentation, temporary buffers during key switching or rescaling, input/output encoding overhead, or memory allocator headroom.
- If physical execution on a GPU or CPU requires 800+ MB for even small circuits, treating peak memory as 30-70 MB may provide unrealistic signals to the RL optimizer.

---

## 6. The Budget Selection Problem for RL Training

In Lagrangian Constrained RL:
```
Constraint Violation Delta = Estimated_Memory - Budget
Penalty = lambda * max(0, Delta)
```

The selection of `--budgets` directly dictates what the agent learns:

### Scenario 1: Budgets based on the CSV scale (e.g., 900 MB, 1.1 GB, 2 GB)
- If the agent uses `ckks_memory_estimator.py` (which outputs 20 MB to 200 MB), then `Estimated_Memory` will be strictly less than 900 MB for virtually all expressions.
- Result: Constraint violation $\Delta$ is always $\le 0$, $\lambda$ stays at 0, and the agent **never receives any penalty**, learning nothing about memory constraints.

### Scenario 2: Budgets based on the Analytical Estimator's distribution (e.g., 50 MB, 70 MB, 150 MB)
- In this regime, ~50% of raw expressions violate the 70 MB budget, forcing the agent to discover rewrite rules (such as vectorization, associativity, or reuse) that reduce live ciphertexts and rotation steps.
- Concern: The budget values (50 MB, 70 MB) look synthetic and decoupled from the ~800 MB - 3 GB values observed in process memory measurements.

### Scenario 3: Calibrated Offset Formulation
- An explicit baseline offset (e.g., `+770 MB`) is added to the analytical estimator:
  `Effective_Peak_RAM = Base_Offset + Analytical_Estimator(expr)`.
- Under this formulation, the estimator outputs ~800 MB to ~1,200 MB, making the 900 MB, 1.1 GB, and 2 GB budgets directly applicable.

---

## 7. Proposed Evaluation & Debugging Metrics for Memory

In the legacy implementation (`RL/fhe_rl/test.py`), the summary tables reported:
- `Initial Noise`
- `Final Noise`
- `Noise Margin (Budget - Final Noise)`

### Why "Initial vs Final" Does Not Apply to Peak RAM
Noise was modeled as a resource that decreases/accumulates as multiplications and rescalings occur along the execution depth of a single evaluation. 

In contrast, **Peak RAM is a global static property of the compiled program graph**. During RL optimization, the agent applies rewrite rules (e.g., vectorization, associativity, factorizations) to transform an unoptimized expression into an optimized expression.
- Vectorization reduces total operations, but introduces **Galois rotation keys** (each key requiring several megabytes).
- Re-association or common subexpression elimination (CSE) can either increase or decrease the **peak count of live ciphertexts**.

Therefore, the evaluation must capture whether the optimization caused a **memory explosion** or **remained within hardware headroom**, alongside detailed component breakdowns.

### 1. Main Evaluation Table Metrics
For benchmark and held-out evaluation reports:

1. **Unoptimized Peak RAM (MB)**:
   The peak memory of the initial, raw input expression.
2. **Optimized Peak RAM (MB)**:
   The peak memory of the final rewritten expression selected by the RL agent.
3. **RAM Expansion / Overhead (%)**:
   `RAM_Expansion = ((Optimized_RAM - Unoptimized_RAM) / Unoptimized_RAM) * 100`
   - Indicates whether the rewrites bloated memory (positive %) or compressed memory (negative %).
4. **Target Budget (MB)**:
   The active hardware constraint (e.g., 70 MB or 1100 MB).
5. **Headroom / Slack (MB)**:
   `Headroom = Budget - Optimized_RAM`
   - Positive (+): The circuit fits on the target hardware with remaining memory headroom.
   - Negative (-): Out of Memory (OOM) violation.
6. **Cost Reduction (%)**:
   `Cost_Reduction = ((Initial_Cost - Optimized_Cost) / Initial_Cost) * 100`
   - The primary optimization reward (latency / operation cost reduction).
7. **Feasibility Status**:
   `FEASIBLE` (if `Optimized_RAM <= Budget`) or `VIOLATED (OOM)`.

### 2. Diagnostic & Component Breakdown Metrics (Per Expression)
To facilitate root-cause analysis when an expression violates budget or when analyzing Pareto trade-offs, `estimate_expression_memory(expr, backend)` provides:

1. **Galois Rotation Keys Memory (MB)**:
   Memory occupied by rotation keys: `G * SWK` (on GPU: `(G + 1) * SWK`).
2. **Distinct Rotation Steps (`G`)**:
   Count of distinct non-zero rotation shifts introduced by vectorization rewrites.
3. **Live Ciphertexts Memory (MB)**:
   Working memory occupied by intermediate ciphertexts at the peak liveness point.
4. **Peak Live Ciphertexts Count**:
   The maximum number of simultaneous ciphertexts held in memory during the post-order evaluation DAG.
5. **Context & Scratch Memory (MB)**:
   Evaluator baseline overhead: public/secret keys + context structures + key-switching scratchpad.
6. **Selected Parameters (`N`, `L`, `K`)**:
   Polynomial ring dimension $N$, level modulus chain length $L$, and special modulus count $K$.

---

## 8. Detailed Step-by-Step Integration Guide (Implementation Blueprint)

For future reference and execution, here is the exact, file-by-file blueprint to integrate the peak memory estimator into Imed's unified RL codebase:

### Step 1: Export in PyTRS (`RL/pytrs/__init__.py`)
Ensure both estimators are exported side-by-side:
```python
from noise_estimator import NoiseEstimator
from ckks_memory_estimator import (
    CKKSMemoryEstimator,
    estimate_expression_memory,
    estimate_peak_memory,
    chehab_params,
)
```

### Step 2: Auto-Detection Helper
In `RL/fhe_rl/env.py` (or a utility module), auto-detect hardware availability:
```python
import torch

def get_default_backend() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"
```

### Step 3: Dual-Metric Environment (`RL/fhe_rl/env.py`)
Modify `fheEnv.__init__` to accept the constraint configuration:
```python
def __init__(
    self,
    rules_list,
    expressions,
    max_positions=2,
    embeddings_model=None,
    budget_options=None,
    constraint_method="lagrangian_od_ov",
    constraint_metric="memory",  # "memory" or "noise"
    backend="auto",             # "auto", "gpu", or "cpu"
    verbose=True,
):
    super().__init__()
    self.constraint_metric = constraint_metric
    self.backend = get_default_backend() if backend == "auto" else backend
    
    # Initialize both estimators; select active one based on metric
    self.noise_estimator = NoiseEstimator()
    self.memory_estimator = CKKSMemoryEstimator(backend=self.backend)

    # Set default budget options and unconstrained threshold
    if budget_options is not None:
        self.budget_options = budget_options
    else:
        if self.constraint_metric == "memory":
            self.budget_options = [50, 70, 150, 100_000]
        else:
            self.budget_options = [240, 300, 1_000_000]
            
    self.unconstrained_threshold = (
        10_000 if self.constraint_metric == "memory" else 100_000
    )
    ...
```

In `env.reset()` and `env.step()`:
Compute both values to ensure downstream compatibility, and assign `constraint_value`:
```python
mem_val = float(self.memory_estimator.estimate(self.expression))
noise_val = float(self.noise_estimator.estimate(self.expression))
current_val = mem_val if self.constraint_metric == "memory" else noise_val

info = {
    "expression": self.expression,
    "budget": self.budget,
    "constraint_value": current_val,
    "memory": mem_val,
    "noise": noise_val,
    "cost": self.current_cost,
}
```

Update terminal penalties (`margin_barrier` and `nato_sc`):
```python
if current_val > self.budget:
    # apply penalty based on constraint_value
```

In `env.get_action_mask()`:
```python
use_mask = (
    self.constraint_method in ("noise_masking", "memory_masking")
    and self.budget is not None
    and self.budget < self.unconstrained_threshold
)
if use_mask:
    for pos_idx in range(valid_positions):
        k, _ = matches[pos_idx]
        try:
            new_tree = rule_obj.apply_rule(parsed, path=k)
            est = (
                self.memory_estimator.estimate(new_tree)
                if self.constraint_metric == "memory"
                else self.noise_estimator.estimate(new_tree)
            )
            if est <= self.budget:
                mask[start + pos_idx] = 1.0
        except Exception:
            pass
```

### Step 4: Lagrangian Wrappers (`RL/fhe_rl/wrappers.py`)
Ensure all Lagrangian wrapper classes calculate `delta` generically:
```python
def step_wait(self):
    obs, rewards, dones, infos = self.venv.step_wait()
    for env_index, info in enumerate(infos):
        val = info.get("constraint_value", info.get("memory", info.get("noise", 0.0)))
        delta = val - info["budget"]
        ON_DONE, ON_VIOLATION = dones[env_index], delta > 0
        if ON_DONE and ON_VIOLATION:
            rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
    return obs, rewards, dones, infos
```

In `BaseLagrangianWrapper.update_lambda_penalty`:
```python
def update_lambda_penalty(self, constraint_val, budget):
    if constraint_val > budget:
        self.lambda_penalty += 0.01 * (constraint_val - budget)
    else:
        self.lambda_penalty = max(0, self.lambda_penalty - 0.01)
```

### Step 5: Training Loop (`RL/fhe_rl/train.py`)
In the validation evaluation loop of `_train_lagrangian_loop`:
```python
total_constraint = 0
for _ in range(num_benchmarks):
    obs = val_env.reset()
    done, ep_val = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = val_env.step(action)
        ep_val = info[0].get("constraint_value", info[0].get("noise", 0.0))
    total_constraint += ep_val

    val_env.update_lambda_penalty(
        constraint_val=ep_val,
        budget=val_env.unwrapped.reset_infos[0]["budget"],
    )
    env.update_lambda_penalty(
        constraint_val=ep_val,
        budget=val_env.unwrapped.reset_infos[0]["budget"],
    )

avg_val = total_constraint / num_benchmarks
metric_name = "peak_memory_mb" if getattr(env, "constraint_metric", "memory") == "memory" else "noise"
tensorboard_writer.add_scalar(f"Lagrange/avg_{metric_name}", avg_val, model.num_timesteps)
tensorboard_writer.add_scalar("Lagrange/lambda_penalty", val_env.lambda_penalty, model.num_timesteps)
```

### Step 6: Evaluation Script (`RL/fhe_rl/test.py`)
Format the evaluation tables using Section 7's 7 metrics when `constraint_metric == "memory"`, while retaining the legacy format when `constraint_metric == "noise"`.
Ensure the rollback checkpoint selector checks:
```python
valid_checkpoints = [cp for cp in checkpoints if cp["constraint_value"] <= budget]
```

### Step 7: CLI Entry Point (`RL/fhe_rl/__main__.py`)
Add the flags to argument parsers:
```python
parser.add_argument(
    '--constraint_metric',
    type=str,
    default='memory',
    choices=['memory', 'noise'],
    help='Constraint target to enforce: "memory" (default, CKKS peak RAM in MB) or "noise" (legacy)'
)
parser.add_argument(
    '--backend',
    type=str,
    default='auto',
    choices=['auto', 'gpu', 'cpu'],
    help='Backend for memory estimation (default: auto)'
)
```
Set dynamic default budgets if `--budgets` is omitted:
- If `--constraint_metric memory`: default `"50,70,150,100000"`
- If `--constraint_metric noise`: default `"240,300,1000000"`

### Step 8: Smoke Test Verification Protocol
Run these commands to confirm full end-to-end integration:
1. **Unit Test**: Test estimator with simple expressions (`test_ckks_memory_estimator.py`).
2. **Environment Reset & Step**: Confirm `fheEnv` steps without NaNs or crashes.
3. **Short 1,000-Step Training Run**:
   ```bash
   python -m fhe_rl train --constraint_metric memory --backend auto --budgets "50,70,100000" --timesteps 1000 --num_envs 2
   ```

---

## 9. Specific Questions for Review

1. **Estimator Interpretation**: Should the RL agent be constrained on the **Net FHE Cryptographic Footprint** (20–200 MB as modeled by Mounir's closed-form DAG estimator) or on **Gross Process Resident Set Size** (800 MB–3 GB as measured by process RSS in `memory_estimator_stats_dataset.csv`)?
2. **Budget Calibration**:
   - If using Net FHE footprint: Are percentile-based tiers like `[50, 70, 150, 100000]` MB appropriate?
   - If using Gross Process RSS: Should a constant baseline (e.g. ~770 MB) be formally added to the analytical estimator to match the CSV's scale?
3. **Validation Strategy**: Given that `memory_estimator_stats_dataset.csv` does not record the input expression strings, what is the recommended protocol to validate the closed-form estimator against ground-truth execution?
4. **Metrics Alignment**: Do the proposed memory metrics (Unoptimized vs. Optimized RAM, Expansion %, Headroom, and component breakdown of keys vs. live ciphertexts) sufficiently capture what is needed to benchmark memory-constrained circuit rewriting?
