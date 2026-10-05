import json
import statistics
from datetime import datetime

EXAMPLE_1_CONTEXT = {
    "title": "Example 1 — Joint vs Separated Optimization",
    "description": (
        "Consider an FHE circuit that initially has operations cost C_ops and "
        "uses K unique rotation keys.  In joint optimization (MORL), a single "
        "RL agent simultaneously minimizes both the operations cost and the "
        "number of rotation keys by applying rewrite rules that may increase "
        "one objective slightly to achieve a large reduction in the other.  "
        "In the separated approach, Phase 1 minimizes operations cost alone "
        "(ignoring keys), and Phase 2 then tries to reduce rotation keys by "
        "inserting rotation-merging rewrites on the already-optimized circuit.  "
        "Because Phase 2 operates on a locally-optimal circuit, each added "
        "rotation to merge keys comes at a fixed cost of 51 (rotation_cost=50 "
        "+ depth_increment=1).  Joint optimization can discover rewrites that "
        "are globally better because it sees both objectives at once."
    ),
}

SEPARATED_APPROACH_COST_MODEL = {
    "rotation_cost": 50,
    "depth_cost_per_rotation": 1,
    "total_cost_per_added_rotation": 51,
    "explanation": (
        "In the separated approach Phase 2, each additional rotation inserted "
        "to merge/reduce rotation keys costs exactly 51 units: 50 for the "
        "rotation operation itself plus 1 for the added depth (depth estimation)."
    ),
}

def compute_separated_approach_costs(ops_cost_phase1, n_rotations_to_add):
    phase2_overhead = n_rotations_to_add * SEPARATED_APPROACH_COST_MODEL["total_cost_per_added_rotation"]
    return {
        "phase1_ops_cost": ops_cost_phase1,
        "phase2_rotations_added": n_rotations_to_add,
        "phase2_overhead": phase2_overhead,
        "total_separated_cost": ops_cost_phase1 + phase2_overhead,
    }

class LLMLogger:
    def __init__(self, output_csv, optimization_method):
        self.output_log_json = output_csv.replace(".csv", "_logs.json")
        self.optimization_method = optimization_method
        self.benchmark_logs = []

    def log_benchmark_run(
        self,
        bench_name,
        benchmark_type,
        w_ops,
        w_keys,
        operation_stats,
        operations,
        slot_count=None,
        tree_depth=None,
        instance=None,
        regime=None
    ):
        log_ops_cost = statistics.median(operation_stats["final_ops_cost"]) if operation_stats["final_ops_cost"] else None
        log_keys_cost = statistics.median(operation_stats["final_keys_cost"]) if operation_stats["final_keys_cost"] else None
        log_depth = statistics.median(operation_stats["Depth"]) if operation_stats["Depth"] else None
        log_mul_depth = statistics.median(operation_stats["Multiplicative Depth"]) if operation_stats["Multiplicative Depth"] else None

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "benchmark": bench_name,
            "benchmark_type": benchmark_type,
            "preference_vector": {"w_ops": w_ops, "w_keys": w_keys},
            "costs": {
                "final_ops_cost": log_ops_cost,
                "final_keys_cost": log_keys_cost,
                "depth": log_depth,
                "multiplicative_depth": log_mul_depth,
            },
            "operation_counts": {
                op: statistics.median(operation_stats[op]) if operation_stats[op] else 0
                for op in operations
            },
            "morl_joint_optimization": {
                "approach": "joint",
                "description": "Single RL agent optimizes ops and keys simultaneously",
                "total_weighted_cost": (
                    w_ops * log_ops_cost + w_keys * log_keys_cost
                    if log_ops_cost is not None and log_keys_cost is not None
                    else None
                ),
            },
        }

        if slot_count is not None:
            log_entry["slot_count"] = slot_count
        if tree_depth is not None:
            log_entry["tree_depth"] = tree_depth
        if instance is not None:
            log_entry["instance"] = instance
        if regime is not None:
            log_entry["regime"] = regime

        if log_ops_cost is not None and log_keys_cost is not None:
            separated_estimate = compute_separated_approach_costs(
                ops_cost_phase1=log_ops_cost,
                n_rotations_to_add=int(log_keys_cost) if log_keys_cost else 0,
            )
            log_entry["separated_approach_estimate"] = separated_estimate
            log_entry["separated_approach_cost_model"] = SEPARATED_APPROACH_COST_MODEL

            joint_total = log_ops_cost + log_keys_cost
            sep_total = separated_estimate["total_separated_cost"]
            joint_is_better = joint_total < sep_total

            # Generate a structured explanation if joint optimization is better
            explanation = None
            if joint_is_better:
                explanation = (
                    f"Example Found: {bench_name} (w_ops={w_ops}, w_keys={w_keys})\n"
                    f"In this configuration, Joint Optimization found a rewrite sequence with:\n"
                    f"- Operations cost = {log_ops_cost}\n"
                    f"- Unique keys = {log_keys_cost}\n"
                    f"Total Joint Cost = {joint_total}.\n\n"
                    f"If we used the Separated Approach:\n"
                    f"- Phase 1 would give an operations cost of at best {log_ops_cost}.\n"
                    f"- Phase 2 would need to reduce keys to {log_keys_cost}, adding {log_keys_cost} rotations.\n"
                    f"- Each added rotation costs 51 (50 for rotation + 1 for depth).\n"
                    f"- Phase 2 overhead = {log_keys_cost} * 51 = {separated_estimate['phase2_overhead']}.\n"
                    f"Total Separated Cost = {log_ops_cost} + {separated_estimate['phase2_overhead']} = {sep_total}.\n\n"
                    f"Conclusion: Joint optimization is better by {sep_total - joint_total} units "
                    f"because it traded a smaller increase in operations cost to avoid the fixed 51-unit penalty "
                    f"of merging keys post-optimization."
                )

            log_entry["joint_vs_separated"] = {
                "joint_total_cost": joint_total,
                "separated_total_cost": sep_total,
                "joint_advantage": sep_total - joint_total,
                "joint_advantage_pct": (
                    round((sep_total - joint_total) / sep_total * 100, 2)
                    if sep_total > 0 else 0.0
                ),
                "joint_is_better": joint_is_better,
                "explanation": explanation
            }

        self.benchmark_logs.append(log_entry)
        print(f"  [LOG] Saved log entry for {bench_name} w_ops={w_ops}")

    def save_logs(self):
        full_log = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "script": "run_benchmarks.py",
                "backend": "SEAL",
                "optimization_method": "RL" if self.optimization_method == 1 else "EGraph",
                "purpose": (
                    "Structured logs for LLM analysis. The LLM should use these logs "
                    "together with Example 1 to identify benchmark configurations where "
                    "joint (MORL) optimization outperforms the separated approach. "
                    "Focus on costs, rewards, and preference vectors — not execution time."
                ),
            },
            "reference_example": EXAMPLE_1_CONTEXT,
            "separated_approach_cost_model": SEPARATED_APPROACH_COST_MODEL,
            "cost_model_explanation": {
                "ops_cost": (
                    "Operations cost: weighted sum of FHE operations. "
                    "Add/Sub/Neg cost 250 each, Multiply costs 100, "
                    "Rotation costs 50 per rotation, plus depth and multiplicative depth."
                ),
                "keys_cost": (
                    "Keys cost: number of unique rotation keys needed. "
                    "Each unique rotation offset requires a separate Galois key."
                ),
                "morl_reward": (
                    "MORL reward = w_ops * r_ops + w_keys * r_keys, where "
                    "r_ops = (old_ops - new_ops) / old_ops, "
                    "r_keys = (old_keys - new_keys) / n_budget. "
                    "Final reward is scaled by 100x."
                ),
                "separated_phase2_cost": (
                    "In the separated approach, Phase 2 adds rotations to reduce keys. "
                    "Each added rotation costs 51 = 50 (rotation operation) + 1 (depth increment). "
                    "The LLM agent should use this cost during reasoning to compare approaches."
                ),
            },
            "llm_reasoning_instructions": {
                "task": (
                    "Analyze these benchmark logs to find examples where joint optimization "
                    "achieves better total cost than the separated approach. For each benchmark "
                    "configuration, compare the joint total cost (ops_cost + keys_cost) against "
                    "the separated total cost (ops_cost + keys_to_reduce * 51). Highlight cases "
                    "where the joint advantage is significant."
                ),
                "key_insight": (
                    "Joint optimization can find rewrites that trade a small increase in ops "
                    "cost for a large reduction in keys, or vice versa. The separated approach "
                    "cannot do this because Phase 2 operates on an already-optimized circuit "
                    "and each rotation insertion has a fixed cost of 51."
                ),
                "rotation_cost_rule": (
                    "IMPORTANT: In the separated approach Phase 2, each added rotation to "
                    "merge/reduce rotation keys costs exactly 51 (rotation_cost=50 + "
                    "depth_cost=1). Use this when reasoning about whether the separated "
                    "approach can match the joint optimization result."
                ),
            },
            "benchmark_runs": self.benchmark_logs,
            "summary": {
                "total_runs": len(self.benchmark_logs),
                "joint_better_count": sum(
                    1 for log in self.benchmark_logs
                    if log.get("joint_vs_separated", {}).get("joint_is_better", False)
                ),
                "joint_worse_count": sum(
                    1 for log in self.benchmark_logs
                    if not log.get("joint_vs_separated", {}).get("joint_is_better", True)
                ),
                "avg_joint_advantage_pct": (
                    round(
                        sum(
                            log.get("joint_vs_separated", {}).get("joint_advantage_pct", 0)
                            for log in self.benchmark_logs
                            if "joint_vs_separated" in log
                        ) / max(1, sum(1 for log in self.benchmark_logs if "joint_vs_separated" in log)),
                        2,
                    )
                    if self.benchmark_logs else 0.0
                ),
            },
        }

        with open(self.output_log_json, "w") as f:
            json.dump(full_log, f, indent=2, default=str)

        print(f"\n{'='*70}")
        print(f"  JSON logs written to: {self.output_log_json}")
        print(f"  Total benchmark runs logged: {len(self.benchmark_logs)}")
        print(f"  Joint better than separated: {full_log['summary']['joint_better_count']} / {full_log['summary']['total_runs']}")
        print(f"  Avg joint advantage: {full_log['summary']['avg_joint_advantage_pct']}%")
        print(f"{'='*70}")
