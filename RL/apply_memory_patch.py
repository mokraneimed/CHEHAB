#!/usr/bin/env python3
"""
Swap the noise constraint for the peak-RAM constraint in the CHEHAB RL code.

Run from the RL/ directory (the one that contains fhe_rl/):
    python3 apply_memory_patch.py
Then review with:  git --no-pager diff --stat ; git --no-pager diff fhe_rl/env.py

What it does (and why):
  * memory.py   : adds class MemoryEstimator(.estimate(expr_or_tree) -> peak RAM in MB, margin OFF),
                  a drop-in replacement of NoiseEstimator, so the rest of the code keeps its structure.
  * env.py      : uses MemoryEstimator; budgets default to 800/1500/2600 MB (+1_000_000 = unconstrained);
                  info["noise"] becomes info["constraint_value"]; skips expressions the estimator cannot
                  handle; keeps the last value if a rewritten expression fails to parse.
  * wrappers.py : penalty and lambda update use the RELATIVE violation in percent,
                  100*(value-budget)/budget, so lambda has the same scale whatever the budget (MB).
  * lagrangian_pid.py, focops.py : read info["constraint_value"] (same relative scale).
  * train.py    : validation loop reads constraint_value, logs avg peak RAM (MB) and violation rate.
  * test.py     : uses MemoryEstimator, terminal["constraint_value"]; report columns renamed (RAM, MB).
  * __main__.py / run.py : default test budget 1500 MB, help texts.
Each exact-text edit aborts with a message if the file is not what was expected (nothing is silently skipped).
"""
import re, sys, pathlib

ROOT = pathlib.Path("fhe_rl")
if not ROOT.is_dir():
    sys.exit("Run this from the RL/ directory (fhe_rl/ not found).")

def load(name):
    return (ROOT / name).read_text()

def save(name, s):
    (ROOT / name).write_text(s)
    print("patched", name)

def exact(s, old, new, count, name):
    n = s.count(old)
    if n != count:
        sys.exit(f"{name}: expected {count} occurrence(s) of {old!r}, found {n}. File differs from what the patch expects.")
    return s.replace(old, new)

# ───────────────────────────── memory.py ─────────────────────────────
s = load("memory.py")
if "class MemoryEstimator" not in s:
    s += '''

class MemoryEstimator:
    """Drop-in replacement of NoiseEstimator: .estimate(expr_or_tree) -> peak RAM in MB.
    Training uses margin=False (central estimate); keep the margin for deployment checks only."""

    def __init__(self, margin=False):
        self.margin = margin

    def estimate(self, expr):
        if not isinstance(expr, str):
            from serializer import expr_to_str      # pytrs is on sys.path (see top of this file)
            expr = expr_to_str(expr)
        return peak_ram_mb(expr, margin=self.margin)
'''
    save("memory.py", s)
else:
    print("memory.py already has MemoryEstimator")

# ───────────────────────────── env.py ────────────────────────────────
n = "env.py"; s = load(n)
s = exact(s, "from pytrs import parse_sexpr, calculate_cost, NoiseEstimator, expr_to_str\n",
          "from pytrs import parse_sexpr, calculate_cost, expr_to_str\nfrom .memory import MemoryEstimator\n", 1, n)
s = exact(s, "DEFAULT_BUDGET_OPTIONS = [240, 300, 1_000_000]",
          "DEFAULT_BUDGET_OPTIONS = [800, 1500, 2600, 1_000_000]   # peak-RAM budgets in MB (1_000_000 = unconstrained)", 1, n)
s = exact(s, "self.noise_estimator = NoiseEstimator()",
          "self.noise_estimator = MemoryEstimator()   # name kept for compatibility: it now returns peak RAM in MB", 1, n)
# reset: skip expressions the estimator cannot handle
s = exact(s, "        noise = self.noise_estimator.estimate(self.expression)\n",
          "        try:\n            noise = float(self.noise_estimator.estimate(self.expression))\n"
          "        except Exception:      # estimator cannot handle this expression: take the next one\n"
          "            return self.reset(seed=seed, options=options)\n"
          "        self._last_value = noise\n", 1, n)
# info dicts
s = exact(s, '            "noise": self.noise_estimator.estimate(self.expression),\n',
          '            "constraint_value": self._estimate(self.expression),\n', 1, n)
s = exact(s, '            "noise": noise,\n', '            "constraint_value": noise,\n', 1, n)
s = exact(s, "Noise         {RESET}", "Peak RAM (MB){RESET}", 1, n)
# safe estimator helper, inserted before _valid_end_action
s = exact(s, "    def _valid_end_action(self,expr: str) -> bool:",
          "    def _estimate(self, expr):\n"
          '        """Peak RAM (MB) of expr; if the estimator fails on a rewritten expression, keep the previous value."""\n'
          "        try:\n            v = float(self.noise_estimator.estimate(expr))\n            self._last_value = v\n            return v\n"
          "        except Exception:\n            return getattr(self, \"_last_value\", 0.0)\n\n"
          "    def _valid_end_action(self,expr: str) -> bool:", 1, n)
s = s.replace('info["noise"]', 'info["constraint_value"]').replace("info['noise']", "info['constraint_value']")
s = re.sub(r"\bnoise\b", "memory_mb", s)          # local variable name only (keys like noise_ratio are untouched)
save(n, s)

# ───────────────────────────── wrappers.py ───────────────────────────
n = "wrappers.py"; s = load(n)
for cls in ("BaseLagrangianWrapper", "LagrangianVecEnvWrapper", "LagrangianPerStepWrapper", "LagrangianAlwaysDoneWrapper"):
    if f"class {cls}" not in s:
        sys.exit(f"{n}: class {cls} not found")
WRAPPERS = '''from stable_baselines3.common.vec_env import VecEnvWrapper


def violation_pct(value, budget):
    """Relative constraint violation in percent: 100 * (value - budget) / budget.
    Negative when the budget is respected. Using a relative measure gives lambda the same
    scale for every budget (800 MB or 2600 MB), instead of raw MB."""
    return 100.0 * (value - budget) / max(budget, 1)


class BaseLagrangianWrapper(VecEnvWrapper):
    """Base class for Lagrangian wrappers with shared lambda update logic.

    The constrained quantity is info["constraint_value"] (estimated peak RAM in MB).
      - If it exceeds the budget: increase lambda proportionally to the relative excess (percent)
      - Otherwise: slowly decrease lambda toward 0
    The subclasses differ only in WHEN the penalty is applied (step_wait).
    """

    def __init__(self, venv, lambda_penalty=0.1):
        super().__init__(venv)
        self.lambda_penalty = lambda_penalty

    def update_lambda_penalty(self, constraint_value, budget):
        delta = violation_pct(constraint_value, budget)
        if delta > 0:
            self.lambda_penalty += 0.01 * delta
        else:
            self.lambda_penalty = max(0, self.lambda_penalty - 0.01)

    def reset(self):
        return self.venv.reset()


class LagrangianVecEnvWrapper(BaseLagrangianWrapper):
    """ON_DONE + ON_VIOLATION: penalty only at episode end AND only when the budget is violated."""

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            delta = violation_pct(info["constraint_value"], info["budget"])
            ON_DONE, ON_VIOLATION = dones[env_index], delta > 0
            if ON_DONE and ON_VIOLATION:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos


class LagrangianPerStepWrapper(BaseLagrangianWrapper):
    """Per-step violation penalty: penalize EVERY step where the current expression exceeds the budget."""

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            delta = violation_pct(info["constraint_value"], info["budget"])
            if delta > 0:
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos


class LagrangianAlwaysDoneWrapper(BaseLagrangianWrapper):
    """Always penalize at terminal: lambda * relative violation (a bonus when under budget)."""

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for env_index, info in enumerate(infos):
            if dones[env_index]:
                delta = violation_pct(info["constraint_value"], info["budget"])
                rewards[env_index] = rewards[env_index] - (self.lambda_penalty * delta)
        return obs, rewards, dones, infos
'''
save(n, WRAPPERS)

# ───────────────────────────── algos ─────────────────────────────────
n = "algos/lagrangian_pid.py"; s = load(n)
s = exact(s, "    def update_lambda_penalty(self, noise, budget):", "    def update_lambda_penalty(self, constraint_value, budget):", 1, n)
s = exact(s, "        error = (noise - budget) / max(budget, 1)", "        error = (constraint_value - budget) / max(budget, 1)", 1, n)
s = exact(s, '            noise = info.get("noise", 0.0)\n            budget = info.get("budget", 1)\n            delta = noise - budget\n',
          '            value = info.get("constraint_value", 0.0)\n            budget = info.get("budget", 1)\n            delta = 100.0 * (value - budget) / max(budget, 1)   # relative violation, percent\n', 1, n)
s = exact(s, "            self.update_lambda_penalty(noise, budget)", "            self.update_lambda_penalty(value, budget)", 1, n)
save(n, s)

n = "algos/focops.py"; s = load(n)
s = exact(s, 'noise = info.get("noise", 0.0)', 'noise = info.get("constraint_value", 0.0)', 1, n)
save(n, s)

# ───────────────────────────── train.py ──────────────────────────────
n = "train.py"; s = load(n)
s = exact(s, "        total_noise = 0\n", "        total_noise = 0\n        n_viol = 0\n", 1, n)
s = exact(s, 'ep_noise = info[0].get("noise", 0.0)', 'ep_noise = info[0].get("constraint_value", 0.0)', 1, n)
s = exact(s, "            total_noise += ep_noise\n",
          "            total_noise += ep_noise\n            n_viol += int(ep_noise > val_env.unwrapped.reset_infos[0][\"budget\"])\n", 1, n)
s = exact(s, "                noise=ep_noise,", "                constraint_value=ep_noise,", 2, n)
s = exact(s, "        avg_noise = total_noise / num_benchmarks\n", "        avg_noise = total_noise / num_benchmarks\n        viol_rate = n_viol / num_benchmarks\n", 1, n)
s = exact(s, 'print(f"[Step {model.num_timesteps}] Avg noise: {avg_noise:.2f}, "',
          'print(f"[Step {model.num_timesteps}] Avg peak RAM: {avg_noise:.1f} MB, violation rate: {viol_rate:.2f}, "', 1, n)
s = exact(s, '"Lagrange/avg_noise", avg_noise, model.num_timesteps,\n        )\n',
          '"Lagrange/avg_peak_ram_mb", avg_noise, model.num_timesteps,\n        )\n'
          '        tensorboard_writer.add_scalar(\n            "Lagrange/violation_rate", viol_rate, model.num_timesteps,\n        )\n', 1, n)
save(n, s)

# ───────────────────────────── test.py ───────────────────────────────
n = "test.py"; s = load(n)
s = exact(s, "from pytrs import parse_sexpr, NoiseEstimator\n", "from pytrs import parse_sexpr\nfrom .memory import MemoryEstimator\n", 1, n)
s = exact(s, "NoiseEstimator()", "MemoryEstimator()", 2, n)
s = exact(s, 'terminal["noise"]', 'terminal["constraint_value"]', 2, n)
s = exact(s, "noise_budget: int = 300", "noise_budget: int = 1500", 2, n)
for old, new in [("Noise Margin", "RAM Headroom (MB)"), ("Final Noise", "Final RAM (MB)"), ("Initial Noise", "Initial RAM (MB)"),
                 ("Avg Noise Increase", "Avg RAM Increase (MB)"), ("Trajectory Noises", "Trajectory RAM (MB)"),
                 ("'Avg Noise'", "'Avg RAM (MB)'"), (", noise {initial_noise", ", RAM {initial_noise"),
                 ("Avg Noise", "Avg RAM (MB)")]:
    s = s.replace(old, new)
save(n, s)
left = [i + 1 for i, l in enumerate(s.splitlines()) if re.search(r"[Nn]oise", l) and "noise_estimator" not in l]
print("  test.py lines still mentioning 'noise' (internal names, fine unless user-facing):", left[:40])

# ───────────────────────────── CLI / run ─────────────────────────────
n = "__main__.py"; s = load(n)
s = s.replace('Comma-separated list of noise budgets (e.g., "300,400,500,1000,9000000")', 'Comma-separated list of peak-RAM budgets in MB (e.g., "800,1500,2600,1000000")')
s = s.replace('(e.g., "240,300,1000000")', '(e.g., "800,1500,2600,1000000")').replace("noise_budget=300", "noise_budget=1500")
save(n, s)
n = "run.py"; s = load(n)
s = s.replace("noise_budget: int = 300", "noise_budget: int = 1500")
save(n, s)
print("\nDone. Now: python3 -m py_compile fhe_rl/*.py fhe_rl/algos/*.py   and   git --no-pager diff --stat")