from stable_baselines3.common.vec_env import VecEnvWrapper


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
