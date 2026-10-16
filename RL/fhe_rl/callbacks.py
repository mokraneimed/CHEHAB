from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import torch
import os


class EntCoefScheduler(BaseCallback):
    def __init__(self, schedule, verbose: int = 0):
        super().__init__(verbose)
        self.schedule = schedule

    def _on_training_start(self) -> None:
        p = self.model._current_progress_remaining
        self.model.ent_coef = float(self.schedule(p))

    def _on_rollout_end(self) -> None:
        p = self.model._current_progress_remaining
        self.model.ent_coef = float(self.schedule(p))

    def _on_step(self) -> bool:
        return True

class ParetoEvalCallback(BaseCallback):
    def __init__(self, eval_env, pref_list, 
                 best_model_save_path=None, 
                 log_path=None, 
                 eval_freq=512, 
                 n_eval_episodes=5, 
                 deterministic=True, 
                 verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.pref_list = pref_list
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        
        # Track best performance (Average across the Pareto Front)
        self.best_mean_reward = -np.inf

        # Ensure directories exist
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n{BLUE}{'='*95}{RESET}")
                print(f"{BOLD}Step {self.num_timesteps:,}: Starting Pareto & Safety Evaluation ({len(self.pref_list)} front points, {self.n_eval_episodes} benchmarks each){RESET}")
                print(f"{BLUE}{'='*95}{RESET}")

            GREEN = "\033[92m"
            RED = "\033[91m"
            YELLOW = "\033[93m"
            CYAN = "\033[96m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            all_means = []
            all_lengths = []
            all_cost_reds = []
            all_exec_reds = []
            all_keys_reds = []
            all_compliances = []
            all_violations = []
            all_margins = []

            point_summaries = []

            for p_idx, w in enumerate(self.pref_list):
                # Lock eval env to this specific preference vector
                self.eval_env.env_method("set_preference_vector", w)

                pt_rewards = []
                pt_lengths = []
                pt_cost_reds = []
                pt_exec_reds = []
                pt_keys_reds = []
                pt_compliances = []
                pt_violations = []
                pt_margins = []
                pt_constraints = []
                pt_budgets = []
                metric_name = "Peak RAM"
                metric_unit = "MB"

                for _ in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    done = False
                    ep_rew = 0.0
                    ep_len = 0
                    last_info = {}

                    while not done:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                        obs, reward, dones, infos = self.eval_env.step(action)
                        ep_rew += float(reward[0])
                        ep_len += 1
                        done = bool(dones[0])
                        last_info = infos[0]

                    pt_rewards.append(ep_rew)
                    pt_lengths.append(ep_len)
                    pt_cost_reds.append(last_info.get("cost_reduction", 0.0) * 100.0)
                    pt_exec_reds.append(last_info.get("ops_reduction", 0.0) * 100.0)
                    pt_keys_reds.append(last_info.get("keys_reduction", 0.0) * 100.0)
                    is_safe = last_info.get("is_compliant", True)
                    pt_compliances.append(1.0 if is_safe else 0.0)
                    pt_violations.append(last_info.get("violation", 0.0))
                    pt_margins.append(last_info.get("margin", 0.0))
                    pt_constraints.append(last_info.get("final_constraint", 0.0))
                    pt_budgets.append(last_info.get("budget", 0))
                    metric_name = last_info.get("constraint_name", metric_name)
                    metric_unit = last_info.get("constraint_unit", metric_unit)

                mean_r = float(np.mean(pt_rewards))
                mean_l = float(np.mean(pt_lengths))
                mean_cost_red = float(np.mean(pt_cost_reds))
                mean_exec_red = float(np.mean(pt_exec_reds))
                mean_keys_red = float(np.mean(pt_keys_reds))
                safe_rate = float(np.mean(pt_compliances)) * 100.0
                mean_viol = float(np.mean(pt_violations))
                mean_cons = float(np.mean(pt_constraints))
                mean_bgt = float(np.mean(pt_budgets))

                all_means.append(mean_r)
                all_lengths.append(mean_l)
                all_cost_reds.append(mean_cost_red)
                all_exec_reds.append(mean_exec_red)
                all_keys_reds.append(mean_keys_red)
                all_compliances.extend(pt_compliances)
                all_violations.extend(pt_violations)
                all_margins.extend(pt_margins)

                # Log per-point metrics to TensorBoard
                self.logger.record(f"eval/reward_w_{w[1]:.2f}", mean_r)
                self.logger.record(f"eval/safe_rate_w_{w[1]:.2f}", safe_rate)
                self.logger.record(f"eval/cost_red_pct_w_{w[1]:.2f}", mean_cost_red)
                self.logger.record(f"eval/exec_red_pct_w_{w[1]:.2f}", mean_exec_red)
                self.logger.record(f"eval/keys_red_pct_w_{w[1]:.2f}", mean_keys_red)

                point_summaries.append({
                    "p_idx": p_idx + 1,
                    "w": w,
                    "mean_r": mean_r,
                    "cost_red": mean_cost_red,
                    "exec_red": mean_exec_red,
                    "keys_red": mean_keys_red,
                    "constraint": mean_cons,
                    "budget": mean_bgt,
                    "safe_rate": safe_rate,
                    "violations": int(len(pt_compliances) - sum(pt_compliances)),
                    "metric_name": metric_name,
                    "metric_unit": metric_unit,
                })

            self.eval_env.env_method("unlock_preferences")

            # Global Front Scores
            current_mean_reward = float(np.mean(all_means))
            current_mean_length = float(np.mean(all_lengths))
            current_safe_rate = float(np.mean(all_compliances)) * 100.0
            current_mean_cost_red = float(np.mean(all_cost_reds))
            current_mean_exec_red = float(np.mean(all_exec_reds))
            current_mean_keys_red = float(np.mean(all_keys_reds))
            current_mean_violation = float(np.mean(all_violations))
            current_mean_margin = float(np.mean(all_margins))
            total_violations = int(len(all_compliances) - sum(all_compliances))

            # Log to TensorBoard
            self.logger.record("eval/pareto_avg_reward", current_mean_reward)
            self.logger.record("eval/pareto_avg_ep_length", current_mean_length)
            self.logger.record("eval/pareto_safe_compliance_rate", current_safe_rate)
            self.logger.record("eval/pareto_avg_cost_red_pct", current_mean_cost_red)
            self.logger.record("eval/pareto_avg_exec_red_pct", current_mean_exec_red)
            self.logger.record("eval/pareto_avg_keys_red_pct", current_mean_keys_red)
            self.logger.record("eval/pareto_avg_violation", current_mean_violation)
            self.logger.record("eval/pareto_avg_margin", current_mean_margin)

            # Print Detailed Evaluation Table
            if self.verbose > 0:
                print(f"{BOLD}{'Pt':<4} | {'Preference w':<24} | {'Total Red%':<11} | {'Exec Red%':<11} | {'Keys Red%':<11} | {'Avg ' + metric_name:<16} | {'Safety Status'}{RESET}")
                print(f"{'-'*95}")
                for s in point_summaries:
                    safe_color = GREEN if s["safe_rate"] >= 100.0 else (YELLOW if s["safe_rate"] >= 80.0 else RED)
                    safe_str = f"{safe_color}{s['safe_rate']:>5.1f}% SAFE ({s['violations']} viol){RESET}"
                    print(
                        f"#{s['p_idx']:<3} | "
                        f"[ops={s['w'][0]:.2f}, keys={s['w'][1]:.2f}] | "
                        f"{s['cost_red']:>+9.1f}% | "
                        f"{s['exec_red']:>+9.1f}% | "
                        f"{s['keys_red']:>+9.1f}% | "
                        f"{s['constraint']:>6.1f} / {s['budget']:<4.0f} {s['metric_unit']} | "
                        f"{safe_str}"
                    )
                print(f"{'-'*95}")
                overall_color = GREEN if current_safe_rate >= 100.0 else (YELLOW if current_safe_rate >= 80.0 else RED)
                print(f"{BOLD}Front Summary (Step {self.num_timesteps:,}):{RESET}")
                print(f"  • {BOLD}Mean Pareto Reward{RESET}     : {CYAN}{current_mean_reward:.4f}{RESET} +/- {np.std(all_means):.4f}")
                print(f"  • {BOLD}Safe Compliance Rate{RESET}   : {overall_color}{current_safe_rate:.1f}% SAFE{RESET} ({total_violations} total violations / {len(all_compliances)} evaluations)")
                print(f"  • {BOLD}Avg Total Cost Red% {RESET}   : {GREEN if current_mean_cost_red > 0 else RED}{current_mean_cost_red:+.1f}%{RESET}")
                print(f"  • {BOLD}Avg Exec Cost Red%  {RESET}   : {GREEN if current_mean_exec_red > 0 else RED}{current_mean_exec_red:+.1f}%{RESET} (Ops)")
                print(f"  • {BOLD}Avg Keys Cost Red%  {RESET}   : {GREEN if current_mean_keys_red > 0 else RED}{current_mean_keys_red:+.1f}%{RESET} (Keys)")
                print(f"  • {BOLD}Avg Safe Headroom   {RESET}   : {CYAN}{current_mean_margin:+.1f} {metric_unit}{RESET}")
                print(f"  • {BOLD}Avg Episode Length  {RESET}   : {current_mean_length:.1f} steps")

            # Model Checkpointing for Best Score
            if current_mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"  ★ {GREEN}New Best Pareto Frontier Score: {current_mean_reward:.4f}! Saving checkpoint...{RESET}")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = current_mean_reward

            print(f"{BLUE}{'='*95}{RESET}\n")
            self.logger.dump(step=self.num_timesteps)

        return True


class CurriculumCallback(BaseCallback):
    """Gradually widens the set of budgets sampled during training.

    Parameters
    ----------
    budget_phases : list[list[int]]
        Budget lists for each phase.  Phase 0 is used from the start.
        Example: [[60,80], [60,80,100,200], [60,80,100,200,1000000]]
    phase_boundaries : list[float]
        Training progress thresholds (0→1) at which to advance.
        Example: [0.33, 0.66] means phase 1 starts at 33% and phase 2 at 66%.
    """

    def __init__(self, budget_phases, phase_boundaries, verbose=0):
        super().__init__(verbose)
        self.budget_phases = budget_phases
        self.phase_boundaries = phase_boundaries
        self._current_phase = 0

    def _on_training_start(self) -> None:
        self._update_envs(self.budget_phases[0])
        print(f"[Curriculum] Phase 0: budgets = {self.budget_phases[0]}")

    def _on_step(self) -> bool:
        if self._current_phase >= len(self.phase_boundaries):
            return True
        progress = 1.0 - self.model._current_progress_remaining
        if progress >= self.phase_boundaries[self._current_phase]:
            self._current_phase += 1
            new_budgets = self.budget_phases[self._current_phase]
            self._update_envs(new_budgets)
            print(f"[Curriculum] Phase {self._current_phase}: budgets = {new_budgets}")
        return True

    def _update_envs(self, budgets):
        env = self.model.get_env()
        while hasattr(env, 'venv'):
            env = env.venv
        env.env_method('set_active_budgets', budgets)


def linear_schedule(start: float, end: float = 0.0):
    def sched(progress_remaining):
        return (start - end) * progress_remaining + end
    return sched


class UnifiedMetricsCallback(BaseCallback):
    """Tracks and logs comprehensive metrics from the unified environment to TensorBoard.
    
    Logs:
    - unified/cost_final_mean
    - unified/cost_reduction_pct_mean
    - unified/exec_cost_final_mean
    - unified/exec_reduction_pct_mean
    - unified/keys_cost_final_mean
    - unified/keys_reduction_pct_mean
    - unified/constraint_final_mean
    - unified/violation_mean
    - unified/compliance_rate
    - unified/lagrangian_lambda
    """
    def __init__(self, pid_wrapper=None, verbose: int = 0):
        super().__init__(verbose)
        self.pid_wrapper = pid_wrapper
        self.ep_costs = []
        self.ep_cost_reductions = []
        self.ep_ops = []
        self.ep_ops_reductions = []
        self.ep_keys = []
        self.ep_keys_reductions = []
        self.ep_constraints = []
        self.ep_violations = []
        self.ep_compliances = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done and "final_cost" in info:
                self.ep_costs.append(info["final_cost"])
                self.ep_cost_reductions.append(info.get("cost_reduction", 0.0) * 100.0)
                self.ep_ops.append(info.get("final_ops", 0))
                self.ep_ops_reductions.append(info.get("ops_reduction", 0.0) * 100.0)
                self.ep_keys.append(info.get("final_keys", 0))
                self.ep_keys_reductions.append(info.get("keys_reduction", 0.0) * 100.0)
                self.ep_constraints.append(info.get("final_constraint", 0.0))
                self.ep_violations.append(info.get("violation", 0.0))
                self.ep_compliances.append(1.0 if info.get("is_compliant", True) else 0.0)
        return True

    def _on_rollout_end(self) -> None:
        if self.ep_costs:
            self.logger.record("unified/cost_final_mean", float(np.mean(self.ep_costs)))
            self.logger.record("unified/cost_reduction_pct_mean", float(np.mean(self.ep_cost_reductions)))
            self.logger.record("unified/exec_cost_final_mean", float(np.mean(self.ep_ops)))
            self.logger.record("unified/exec_reduction_pct_mean", float(np.mean(self.ep_ops_reductions)))
            self.logger.record("unified/keys_cost_final_mean", float(np.mean(self.ep_keys)))
            self.logger.record("unified/keys_reduction_pct_mean", float(np.mean(self.ep_keys_reductions)))
            self.logger.record("unified/constraint_final_mean", float(np.mean(self.ep_constraints)))
            self.logger.record("unified/violation_mean", float(np.mean(self.ep_violations)))
            self.logger.record("unified/compliance_rate", float(np.mean(self.ep_compliances)))
            self.ep_costs.clear()
            self.ep_cost_reductions.clear()
            self.ep_ops.clear()
            self.ep_ops_reductions.clear()
            self.ep_keys.clear()
            self.ep_keys_reductions.clear()
            self.ep_constraints.clear()
            self.ep_violations.clear()
            self.ep_compliances.clear()

        if self.pid_wrapper is not None:
            self.logger.record("unified/lagrangian_lambda", float(getattr(self.pid_wrapper, "lambda_penalty", 0.0)))