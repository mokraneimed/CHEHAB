from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import torch
import os

class EntCoefScheduler(BaseCallback):
    def __init__(self, schedule, verbose: int = 0):
        super().__init__(verbose)
        self.schedule = schedule
        self.rollout_count = 0

    def _on_training_start(self) -> None:
        p = self.model._current_progress_remaining
        self.model.ent_coef = float(self.schedule(p))

    def _on_rollout_end(self) -> None:
        # p = self.model._current_progress_remaining
        # self.model.ent_coef = float(self.schedule(p))
        self.rollout_count += 1
        
        # --- CHANGE: Update entropy ONLY every 2nd rollout ---
        if self.rollout_count % 2 == 0:
            p = self.model._current_progress_remaining
            self.model.ent_coef = float(self.schedule(p))
            if self.verbose > 0:
                print(f"[Entropy] Updated to {self.model.ent_coef:.4f} at rollout {self.rollout_count}")        

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
                print(f"\nStep {self.num_timesteps}: Starting Pareto Evaluation ({len(self.pref_list)} points)")
            
            all_means = []
            all_lengths = []
            
            for w in self.pref_list:
                # Force eval env to this specific goal
                self.eval_env.env_method("set_preference_vector", w)
                
                # Run evaluation
                episode_rewards, episode_lengths = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes, 
                    deterministic=self.deterministic,
                    return_episode_rewards=True 
                )
                
                mean_r = np.mean(episode_rewards)
                all_means.append(mean_r)
                all_lengths.append(np.mean(episode_lengths))

            # 1. Calculate the Global Score
            current_mean_reward = np.mean(all_means)
            current_mean_length = np.mean(all_lengths)

            # 2. Log to Tensorboard/Logger (standard EvalCallback style)
            self.logger.record("eval/pareto_avg_reward", current_mean_reward)
            self.logger.record("eval/pareto_avg_ep_length", current_mean_length)
            
            # Optional: Log specific weight performance for deeper insight
            for i, w in enumerate(self.pref_list):
                self.logger.record(f"eval/reward_w_{w[1]}", all_means[i])
            
            self.eval_env.env_method("unlock_preferences")

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={current_mean_reward:.2f} +/- {np.std(all_means):.2f}")
                print(f"Episode length: {current_mean_length:.2f} +/- {np.std(all_lengths):.2f}")

            # 3. Check if this is the "Best" model found so far
            if current_mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                
                self.best_mean_reward = current_mean_reward

            # Trigger potential logging for SB3 monitoring
            self.logger.dump(step=self.num_timesteps)

        return True

class PreferenceSamplerCallback(BaseCallback):
    def __init__(self, eval_env, use_cl=False, alpha=1.0, total_timesteps=1_000_000, verbose=0):
        super().__init__(verbose)
        self.use_cl = use_cl
        self.alpha = alpha
        self.total_timesteps = total_timesteps
        self.eval_env = eval_env
        # --- MINIMAL CHANGE: Add a toggle flag ---
        self.toggle_speed = True
    def _on_rollout_start(self) -> None:
        # progress = self.num_timesteps / self.total_timesteps
        # if self.use_cl:
        #     if progress < 0.5:
        #         w_keys = 0.0
        #     elif progress < 0.75:
        #         max_key_w = (progress - 0.5) / 0.25
        #         w_keys = np.random.uniform(0.0, max_key_w)
        #     else :
        #         alpha_vec = np.array([self.alpha, self.alpha])  
        #         w = np.random.dirichlet(alpha_vec)
        #         w_keys = w[1]
        #     if progress < 0.75:
        #         w = np.array([1.0 - w_keys, w_keys], dtype=np.float32)
        # else:
        #     alpha_vec = np.array([self.alpha, self.alpha])
        #     w = np.random.dirichlet(alpha_vec).astype(np.float32)
                # --- MINIMAL CHANGE: Logic to switch between [1,0] and [0,1] ---
        if self.toggle_speed:
            w = np.array([1.0, 0.0], dtype=np.float32) # Pure Speed
        else:
            w = np.array([0.0, 1.0], dtype=np.float32) # Pure Keys
            
        # Switch the flag for the NEXT rollout
        self.toggle_speed = not self.toggle_speed
                
        self.training_env.env_method("set_preference_vector", w)
        self.eval_env.env_method("set_preference_vector", w)
        if self.verbose > 0:
            print(f"[PEARL] Sampled w: {w} (alpha={self.alpha})")

    def _on_step(self) -> bool:
        return True        

def linear_schedule(start: float, end: float = 0.0):
    def sched(progress_remaining):
        return (start - end) * progress_remaining + end
    return sched

def update_buffer(training_env, model):
    try:
        alt_rewards_per_env = training_env.env_method('get_and_clear_alternate_rewards')
        buffer = model.rollout_buffer
        total_steps = buffer.buffer_size * buffer.n_envs
        all_alt_rewards = []
        for env_rewards in alt_rewards_per_env:
            all_alt_rewards.extend(env_rewards)
        all_alt_rewards = np.array(all_alt_rewards, dtype=np.float32)
        if len(all_alt_rewards) >= total_steps:
            recent_rewards = all_alt_rewards[-total_steps:]
            new_rewards = recent_rewards.reshape(buffer.buffer_size, buffer.n_envs)
            old_reward_mean = buffer.rewards.mean()
            buffer.rewards = new_rewards
            with torch.no_grad():
                last_obs = {
                    k: torch.as_tensor(v[-1], device=model.device)
                    for k, v in buffer.observations.items()
                }
                last_values = model.policy.predict_values(last_obs)
            buffer.compute_returns_and_advantage(
                last_values=last_values,
                dones=buffer.episode_starts[-1]
            )
            new_reward_mean = buffer.rewards.mean()
            print(f"[Phase Transition] Rewards updated in buffer:")
            print(f"  Old mean reward: {old_reward_mean:.4f}")
            print(f"  New mean reward: {new_reward_mean:.4f}")
            print(f"  Buffer size: {buffer.buffer_size} steps × {buffer.n_envs} envs")
            print(f"  Advantages recomputed ✓")
            print(f"{'='*80}\n")            
        else:
            print(f"[Warning] Not enough alternate rewards collected:")
            print(f"  Expected: {total_steps}, Got: {len(all_alt_rewards)}")
            print(f"  Skipping reward replacement")
    except:
            print(f"[Error] Failed to replace rewards")
            import traceback
            traceback.print_exc()           