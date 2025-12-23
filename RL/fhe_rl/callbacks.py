from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


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

class KeysWeightLogger(BaseCallback):
    """Logs the current keys weight to tensorboard"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_logged_weight = None
    
    def _on_rollout_end(self) -> bool:
        # Get weight from first environment (they should all be in sync)
        if hasattr(self.training_env, 'get_attr'):
            weights = self.training_env.get_attr('current_keys_weight')
            if weights:
                current_weight = weights[0]
                self.logger.record('train/keys_weight', current_weight)
                self.last_logged_weight = current_weight
        return True
    def _on_step(self) -> bool:
        return True
     
class TimestepUpdater(BaseCallback):
    """
    Callback that updates all environments with the current global timestep.
    
    This ensures all environments use the correct timestep for their
    keys weight schedule, even when resuming from checkpoints.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_logged_timestep = None

    def _on_training_start(self) -> None:
        """
        Called at the very beginning of training (or when resuming).
        This is CRITICAL for checkpoint resuming to set the correct initial weight.
        """
        current_timestep = self.model.num_timesteps
        
        if self.verbose > 0:
            print(f"[TimestepUpdater] Training start - setting all envs to timestep: {current_timestep}", flush=True)
        
        # Update all environments with the current timestep
        try:
            self.training_env.env_method('set_timestep', current_timestep)
        except AttributeError:
            print("[TimestepUpdater] Warning: training_env does not support env_method. Trying direct call.", flush=True)
            if hasattr(self.training_env, 'set_timestep'):
                self.training_env.set_timestep(current_timestep)
        
        # Also log the keys weight for verification
        if self.verbose > 0:
            try:
                weights = self.training_env.get_attr('current_keys_weight')
                if weights:
                    print(f"[TimestepUpdater] Keys weights set to: {weights[0]:.4f}", flush=True)
            except:
                pass 
    
    def _on_rollout_end(self) -> None:
        current_timestep = self.model.num_timesteps
        if self.verbose > 0 and (self.last_logged_timestep is None or current_timestep - self.last_logged_timestep >= 10000):
            print(f"[TimestepUpdater] Rollout ended - updating envs to timestep: {current_timestep}", flush=True)
            self.last_logged_timestep = current_timestep
        
        try:
            self.training_env.env_method('set_timestep', current_timestep)
        except AttributeError:
            print("[TimestepUpdater] Warning: training_env does not support env_method. Trying direct call.", flush=True)
            if hasattr(self.training_env, 'set_timestep'):
                self.training_env.set_timestep(current_timestep)

    def _on_step(self) -> bool:
        return True

class DynamicEntCoefScheduler(BaseCallback):
    """
    Schedules entropy coefficient with a boost during objective transition.

    Timeline:
    - 0 → transition_point: decay from initial_ent → min_ent
    - At transition_point: jump to boost_ent
    - transition_point → end: decay from boost_ent → min_ent
    """

    def __init__(self, initial_ent=0.1, min_ent=0.0, boost_ent=0.07,
                 transition_point=0.75, total_timesteps=1_000_000, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.min_ent = min_ent
        self.boost_ent = boost_ent
        self.transition_point = transition_point
        self.total_timesteps = total_timesteps

        self.keys_cost_triggered = False
        self.trigger_timestep = None        

        self.transition_timestep = int(self.total_timesteps * self.transition_point)

    def _on_training_start(self) -> None:
        # Set ent_coef at start
        ent_coef = self.initial_ent
        self.model.ent_coef = float(ent_coef)
        if self.verbose > 0:
            print(f"[DynamicEntCoefScheduler] Training start: ent_coef = {ent_coef}", flush=True)
        self.logger.record("train/ent_coef", float(ent_coef))

    def trigger_keys_cost(self):
        """Called by CustomEvalCallback when keys cost objective is triggered."""
        if not self.keys_cost_triggered:
            self.keys_cost_triggered = True
            self.trigger_timestep = self.model.num_timesteps
            
            if self.verbose > 0:
                print(f"\n[DynamicEntCoefScheduler] Keys cost triggered at timestep {self.trigger_timestep}", flush=True)
                print(f"[DynamicEntCoefScheduler] Boosting entropy to {self.boost_ent}, will decay to {self.min_ent} until end", flush=True)

    def _on_rollout_end(self) -> bool:
        current_t = self.model.num_timesteps
        if current_t > self.total_timesteps:
            current_t = self.total_timesteps

        # if current_t < self.transition_timestep:
        #     # Phase 1: decay from initial_ent to min_ent
        #     frac = current_t / self.transition_timestep
        #     ent_coef = self.initial_ent + frac * (self.min_ent - self.initial_ent)
        # elif current_t == self.transition_timestep:
        #     # Exactly at transition point: jump to boost_ent
        #     ent_coef = self.boost_ent
        # else:
        #     # Phase 2: decay from boost_ent to min_ent until end
        #     frac2 = (current_t - self.transition_timestep) / (self.total_timesteps - self.transition_timestep)
        #     ent_coef = self.boost_ent + frac2 * (self.min_ent - self.boost_ent)

        if not self.keys_cost_triggered:
            # Phase 1: Before trigger - decay from initial_ent to min_ent
            frac = current_t / self.total_timesteps
            ent_coef = self.initial_ent + frac * (self.min_ent - self.initial_ent)
        
        else:
            # Phase 2: After trigger - decay from boost_ent to min_ent
            remaining_total = self.total_timesteps - self.trigger_timestep
            if remaining_total > 0:
                frac = (current_t - self.trigger_timestep) / remaining_total
                ent_coef = self.boost_ent + frac * (self.min_ent - self.boost_ent)
            else:
                ent_coef = self.min_ent        

        # Update the model’s entropy coefficient
        self.model.ent_coef = float(ent_coef)
        # Log for TensorBoard or logger
        self.logger.record("train/ent_coef", float(ent_coef))

    def _on_step(self) -> bool:
        return True

class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, ent_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_file = 'mean_rewards.txt'
        self.previous_mean_reward = None
        self.counter = 0
        self.improvement_threshold = 5  # Minimum improvement to reset counter
        self.counter_threshold = 1  # Number of evaluations with no improvement before stopping
        self.keys_cost_triggered = False
        self.ent_scheduler = ent_scheduler



    def _init_callback(self) -> None:
        """Called when callback is initialized - find the entropy scheduler."""
        super()._init_callback()
        
        # If using CallbackList, search for the entropy scheduler
        if hasattr(self.parent, 'callbacks'):
            for callback in self.parent.callbacks:
                if isinstance(callback, DynamicEntCoefScheduler):
                    self.ent_scheduler = callback
                    if self.verbose > 0:
                        print("[CustomEvalCallback] Found DynamicEntCoefScheduler", flush=True)
                    break

    def _on_step(self) -> bool:
        # You can add custom logging or behavior here if needed
        result = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            # with open(self.save_file, 'a') as f:
            #     f.write(f"{self.last_mean_reward}\n")
            if not self.keys_cost_triggered:
                if self.previous_mean_reward is None or self.last_mean_reward >= self.previous_mean_reward + self.improvement_threshold:
                    self.previous_mean_reward = self.last_mean_reward
                    self.counter = 0
                else:
                    self.counter += 1
                    if self.counter >= self.counter_threshold:
                        self.keys_cost_triggered = True
                        # self.in_warmup = True
                        # self.training_env.env_method('set_warmup', True)
                        print(f"No improvement in mean reward for {self.counter} evaluations. New objective triggered.")
                        # self.model.stop_training = True
                        try:
                            self.training_env.env_method('trigger_keys_cost')
                            print("[CustomEvalCallback] Training envs: keys cost triggered")
                            
                            # Trigger evaluation environment (IMPORTANT!)
                            self.eval_env.env_method('trigger_keys_cost')
                            print("[CustomEvalCallback] Eval env: keys cost triggered")
                            
                        except AttributeError:
                            print("[TimestepUpdater] Warning: training_env does not support env_method. Trying direct call.", flush=True)
                            if hasattr(self.training_env, 'set_timestep'):
                                self.training_env.trigger_keys_cost()

                            if hasattr(self.eval_env, 'trigger_keys_cost'):
                                self.eval_env.trigger_keys_cost()
                                print("[CustomEvalCallback] Eval env: keys cost triggered", flush=True)                                

                        if self.ent_scheduler is not None:
                            self.ent_scheduler.trigger_keys_cost()
                            if self.verbose > 0:
                                print("[CustomEvalCallback] Successfully triggered entropy boost!", flush=True)
                        else:
                            print("[CustomEvalCallback] Warning: DynamicEntCoefScheduler not found!", flush=True)                                                
        return result    

class WarmupPhaseCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.last_logged_rollout = None

    def _on_rollout_end(self) -> None:
        in_warmup = self.training_env.get_attr('in_warmup')[0]
        warmup_rollouts = self.training_env.get_attr('warmup_rollouts')[0]
        warmup_counter = self.training_env.get_attr('warmup_counter')[0]

        if in_warmup:
            self.training_env.env_method('increment_warmup')
            warmup_counter += 1
            # self.last_logged_rollout = self.model.rollout_buffer
            # self.model.rollout_buffer.reset()

            if self.verbose > 0:
                print(f"\n[WarmupPhase] Rollout {warmup_counter}/{warmup_rollouts}")
                print(f"[WarmupPhase] Collecting data under new objective, SKIPPING policy update")

              

            if warmup_counter > warmup_rollouts:
                self.training_env.env_method('set_warmup', False)
                # self.model.rollout_buffer = self.last_logged_rollout

                if self.verbose > 0:
                    print(f"\n{'='*80}")
                    print(f"✅ WARMUP PHASE COMPLETE")
                    print(f"{'='*80}")
                    print(f"Resuming normal training with original method")
                    print(f"Policy will now learn under the new objective")
                    print(f"{'='*80}\n")
        

    def _on_step(self) -> bool:
        return True            

def linear_schedule(start: float, end: float = 0.0):
    def sched(progress_remaining):
        return (start - end) * progress_remaining + end
    return sched