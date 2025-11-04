from stable_baselines3.common.callbacks import BaseCallback


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
    
    def _on_step(self) -> bool:
        # Get weight from first environment (they should all be in sync)
        if hasattr(self.training_env, 'get_attr'):
            weights = self.training_env.get_attr('current_keys_weight')
            if weights:
                current_weight = weights[0]
                self.logger.record('train/keys_weight', current_weight)
                self.last_logged_weight = current_weight
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

        self.transition_timestep = int(self.total_timesteps * self.transition_point)

    def _on_training_start(self) -> None:
        # Set ent_coef at start
        ent_coef = self.initial_ent
        self.model.ent_coef = float(ent_coef)
        if self.verbose > 0:
            print(f"[DynamicEntCoefScheduler] Training start: ent_coef = {ent_coef}", flush=True)
        self.logger.record("train/ent_coef", float(ent_coef))

    def _on_rollout_end(self) -> bool:
        current_t = self.model.num_timesteps
        if current_t > self.total_timesteps:
            current_t = self.total_timesteps

        if current_t < self.transition_timestep:
            # Phase 1: decay from initial_ent to min_ent
            frac = current_t / self.transition_timestep
            ent_coef = self.initial_ent + frac * (self.min_ent - self.initial_ent)
        elif current_t == self.transition_timestep:
            # Exactly at transition point: jump to boost_ent
            ent_coef = self.boost_ent
        else:
            # Phase 2: decay from boost_ent to min_ent until end
            frac2 = (current_t - self.transition_timestep) / (self.total_timesteps - self.transition_timestep)
            ent_coef = self.boost_ent + frac2 * (self.min_ent - self.boost_ent)

        # Update the model’s entropy coefficient
        self.model.ent_coef = float(ent_coef)
        # Log for TensorBoard or logger
        self.logger.record("train/ent_coef", float(ent_coef))

    def _on_step(self) -> bool:
        return True


def linear_schedule(start: float, end: float = 0.0):
    def sched(progress_remaining):
        return (start - end) * progress_remaining + end
    return sched