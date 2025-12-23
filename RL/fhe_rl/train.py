import os
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from .env    import fheEnv
from .policy import HierarchicalMaskablePolicy
from stable_baselines3 import PPO
from .utils  import load_expressions, create_rules, load_embeddings
from .logger import log_training_details
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize,DummyVecEnv
from .callbacks import linear_schedule, EntCoefScheduler, KeysWeightLogger, TimestepUpdater, DynamicEntCoefScheduler, CustomEvalCallback, WarmupPhaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from .schedules import step_schedule, linear_schedule as linear_schedule_keys, sigmoid_schedule, cosine_schedule

import random
import numpy as np
import torch

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_agent(expressions_file: str, embeddings_model, total_timesteps: int = 1_000_000, num_envs: int = 8, keys_schedule_type="step", transition_point=0.75, seed: int = 42, auto_transition: bool = False):
    set_random_seed(seed)
    benchmarks = load_expressions("./fhe_rl/datasets/benchmarks.txt") 
    expressions = load_expressions(expressions_file, benchmarks)
    max_positions = 16
    rules_list  = create_rules("rules.txt", "rotations_rules.txt")
    rules_list["END"] = None
    job_id = os.environ.get("SLURM_JOB_ID", "jobid")
    run_name = f"model_{job_id}"
    tensorboard_log_dir = f"./tensorboard/{run_name}"

    checkpoint_dir = f"./checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if keys_schedule_type == 'step':
        keys_weight_schedule = step_schedule(total_timesteps, transition_point=transition_point)
    elif keys_schedule_type == 'linear':
        keys_weight_schedule = linear_schedule_keys(total_timesteps, start_point=transition_point)
    elif keys_schedule_type == 'sigmoid':
        keys_weight_schedule = sigmoid_schedule(total_timesteps, midpoint=transition_point, steepness=10)
    elif keys_schedule_type == 'cosine':
        keys_weight_schedule = cosine_schedule(total_timesteps, start_point=transition_point)
    else:
        raise ValueError(f"Unknown schedule type: {keys_schedule_type}")

    checkpoint_path = None
    steps_done = 0
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("rl_model_") and f.endswith("_steps.zip")]
        if checkpoints:
            # Get the latest checkpoint
            latest = max(checkpoints, key=lambda x: int(x.split("_")[2]))
            checkpoint_path = os.path.join(checkpoint_dir, latest)
            steps_done = int(latest.split("_")[2])
            print(f"Found checkpoint: {checkpoint_path}")
            print(f"Resuming from step {steps_done}")

    # def make_env(seed, expressions):
    #     def _init():
    #         set_random_seed(seed)  # Seed the subprocess
    #         env = Monitor(fheEnv(rules_list, expressions, max_positions=max_positions, 
    #                             embeddings_model=embeddings_model, 
    #                             keys_weight_schedule=lambda t: 0.0,
    #                             seed=seed))
    #         return env
    #     return _init

    # env = SubprocVecEnv([make_env(seed + i, expressions=expressions) for i in range(num_envs)], start_method='spawn')
    # val_env = DummyVecEnv([make_env(seed + num_envs, expressions=benchmarks)])

    def make_env(): return Monitor(fheEnv(rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model, keys_weight_schedule = lambda t: 0.0))
    env = SubprocVecEnv([make_env for _ in range(num_envs)], start_method='spawn')    
    val_env = DummyVecEnv([
    lambda: Monitor(fheEnv(rules_list, benchmarks, max_positions=max_positions,embeddings_model=embeddings_model, keys_weight_schedule=lambda t: 0.0))
    ])
    ent_schedule = linear_schedule(0.1)
    model_params = {
        "policy": HierarchicalMaskablePolicy,
        "env": env,
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.98,
        "n_epochs": 15,
        "clip_range": 0.1,
        "clip_range_vf": 0.2,
        "ent_coef": 0.1,
        "verbose": 1,
        "tensorboard_log": tensorboard_log_dir,
        "seed": seed,
        "policy_kwargs": {
            "ent_coef": 0.1,
            "rule_dim":      len(rules_list),
            "max_positions": max_positions,
            "rule_hidden_dims":   [128, 64],
            "pos_hidden_dims":    [64, 64],
            "value_hidden_dims":    [256, 128, 64],
            "seed": seed,
        }
    }
    ## model = PPO(**model_params)

    # Load model from checkpoint or create new
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, tensorboard_log=tensorboard_log_dir)
    else:
        model = PPO(**model_params)

    log_training_details(
        model_params,
        job_id,
        num_data=len(expressions),
        num_actions=len(rules_list),
        total_timesteps=total_timesteps,
        output_model_name=run_name,
        notes="2 level hierarchical PPO max steps 75 and 8 envs"
    )
    num_benchmarks = len(benchmarks)

    checkpoint_callback = CheckpointCallback(
        save_freq=80000,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )

    ent_scheduler = DynamicEntCoefScheduler(total_timesteps=total_timesteps)

    eval_callback = CustomEvalCallback(
            val_env, 
            best_model_save_path=f"./eval/best_model_{run_name}", 
            log_path=tensorboard_log_dir, 
            eval_freq=4096,
            n_eval_episodes=num_benchmarks,
            deterministic=True, 
            render=False, 
            verbose=1,
            ent_scheduler=ent_scheduler
    )

    # eval_callback = EvalCallback(
    #     val_env, 
    #     best_model_save_path=f"./eval/best_model_{run_name}", 
    #     log_path=tensorboard_log_dir, 
    #     eval_freq=8000,
    #     n_eval_episodes=num_benchmarks,
    #     deterministic=True, 
    #     render=False, 
    #     verbose=1
    # )

    if checkpoint_path:
        remaining_steps = total_timesteps - steps_done
        print(f"Resuming training: {remaining_steps} steps remaining out of {total_timesteps} total")
    else:
        remaining_steps = total_timesteps
        print(f"Starting fresh training: {total_timesteps} total steps")

    warmup_callback = WarmupPhaseCallback(verbose=1)
    timestep_updater = TimestepUpdater(verbose=1)
    keys_logger = KeysWeightLogger(verbose=1)


    model.learn(
        total_timesteps=remaining_steps, 
        log_interval=1, 
        progress_bar=True, 
        # callback=[eval_callback, ent_scheduler, warmup_callback, checkpoint_callback, timestep_updater, keys_logger],
        callback=[eval_callback, ent_scheduler, checkpoint_callback, timestep_updater, keys_logger],
        reset_num_timesteps = (checkpoint_path is None)
    )
    model.save(run_name)    