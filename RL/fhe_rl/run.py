from stable_baselines3 import PPO
import time
from .utils import load_expressions, create_rules, parse_sexpr, calc_vec_sizes
import sys
import importlib
from stable_baselines3.common.vec_env import DummyVecEnv
from .config import get_env_class, get_policy_class
from stable_baselines3.common.monitor import Monitor


def run_agent(expressions_file: str, embeddings_model, model_filepath: str, output_file: str, noise_budget: int = 300, framework: str = "constrained"):
    start_time = time.perf_counter()
    expressions = load_expressions(expressions_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    from pathlib import Path
    rules_path = Path("rules.txt")
    if not rules_path.is_file():
        rules_path = Path(__file__).resolve().parent.parent / "rules.txt"
    rules_list = create_rules(str(rules_path))
    rules_list["END"] = None
    results = []
    max_positions = 16
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    EnvCls = get_env_class()
    PolicyCls = get_policy_class()
    if framework in ("constrained", "constrained_rl"):
        env_fn = lambda: Monitor(EnvCls(
            rules_list, expressions, max_positions=max_positions,
            embeddings_model=embeddings_model,
            budget_options=[230, 236, 369, 9_000_000],
            constraint_method="nato_sc",
            verbose=False,
        ))
    else:
        env_fn = lambda: Monitor(EnvCls(
            rules_list, expressions, max_positions=max_positions,
            embeddings_model=embeddings_model,
            budget_options=[9_000_000],
            constraint_method="none",
            verbose=False,
        ))
    if "fhe_rl_new" not in sys.modules:
        try:
            sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
        except Exception:
            pass
    try:
        import fhe_rl.policy
        fhe_rl.policy.HierarchicalMaskablePolicy = PolicyCls
    except Exception:
        pass

    env = DummyVecEnv([env_fn])
    env.set_options({ "budget": noise_budget })

    model = PPO(
        policy=PolicyCls,
        env=env
    )
    model = model.load(model_filepath)
    model_obs_keys = set(
        list(model.observation_space.spaces.keys())
        if hasattr(model, "observation_space") and hasattr(model.observation_space, "spaces") else []
    )
    start_time = time.perf_counter()
    obs = env.reset()
    wrapper  = env.envs[0]
    fhe_env   = wrapper.env
    test_expr = fhe_env.initial_expression
    initial_cost = fhe_env.initial_cost
    done = False
    steps = 0
    last_expr = None
    bad_count = 0
    while not done:
        last_expr = fhe_env.expression
        last_cost = fhe_env.current_cost
        obs_for_model = {
            k: v for k, v in obs.items()
            if not model_obs_keys or k in model_obs_keys or k in ("action_mask", "observation")
        }
        action, _ = model.predict(obs_for_model, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        # if rewards[0] < 0:
        #     bad_count += 1
        #     # on first bad decision, remember the expression
        #     if bad_count == 1:
        #         final_expr = last_expr
        # else:
        #     # any non-negative reward resets the streak
        #     bad_count = 0
        #     final_expr = None

        # # once we've seen 3 negatives in a row, capture and break
        # if bad_count >= 3:
        #     final_expr = last_expr
        #     break
        done = bool(dones[0])
        steps += 1
    # if final_expr is not None:
    #     last_expr = final_expr
    parsed = parse_sexpr(last_expr)
    vec_s = calc_vec_sizes(parsed)
    if vec_s:
        sz = vec_s[0]
        footer = f"{sz} {sz} 1 0"
    else:
        footer = "1 1 1 0"
    with open(output_file, "w") as f:
        f.write(last_expr + "\n" + footer + "\n")
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    