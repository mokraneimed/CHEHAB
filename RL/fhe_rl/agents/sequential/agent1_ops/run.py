from stable_baselines3 import PPO
import time
from ....shared.utils import load_expressions, create_rules, parse_sexpr, load_embeddings,predict_method,calc_vec_sizes
import sys
from .env import opsEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
from ....shared.policy import HierarchicalMaskablePolicy
import sys, importlib

from stable_baselines3.common.monitor import Monitor
def run_ops_agent(model, output_file: str, expressions, env):

    start_time = time.perf_counter()
    # expressions = load_expressions(expressions_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    print(expressions)
    # rules_list = create_rules(ops_rules_path = "rules.txt")
    results = []
    max_positions = 16
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    env.envs[0].env.set_expressions(expressions)
    # env = DummyVecEnv([
    # lambda: Monitor(opsEnv(rules_list, expressions, max_positions=max_positions,embeddings_model=embeddings_model))
    # ])
    # model = PPO(
    #     policy=HierarchicalMaskablePolicy,
    #     env=env
    # )
    # sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    # model = model.load(model_filepath)
    start_time = time.perf_counter()
    obs = env.reset()
    wrapper  = env.envs[0]
    ops_env   = wrapper.env
    test_expr = ops_env.initial_expression
    initial_cost = ops_env.initial_cost
    done = False
    steps = 0
    last_expr = None
    last_embedding = None
    bad_count = 0
    while not done:
        last_expr = ops_env.expression
        last_cost = ops_env.current_cost
        last_embedding = ops_env.embedded_expr
        action, _ = model.predict(obs, deterministic=True)
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
    vec_sizes=" ".join(str(x) for x in calc_vec_sizes(parsed))
    with open (output_file, "w") as f:
        f.write(last_expr+"\n"+vec_sizes)
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    return last_expr, last_embedding