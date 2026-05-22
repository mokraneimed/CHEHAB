import time
from ..shared.utils import load_expressions, create_rules, parse_sexpr, calc_vec_sizes
import sys
from ..agents.sequential.agent2_key.env import keysEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from ..agents.sequential.agent1_ops.env import opsEnv
from ..shared.ops_cache import load_cache
import importlib
from stable_baselines3 import PPO
from ..shared.policy import HierarchicalMaskablePolicy

def run_agent(expression_file: str, embeddings_model, model_filepath: str,output_file: str, w_ops=0.5, w_keys=0.5):
    pref = [w_ops, w_keys]
    start_time = time.perf_counter()
    expressions = load_expressions(expression_file)
    if not len(expressions):
        print("No valid expressions found in the file.")
        sys.exit(1)
        return
    print(expressions)
    rotations_rules_list = create_rules(rotations_rules_path='rotations_rules.txt')
    ops_rules_list  = create_rules(ops_rules_path="rules.txt")
    rotations_rules_list["END"] = None
    ops_rules_list["END"] = None

    max_positions = 16
    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time

    ops_model_filepath = "./fhe_rl/trained_models/sorl/agent_dynamic_llm_data.zip"
    ops_output_file = "try_this.txt"
    lambda_env = 0.0

    cache_path = "./fhe_rl/shared/ops_cache.json"
    ops_cache = load_cache(cache_path)

    ops_env = DummyVecEnv([lambda: Monitor(opsEnv(ops_rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model))])

    keys_env = DummyVecEnv([lambda: Monitor(keysEnv(rotations_rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model, 
                                   ops_env=ops_env, ops_model_filepath=ops_model_filepath, ops_output_file=ops_output_file, lambda_env=lambda_env, 
                                   pref_list=[pref], env_idx=0, ops_cache=ops_cache, cache_path=cache_path,))])
    
    keys_env.env_method("set_preference_vector", pref)
    model = PPO(
        policy=HierarchicalMaskablePolicy,
        env=keys_env
    )
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    sys.modules["fhe_rl_new.policy"] = importlib.import_module("fhe_rl.shared.policy")    

    model = model.load(model_filepath)
    
    obs = keys_env.reset()
    wrapper  = keys_env.envs[0]
    fhe_env   = wrapper.env
    done = False
    steps = 0
    last_expr = None
    while not done:
        last_expr = fhe_env.expression
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = keys_env.step(action)
        done = bool(dones[0])
        steps += 1
    parsed = parse_sexpr(last_expr)
    vec_sizes=" ".join(str(x) for x in calc_vec_sizes(parsed))    
    with open (output_file, "w") as f:
        f.write(last_expr+"\n"+vec_sizes)