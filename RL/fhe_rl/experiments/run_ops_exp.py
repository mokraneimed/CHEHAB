##

# 2. USE ABSOLUTE IMPORTS: Now Python knows what "fhe_rl" is
from ..agents.sequential.agent1_ops.run import run_ops_agent
from fhe_rl.agents.sequential.agent1_ops.env import opsEnv
from stable_baselines3 import PPO
from ..shared.policy import HierarchicalMaskablePolicy
from ..shared.utils import load_expressions, create_rules, load_embeddings
from ..shared.config import get_model_path, get_tokenizer_type, get_device
from ..shared.utils import load_embeddings
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize,SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import sys
import importlib

# ---------------------------------------------------------
# MAIN SCRIPT LOGIC
# ---------------------------------------------------------
# Note: Since you are running this from inside the 'experiments' folder, 
# you need to make sure the paths to your text files point to the right place!
def run_ops_experiment():
    rules_path = "rules.txt"
    benchmarks_path = "./fhe_rl/shared/datasets/benchmarks.txt"

    actual_embeddings_path =  "./fhe_rl/trained_models/embeddings_ROT_15_32_5m_10742576.pth"

    rules_list = create_rules(rules_path)
    rules_list["END"] = None
    expressions = load_expressions(benchmarks_path)

    max_positions = 16
    embeddings_model, _ = load_embeddings(tokenizer_type="dynamic", checkpoint_path=actual_embeddings_path, device="cpu")

    # Ensure the model path is correct relative to the RL folder
    ops_model_filepath ="./fhe_rl/trained_models/sorl/agent_dynamic_llm_data.zip"
    output_file = "try_this.txt"

    # Initialize Environment
    ops_env = DummyVecEnv([lambda: Monitor(opsEnv(rules_list, expressions, max_positions=max_positions, embeddings_model=embeddings_model))])

    # Load Model
    sys.modules["fhe_rl_new"] = importlib.import_module("fhe_rl")
    sys.modules["fhe_rl_new.policy"] = importlib.import_module("fhe_rl.shared.policy")
    model = PPO.load(
        ops_model_filepath,
        env=ops_env,
        custom_objects={
            "policy_kwargs": {
                "features_dim": 256,
                "rule_dim": 85,
                "max_positions": 16,
                "rule_hidden_dims":  [128, 64],
                "pos_hidden_dims":   [128, 64],
                "value_hidden_dims": [256, 128, 64],
            }
        }
    )

    # Run Inference
    opt_expression, opt_embeddings = run_ops_agent(model, output_file, expressions, ops_env)

    print(opt_expression)
    print(len(opt_embeddings))