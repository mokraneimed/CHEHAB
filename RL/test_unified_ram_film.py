"""
Rigorous Test Suite for Unified RL (MORL + Constrained Optimization)
Tests:
1. Framework registry and configuration resolution
2. fheEnvUnified with Peak RAM ('mem') constraint and MemoryEstimator
3. fheEnvUnified with Noise ('noise') constraint and NoiseEstimator
4. Observation space conformity (observation, budget_one_hot, action_mask, preference_vector, noise_ratio)
5. Reward calculation and NATO-SC quadratic violation penalty
6. Policy Feature Extractor: Design A (film_a: budget-only FiLM)
7. Policy Feature Extractor: Design B (film_b: budget+preference FiLM)
8. HierarchicalMaskablePolicy forward pass for both film_a and film_b
9. PIDLagrangianWrapper integration
10. End-to-end 16-step PPO learning smoke test on unified env
"""

import sys
import os
import unittest
import numpy as np
import torch
import gymnasium as gym

# Ensure RL is on sys.path and is cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.abspath(SCRIPT_DIR)
os.chdir(RL_DIR)

if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)
if os.path.join(RL_DIR, "pytrs") not in sys.path:
    sys.path.insert(0, os.path.join(RL_DIR, "pytrs"))

from fhe_rl.config import set_framework, get_framework, get_framework_spec, FRAMEWORKS
from fhe_rl.env_unified import fheEnvUnified
from fhe_rl.env import fheEnv
from fhe_rl.policy import CustomFeaturesExtractor, HierarchicalMaskablePolicy
from fhe_rl.policy_film_b import CustomFeaturesExtractorFiLMB, HierarchicalMaskablePolicyFiLMB
from fhe_rl.memory import MemoryEstimator
from pytrs import NoiseEstimator, create_rules
from fhe_rl.utils import load_expressions
from fhe_rl.algos.lagrangian_pid import PIDLagrangianWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


class TestUnifiedRL(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rules_path = os.path.join(RL_DIR, "rules.txt")
        rot_rules_path = os.path.join(RL_DIR, "rotations_rules.txt")
        cls.rules_list = create_rules(rules_path, rot_rules_path)
        cls.rules_list["END"] = None
        
        benchmarks_path = os.path.join(RL_DIR, "fhe_rl", "datasets", "benchmarks.txt")
        if os.path.exists(benchmarks_path):
            cls.test_expressions = load_expressions(benchmarks_path)[:3]
        else:
            cls.test_expressions = ["(Vec ( + ( * v1_0 v2_0 ) ( * v1_1 v2_1 ) ) )"]

    def test_01_framework_registry(self):
        """Verify FRAMEWORKS registry and specification integrity."""
        for fw in ["constrained", "morl", "unified", "baseline"]:
            set_framework(fw)
            self.assertEqual(get_framework(), fw)
            spec = get_framework_spec()
            self.assertIn("env", spec)
            self.assertIn("policy", spec)
            self.assertIn("pref_mode", spec)

        # Set back to unified
        set_framework("unified")
        spec = get_framework_spec()
        self.assertEqual(spec["env"], "fhe_rl.env_unified.fheEnvUnified")
        self.assertEqual(spec["pref_mode"], "pareto")

    def test_02_env_mem_constraint(self):
        """Verify fheEnvUnified instantiation and reset with Peak RAM ('mem') metric."""
        env = fheEnvUnified(
            self.rules_list,
            self.test_expressions,
            max_positions=16,
            constraint_metric="mem",
            budget_options=[1500, 2000, 2600, 1000000],
            constraint_method="nato_sc",
            verbose=False,
        )
        self.assertEqual(env.constraint_metric, "mem")
        self.assertIsInstance(env.estimator, MemoryEstimator)
        self.assertEqual(env.metric_name, "Peak RAM")
        self.assertEqual(env.metric_unit, "MB")

        obs, info = env.reset(seed=42)
        # Check observation keys
        for key in ["observation", "budget_one_hot_encoding", "action_mask", "preference_vector", "noise_ratio"]:
            self.assertIn(key, obs)

        self.assertEqual(obs["observation"].shape, (256,))
        self.assertEqual(obs["budget_one_hot_encoding"].shape, (4,))
        self.assertEqual(obs["preference_vector"].shape, (2,))
        self.assertEqual(obs["noise_ratio"].shape, (1,))
        self.assertGreater(info["constraint_value"], 0.0)
        self.assertIn(info["budget"], [1500, 2000, 2600, 1000000])

    def test_03_env_noise_constraint(self):
        """Verify fheEnvUnified instantiation and reset with Noise ('noise') metric."""
        env = fheEnvUnified(
            self.rules_list,
            self.test_expressions,
            max_positions=16,
            constraint_metric="noise",
            budget_options=[230, 236, 369, 9000000],
            constraint_method="nato_sc",
            verbose=False,
        )
        self.assertEqual(env.constraint_metric, "noise")
        self.assertIsInstance(env.estimator, NoiseEstimator)
        self.assertEqual(env.metric_name, "Noise")
        self.assertEqual(env.metric_unit, "bits")

        obs, info = env.reset(seed=42)
        self.assertEqual(obs["budget_one_hot_encoding"].shape, (4,))
        self.assertIn(info["budget"], [230, 236, 369, 9000000])
        self.assertGreaterEqual(info["constraint_value"], 0.0)

    def test_04_nato_sc_violation_penalty(self):
        """Verify NATO-SC quadratic penalty is applied when budget is violated."""
        env = fheEnvUnified(
            self.rules_list,
            self.test_expressions,
            max_positions=16,
            constraint_metric="mem",
            budget_options=[1],  # Ultra small budget to guarantee violation
            constraint_method="nato_sc",
            verbose=False,
        )
        obs, info = env.reset(seed=42)
        # Force END action
        end_idx = list(self.rules_list.keys()).index("END") * env.max_positions
        obs, reward, terminated, truncated, info = env.step(end_idx)
        self.assertTrue(terminated or truncated)
        # Because budget is 1 MB and peak RAM is > 100 MB, violation penalty must make reward negative
        self.assertLess(reward, 0.0)

    def test_05_policy_film_a(self):
        """Verify CustomFeaturesExtractor (Design A: budget only) shape and forward pass."""
        env = fheEnvUnified(
            self.rules_list,
            self.test_expressions,
            max_positions=16,
            constraint_metric="mem",
            budget_options=[1500, 2000, 2600, 1000000],
            verbose=False,
        )
        extractor = CustomFeaturesExtractor(env.observation_space, budget_encoding="film")
        obs, _ = env.reset(seed=42)
        # Convert to tensor batch of size 2
        batch_obs = {
            k: torch.as_tensor(np.stack([obs[k], obs[k]]), dtype=torch.float32)
            for k in obs
        }
        features = extractor(batch_obs)
        self.assertEqual(features.shape, (2, extractor.features_dim))
        self.assertFalse(torch.isnan(features).any())

    def test_06_policy_film_b(self):
        """Verify CustomFeaturesExtractorFiLMB (Design B: budget + preference fused) shape and forward pass."""
        env = fheEnvUnified(
            self.rules_list,
            self.test_expressions,
            max_positions=16,
            constraint_metric="mem",
            budget_options=[1500, 2000, 2600, 1000000],
            verbose=False,
        )
        extractor = CustomFeaturesExtractorFiLMB(env.observation_space, budget_encoding="film")
        obs, _ = env.reset(seed=42)
        batch_obs = {
            k: torch.as_tensor(np.stack([obs[k], obs[k]]), dtype=torch.float32)
            for k in obs
        }
        features = extractor(batch_obs)
        self.assertEqual(features.shape, (2, extractor.features_dim))
        self.assertFalse(torch.isnan(features).any())

    def test_07_policy_actors(self):
        """Verify HierarchicalMaskablePolicy forward with film_a and film_b."""
        env = fheEnvUnified(
            self.rules_list,
            self.test_expressions,
            max_positions=16,
            constraint_metric="mem",
            budget_options=[1500, 2000, 2600, 1000000],
            verbose=False,
        )
        obs, _ = env.reset(seed=42)
        batch_obs = {
            k: torch.as_tensor(np.stack([obs[k], obs[k]]), dtype=torch.float32)
            for k in obs
        }

        # Policy A
        pol_a = HierarchicalMaskablePolicy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 1e-4,
            budget_encoding="film",
            rule_dim=len(self.rules_list),
            max_positions=16,
        )
        act_a, val_a, logp_a = pol_a(batch_obs)
        self.assertEqual(act_a.shape, (2,))
        self.assertEqual(val_a.squeeze().shape, (2,))

        # Policy B
        pol_b = HierarchicalMaskablePolicyFiLMB(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 1e-4,
            budget_encoding="film",
            rule_dim=len(self.rules_list),
            max_positions=16,
        )
        act_b, val_b, logp_b = pol_b(batch_obs)
        self.assertEqual(act_b.shape, (2,))
        self.assertEqual(val_b.squeeze().shape, (2,))

    def test_08_pid_wrapper(self):
        """Verify PIDLagrangianWrapper with fheEnvUnified."""
        def make_env():
            return fheEnvUnified(
                self.rules_list,
                self.test_expressions,
                max_positions=16,
                constraint_metric="mem",
                budget_options=[1500, 2000, 2600, 1000000],
                verbose=False,
            )
        vec_env = DummyVecEnv([make_env])
        pid_env = PIDLagrangianWrapper(vec_env, lambda_init=0.1)
        self.assertEqual(pid_env.lambda_penalty, 0.1)
        obs = pid_env.reset()
        valid_actions = np.where(obs["action_mask"][0] == 1)[0]
        self.assertGreater(len(valid_actions), 0)
        action = valid_actions[0]
        obs, rewards, dones, infos = pid_env.step(np.array([action]))
        self.assertIn("constraint_value", infos[0])
        self.assertIn("budget", infos[0])

    def test_09_end_to_end_smoke_training(self):
        """Run a 16-step PPO training rollout with both film_a and film_b on unified env."""
        for variant, policy_cls in [("film_a", HierarchicalMaskablePolicy), ("film_b", HierarchicalMaskablePolicyFiLMB)]:
            def make_env():
                return fheEnvUnified(
                    self.rules_list,
                    self.test_expressions,
                    max_positions=16,
                    constraint_metric="mem",
                    budget_options=[1500, 2000],
                    constraint_method="nato_sc",
                    verbose=False,
                )
            vec_env = DummyVecEnv([make_env])
            model = PPO(
                policy=policy_cls,
                env=vec_env,
                n_steps=16,
                batch_size=8,
                n_epochs=1,
                policy_kwargs={
                    "budget_encoding": "film",
                    "rule_dim": len(self.rules_list),
                    "max_positions": 16,
                },
                verbose=0,
            )
            model.learn(total_timesteps=16)
            self.assertGreaterEqual(model.num_timesteps, 16)


if __name__ == "__main__":
    unittest.main()
