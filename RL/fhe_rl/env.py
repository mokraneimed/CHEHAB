import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pytrs import parse_sexpr, calculate_cost, Expr, Const, Var, Op,expr_to_str
import torch
from .config import get_tokenizer_type

if get_tokenizer_type() == "bpe":
    
    from .TRAE_bpe import get_expression_cls_embedding
    
else:
    
    from .TRAE import get_expression_cls_embedding



RESET   = "\033[0m"

BOLD    = "\033[1m"
DIM     = "\033[2m"

RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"


class fheEnv(gym.Env):

    def __init__(self, rules_list, expressions, max_positions=2,embeddings_model=None, keys_weight_schedule=None, use_curriculum=True, phase_b_rule_prefix="rotate_", inference_mode=False, auto_transition = False, max_keys_weight = 1.0):
        
        super().__init__()
        self.rules = rules_list
        self.expressions = expressions
        self.max_positions = max_positions
        self.embeddings_model = embeddings_model
        self.max_steps =    75
        self.max_expression_size = 10000
        self.initial_cost = 0

        self.initial_cost_alt = 0

        self.embedding_dim = 256
        self.initial_vectorization_potential = 0
        self.vectorizations_applied = 0
        self.vectorization_helper = 0

        self.keys_weight_schedule = keys_weight_schedule  
        self.current_keys_weight = 0.0
        self.current_global_timestep = 0
        self.keys_cost_triggered = False


        self.use_curriculum = use_curriculum
        self.phase_b_rule_prefix = phase_b_rule_prefix
        self.current_phase = "phase_a" if use_curriculum else "phase_b"

        self.auto_transition = auto_transition
        self.max_keys_weight = max_keys_weight

        if inference_mode:
            self.use_curriculum = False
            self.current_keys_weight = self.max_keys_weight

        self._categorize_rules()

        self.alternate_rewards_buffer = []  # Stores (step_idx, alt_reward) tuples

        self.action_space = spaces.Discrete(len(self.rules.keys()) * self.max_positions)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32
            ),
            "action_mask": spaces.Box(0, 1, (len(self.rules.keys()) * self.max_positions,), np.float32)
        })
        self.reset()

    def get_and_clear_alternate_rewards(self):
        """Retrieve stored alternate rewards and clear buffer"""
        rewards = np.array(self.alternate_rewards_buffer, dtype=np.float32)
        self.alternate_rewards_buffer.clear()
        return rewards

    def _categorize_rules(self):
        self.phase_a_rules = []
        self.phase_b_rules = []

        for rule_name in self.rules.keys():
            if rule_name == "END":
                self.phase_a_rules.append(rule_name)
                self.phase_b_rules.append(rule_name)
            elif rule_name.startswith(self.phase_b_rule_prefix):
                self.phase_b_rules.append(rule_name)
            else:
                self.phase_a_rules.append(rule_name)
                self.phase_b_rules.append(rule_name)

        if self.use_curriculum:
            print(f"\n{'='*80}")
            print(f"CURRICULUM LEARNING ENABLED")
            print(f"{'='*80}")
            print(f"Phase A rules (objective A): {len(self.phase_a_rules)} rules")
            print(f"Phase B rules (objective A+B): {len(self.phase_b_rules)} rules")
            print(f"Phase B exclusive rules: {len(self.phase_b_rules) - len(self.phase_a_rules)} rules")
            print(f"Exclusive rules: {[r for r in self.phase_b_rules if r not in self.phase_a_rules]}")
            print(f"{'='*80}\n")                        

    def _get_available_rules(self):
        if not self.use_curriculum:
            return list(self.rules.keys())
        if self.current_phase == "phase_a":
            return self.phase_a_rules
        else:   
            return self.phase_b_rules

    def set_curriculum_phase(self, phase: str):
        if phase not in ["phase_a", "phase_b"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'phase_a' or 'phase_b'.")

        old_phase = self.current_phase
        self.current_phase = phase

        old_count = len(self._get_available_rules_for_phase(old_phase))
        new_count = len(self._get_available_rules_for_phase(phase))
        
        print(f"\n{'='*80}")
        print(f"CURRICULUM PHASE CHANGE: {old_phase} → {phase}")
        print(f"{'='*80}")
        print(f"Available rules: {old_count} → {new_count}")
        if phase == "phase_b":
            newly_available = [r for r in self.phase_b_rules if r not in self.phase_a_rules]
            print(f"Newly available rules ({len(newly_available)}): {newly_available[:5]}{'...' if len(newly_available) > 5 else ''}")
        print(f"{'='*80}\n")

    def _get_available_rules_for_phase(self, phase):
        if phase == "phase_a":
            return self.phase_a_rules
        else:
            return self.phase_b_rules


    def set_timestep(self, timestep: int):
        """
        Called externally to update the global timestep.
        This is called by the TimestepUpdater callback.
        """
        self.current_global_timestep = timestep
        
        # Update keys weight based on new timestep
        if not self.auto_transition:
            if self.keys_weight_schedule is not None:
                self.current_keys_weight = self.keys_weight_schedule(timestep)
            else: 
                print("Keys weight schedule is wrong")
        elif self.keys_cost_triggered:
            self.current_keys_weight = self.max_keys_weight
                

    def trigger_keys_cost(self):
        self.keys_cost_triggered = True

        if self.use_curriculum:
            self.set_curriculum_phase("phase_b")        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "current_index"):
            self.current_index = 0
        self.expression = self.expressions[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.expressions)
        self.initial_expression = self.expression
        self.steps = 0

        self.initial_cost = self.current_cost = self.get_cost(self.expression)
        self.initial_cost_alt = self.current_cost_alt = calculate_cost(parse_sexpr(self.expression), w_keys=self.max_keys_weight)
        return {
            "observation": self._embed_expression(self.expression),
            "action_mask": self.get_action_mask()
        }, {}

    
    def step(self, action: int):
        self.steps += 1
        rule_idx = action // self.max_positions
        pos_idx = action % self.max_positions
        rule_name = list(self.rules.keys())[rule_idx]
        terminated = False
        truncated = False
        reward = 0
        print(f"\n{CYAN}{'-'*100}{RESET}")
        print(f"{BOLD}{MAGENTA}Old expression{RESET}: {YELLOW}{self.expression}{RESET}")
        print(f"{BOLD}{MAGENTA}Old cost      {RESET}: {RED}{self.current_cost}{RESET}")

        if rule_name == "END":
            terminated = True
            truncated = False
            reward = self.calculate_final_reward()
            if not self.keys_cost_triggered:
                reward_alt = self.calculate_final_reward_alt()
                self.alternate_rewards_buffer.append(reward_alt)
        else:
            parsed = parse_sexpr(self.expression)
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            k, _ = matches[pos_idx]
            new_expr_tree = rule_obj.apply_rule(parsed, path=k)
            temp = expr_to_str(new_expr_tree)
            self.expression = temp
            new_cost = self.get_cost(self.expression)
            new_cost_alt = calculate_cost(parse_sexpr(self.expression), w_keys=self.max_keys_weight)
            reward = self.calculate_intermediate_reward(new_cost)
            if not self.keys_cost_triggered:
                reward_alt = self.calculate_intermediate_reward_alt(new_cost_alt)
                self.alternate_rewards_buffer.append(reward_alt)
            self.current_cost = new_cost
            self.current_cost_alt = new_cost_alt               
            if (self.steps >= self.max_steps):
                terminated = True
                reward = self.calculate_final_reward()
                if not self.keys_cost_triggered:
                    reward_alt = self.calculate_final_reward_alt()
                    self.alternate_rewards_buffer.append(reward_alt)
        info = {"expression": self.expression}
        reward_color = GREEN if reward >= 0 else RED
        print(f"{BOLD}{MAGENTA}New expression{RESET}: {YELLOW}{self.expression}{RESET}")
        print(f"{BOLD}{MAGENTA}New cost      {RESET}: {RED}{self.current_cost}{RESET}")
        print(f"{BOLD}{MAGENTA}Reward        {RESET}: {reward_color}{reward}{RESET}")
        print(f"{BOLD}{MAGENTA}Rule name     {RESET}: {CYAN}{rule_name}{RESET}")
        print(f"{BOLD}{MAGENTA}At position   {RESET}: {BLUE}{pos_idx}{RESET}")
        if self.current_keys_weight > 0:
            print(f"{BOLD}{MAGENTA}Keys weight   {RESET}: {YELLOW}{self.current_keys_weight:.6f}{RESET}")
        print(f"{CYAN}{'-'*100}{RESET}")
        embedding = self._embed_expression(self.expression)
        if embedding is None:
            terminated = True
            truncated = True
            reward = self.calculate_final_reward()
            if not self.keys_cost_triggered:
                reward_alt = self.calculate_final_reward_alt()
                self.alternate_rewards_buffer.append(reward_alt)
        else:
            terminated = terminated or (self.steps >= self.max_steps)
        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "t": None
            }
        return {
            "observation": embedding,
            "action_mask": self.get_action_mask()
        }, reward, terminated, truncated, {"expression": self.expression}
    
    def _valid_end_action(self,expr: str) -> bool:
        expr_tree = parse_sexpr(expr)
        vectorization_potenial = self.vectorisation_potential(expr)
        action_mask = self.get_action_mask()
        isValid = True
        for i, rule_name in enumerate(self.rules.keys()):
            if rule_name == "END":
                continue
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(expr_tree)
            if len(matches) > 0:
                for i,match in enumerate( matches):
                    if i >= self.max_positions:
                        break
                    k, _ = match
                    new_expr_tree = rule_obj.apply_rule(expr_tree, path=k)
                    temp = expr_to_str(new_expr_tree)
                    if calculate_cost(new_expr_tree, w_keys=self.current_keys_weight) < self.current_cost:
                        isValid = False
                        break
                    if self.vectorisation_potential(temp) > vectorization_potenial:
                        isValid = False
                        break
            if not isValid:
                break
        return isValid
    
    def calculate_final_reward(self) -> float:
        if self.initial_cost == 0:
            return 0.0
        return (self.initial_cost - self.current_cost) / self.initial_cost * 100
    
    def calculate_final_reward_alt(self) -> float:
        if self.initial_cost_alt == 0:
            return 0.0
        return (self.initial_cost_alt - self.current_cost_alt) / self.initial_cost_alt * 100
    
    def calculate_intermediate_reward(self,new_cost) -> float:
        if self.current_cost == 0:
            return 0.0
        return ( ( self.current_cost - new_cost) / self.current_cost )
    
    def calculate_intermediate_reward_alt(self, new_cost_alt) -> float:
        """Calculate reward with Phase 2 keys_weight"""
        if self.current_cost_alt == 0:
            return 0.0
        # Use fixed 0.01 for Phase 2
        return (self.current_cost_alt - new_cost_alt) / self.current_cost_alt
    
    def get_cost(self, expr: str) -> float:
        return calculate_cost(parse_sexpr(expr), w_keys=self.current_keys_weight)
    
    def _embed_expression(self, expr: str) -> np.ndarray:
        expr_tree = parse_sexpr(expr)
        with torch.no_grad():
            emb = get_expression_cls_embedding(expr_tree, self.embeddings_model)
        if emb is None:
            return None
        return emb.squeeze(0).cpu().numpy().astype(np.float32)
    
    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.rules.keys()) * self.max_positions, dtype=np.float32)
        parsed = parse_sexpr(self.expression)

        available_rules = self._get_available_rules()

        for rule_idx, rule_name in enumerate(self.rules.keys()):

            if rule_name not in available_rules:
                continue

            if rule_name == "END":
                mask[rule_idx * self.max_positions] = 1.0
                continue
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            valid_positions = min(len(matches), self.max_positions)
            if valid_positions > 0:
                start = rule_idx * self.max_positions
                mask[start:start + valid_positions] = 1.0
        return mask
    

