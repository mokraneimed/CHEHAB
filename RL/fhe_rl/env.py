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

    def __init__(self, rules_list, expressions, max_positions=2,embeddings_model=None, pref_list=[[1.0, 0.0], [0.0, 1.0]], lambda_env=0.1, env_idx=0):
        
        super().__init__()
        self.rules = rules_list
        self.expressions = expressions
        self.max_positions = max_positions
        self.embeddings_model = embeddings_model
        self.max_steps =    75
        self.max_expression_size = 10000
        self.initial_cost = 0


        self.embedding_dim = 256
        self.initial_vectorization_potential = 0
        self.vectorizations_applied = 0
        self.vectorization_helper = 0

        self.pref_list = [np.array(p, dtype=np.float32) for p in pref_list]
        self.current_pref_idx = env_idx % len(self.pref_list)
        self.current_w = self.pref_list[self.current_pref_idx]

        self.lambda_kl = 0.0
        self.lambda_env = lambda_env

        self.pref_locked = False
        self.embeded_expression = None
        self.random_pref = True

        self.action_space = spaces.Discrete(len(self.rules.keys()) * self.max_positions)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(258,), dtype=np.float32
            ),
            "action_mask": spaces.Box(0, 1, (len(self.rules.keys()) * self.max_positions,), np.float32)
        })
        
        # self.reset()

    def set_preference_vector(self, w):
        """Set the preference vector for multi-objective optimization"""
        self.current_w = np.array(w, dtype=np.float32)
        self.pref_locked = True

    def unlock_preferences(self):
        """CHANGE: Method to restore automatic cycling behavior"""
        self.pref_locked = False

    def get_split_costs(self, expr_str):
        parsed = parse_sexpr(expr_str)
        ops = calculate_cost(parsed, w_keys=0.0)
        keys = calculate_cost(parsed, w_keys=1.0) - ops
        return ops, keys    



                    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "current_index"):
            self.current_index = 0
        self.expression = self.expressions[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.expressions)
        self.initial_expression = self.expression
        self.steps = 0

        self.initial_cost = self.current_cost = self.get_cost(self.expression)
        
        self.initial_ops, self.initial_keys = self.get_split_costs(self.expression)
        self.curr_ops, self.curr_keys = self.get_split_costs(self.expression)

        

        if not self.pref_locked:
            if self.random_pref:
                random_idx = self.np_random.integers(0, len(self.pref_list))
                self.current_pref_idx = random_idx
                mode_name = "RANDOM SEARCH"
            else:
                self.current_pref_idx = 0
                mode_name = "FIXED SPEED FOCUS"
            
            self.current_w = self.pref_list[self.current_pref_idx]
            self.random_pref = not self.random_pref           
        else:
            mode_name = "LOCKED (EVALUATION)"

        print(f"\n{BLUE}{'='*40} EPISODE START {'='*40}{RESET}")
        print(f"{BOLD}{MAGENTA}Optimization Mode{RESET}  : {CYAN}{mode_name}{RESET}") 
                   
        self.embeded_expression = self._embed_expression(self.expression)
        return self._get_obs(), {}

    def _get_obs(self):
        if self.embeded_expression is None:
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            emb = self.embeded_expression    
        w = np.atleast_1d(self.current_w).astype(np.float32)
        obs = np.concatenate([emb, w]).astype(np.float32)
        return {
            "observation": obs,
            "action_mask": self.get_action_mask()
        }

    def step(self, action: int):
        if self.current_w[0] == 0.0 and self.current_w[1] == 1.0:
            action = list(self.rules.keys()).index("END") * self.max_positions
        self.steps += 1
        rule_idx = action // self.max_positions
        pos_idx = action % self.max_positions
        rule_name = list(self.rules.keys())[rule_idx]
        terminated = False
        truncated = False
        reward = 0
        print(f"\n{CYAN}{'-'*100}{RESET}")
        print(f"{BOLD}{MAGENTA}Old expression{RESET}: {YELLOW}{self.expression}{RESET}")
        print(f"{BOLD}{MAGENTA}Old ops cost      {RESET}: {RED}{self.curr_ops}{RESET}")
        print(f"{BOLD}{MAGENTA}Old keys cost      {RESET}: {RED}{self.curr_keys}{RESET}")

        if rule_name == "END":
            terminated = True
            truncated = False
            reward = self.calculate_final_reward()
        else:
            parsed = parse_sexpr(self.expression)
            rule_obj = self.rules[rule_name]
            matches = rule_obj.find_matching_subexpressions(parsed)
            k, _ = matches[pos_idx]
            new_expr_tree = rule_obj.apply_rule(parsed, path=k)
            temp = expr_to_str(new_expr_tree)
            self.expression = temp
            new_cost = self.get_cost(self.expression)
 

            new_ops, new_keys = self.get_split_costs(self.expression)
            reward = self.calculate_intermediate_reward(new_ops, new_keys)

            self.curr_ops = new_ops
            self.curr_keys = new_keys

            self.current_cost = new_cost              
            if (self.steps >= self.max_steps):
                terminated = True
                reward = self.calculate_final_reward()
        info = {"expression": self.expression}
        reward_color = GREEN if reward >= 0 else RED
        print(f"{BOLD}{MAGENTA}New expression{RESET}: {YELLOW}{self.expression}{RESET}")
        print(f"{BOLD}{MAGENTA}New ops cost      {RESET}: {RED}{self.curr_ops}{RESET}")
        print(f"{BOLD}{MAGENTA}New keys cost      {RESET}: {RED}{self.curr_keys}{RESET}")
        print(f"{BOLD}{MAGENTA}Reward        {RESET}: {reward_color}{reward}{RESET}")
        print(f"{BOLD}{MAGENTA}Rule name     {RESET}: {CYAN}{rule_name}{RESET}")
        print(f"{BOLD}{MAGENTA}At position   {RESET}: {BLUE}{pos_idx}{RESET}")
        print(f"{CYAN}{'-'*100}{RESET}")
        self.embeded_expression = self._embed_expression(self.expression)
        if self.embeded_expression is None:
            terminated = True
            truncated = True
            reward = self.calculate_final_reward()
        else:
            terminated = terminated or (self.steps >= self.max_steps)
        if terminated or truncated:
            info["episode"] = {
                "r": reward,
                "l": self.steps,
                "t": None
            }
            print(f"\n initial ops cost: {self.initial_ops}")
            print(f"\n initial keys cost: {self.initial_keys}")
            print(f"\n final ops cost: {self.curr_ops}")
            print(f"\n final keys cost: {self.curr_keys}")
        obs = self._get_obs()    
        return obs, reward, terminated, truncated, {"expression": self.expression}
    
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
                    if calculate_cost(new_expr_tree) < self.current_cost:
                        isValid = False
                        break
                    if self.vectorisation_potential(temp) > vectorization_potenial:
                        isValid = False
                        break
            if not isValid:
                break
        return isValid
    def calculate_final_reward(self) -> float:

        if self.initial_ops == 0:
            r_ops = 0.0
        else:
            r_ops = (self.initial_ops - self.curr_ops) / self.initial_ops
        r_keys = (self.initial_keys - self.curr_keys) / 5
        r_vec = np.array([r_ops, r_keys])
        reward_linear = np.dot(self.current_w, r_vec)
        reward_env = max([np.dot(pref, r_vec) for pref in self.pref_list])
        total_reward = reward_linear + self.lambda_env * reward_env
        return total_reward * 100
        
        
    
    def calculate_intermediate_reward(self,new_ops,new_keys) -> float:
        if self.curr_ops == 0:
            r_ops = 0.0
        else:
            r_ops = (self.curr_ops - new_ops) / self.curr_ops
        r_keys = (self.curr_keys - new_keys) / 5
        r_vec = np.array([r_ops, r_keys])
        reward_linear = np.dot(self.current_w, r_vec)
        reward_env = max([np.dot(pref, r_vec) for pref in self.pref_list])
        total_reward = reward_linear + self.lambda_env * reward_env
        return total_reward

    
    def get_cost(self, expr: str) -> float:
        return calculate_cost(parse_sexpr(expr))
    
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
        for rule_idx, rule_name in enumerate(self.rules.keys()):
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
    

