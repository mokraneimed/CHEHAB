import sys
import importlib

# Ensure fhe_rl_new resolves to fhe_rl for pickle/deserialization compatibility
if "fhe_rl_new" not in sys.modules:
    sys.modules["fhe_rl_new"] = sys.modules[__name__]
