import json
import os
import numpy as np
import fcntl  # file locking

def load_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    cache = {}
    with open(cache_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[entry["key"]] = (
                entry["expression"],
                np.array(entry["embedding"], dtype=np.float32)
            )
    return cache

def update_cache(cache_path: str, cache: dict, key: str, value: tuple):
    cache[key] = value
    entry = json.dumps({
        "key": key,
        "expression": value[0],
        "embedding": value[1].tolist()
    })
    with open(cache_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)  # exclusive lock while writing
        f.write(entry + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)