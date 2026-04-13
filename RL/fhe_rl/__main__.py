
import sys
import os
import argparse

from .experiments.run_ops_exp import run_ops_experiment
from .agents.sequential.agent2_key.train import train_agent

from .shared.config import get_model_path, get_tokenizer_type
from .shared.utils import load_embeddings

def usage() -> None:
    print(
        "Usage:\n"
        "  python -m fhe_rl train [--tokenizer_type {dynamic,bpe}]\n"
        "  python -m fhe_rl test  [--tokenizer_type {dynamic,bpe}]\n"
        "  python -m fhe_rl run   [--tokenizer_type {dynamic,bpe}] "
        "<input_expr_file> <output_vector_file>\n"
        "  python -m fhe_rl --show_config  # Show current configuration\n"
        "\n"
        "Options:\n"
        "  --tokenizer_type {dynamic,bpe}  Choose tokenizer type (overrides config)\n"
        "  --show_config                   Show current configuration\n"
        "\n"
        "All model paths are loaded from config.py."
    )
    sys.exit(1)


def load_embeddings_from_config(tokenizer_type=None):
    """Load embeddings using the configuration system"""
    try:
        # Determine the correct embeddings model based on tokenizer type
        if tokenizer_type == "bpe" or (tokenizer_type is None and get_tokenizer_type() == "bpe"):
            embeddings_path = get_model_path("bpe_embeddings_model")
        else:
            embeddings_path = get_model_path("dynamic_embeddings_model")
        
        return load_embeddings(tokenizer_type=tokenizer_type, checkpoint_path=embeddings_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main(args=None):
    embeddings, tokenizer = load_embeddings_from_config("dynamic")
    train_agent(
        expressions_file="./fhe_rl/shared/datasets/benchmarks.txt",
        embeddings_model=embeddings,
        total_timesteps=10_000,
        num_envs=1,
        seed=42
    )


if __name__ == "__main__":
    main()