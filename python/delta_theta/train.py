"""
Training script for the Delta-Theta RL agent using Stable-Baselines3 PPO.

Usage:
    python -m delta_theta.train [--timesteps N] [--lr LR] [--save-path PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from delta_theta.env import DeltaThetaEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Delta-Theta PPO agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/delta_theta_ppo",
        help="Path to save model checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Delta-Theta PPO Training ===")
    print(f"  timesteps:  {args.timesteps}")
    print(f"  lr:         {args.lr}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  n_envs:     {args.n_envs}")
    print(f"  save_path:  {args.save_path}")
    print()

    # Create vectorised environment
    vec_env = make_vec_env(DeltaThetaEnv, n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps)

    # Save checkpoint
    save_dir = Path(args.save_path)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir))
    print(f"\nModel saved to {save_dir}")


if __name__ == "__main__":
    main()
