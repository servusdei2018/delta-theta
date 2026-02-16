"""
SAC/PPO training via Stable Baselines3.

CLI script for training reinforcement learning agents on the DeltaThetaEnv.
Supports SAC (default) and PPO algorithms with configurable hyperparameters,
TensorBoard logging, and model checkpointing.

Usage:
    python -m delta_theta_matrix.train --algo sac --timesteps 100000
    python -m delta_theta_matrix.train --algo ppo --timesteps 500000 --learning-rate 3e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from delta_theta_matrix import EngineConfig
from delta_theta_matrix.env import DeltaThetaEnv


class ThetaMetricsCallback(BaseCallback):
    """Custom callback that logs theta-specific metrics to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_pnls: list[float] = []

    def _on_step(self) -> bool:
        """Log custom metrics on each step."""
        # Extract info from the environment
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict):
                if "episode_pnl" in info:
                    self.logger.record("theta/episode_pnl", info["episode_pnl"])
                if "margin_utilization" in info:
                    self.logger.record(
                        "theta/margin_utilization", info["margin_utilization"]
                    )
                if "theta_exposure" in info:
                    self.logger.record("theta/theta_exposure", info["theta_exposure"])
                if "iv" in info:
                    self.logger.record("theta/implied_vol", info["iv"])

        return True


def create_env(
    config: EngineConfig | None = None,
    penalty_scale: float = 10.0,
    log_dir: str | None = None,
) -> Monitor:
    """Create a monitored training environment.

    Args:
        config: Engine configuration.
        penalty_scale: Margin penalty scale.
        log_dir: Directory for Monitor logs.

    Returns:
        Wrapped environment with monitoring.
    """
    env = DeltaThetaEnv(config=config, penalty_scale=penalty_scale)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, "monitor"))
    else:
        env = Monitor(env)
    return env


def train(
    algo: str = "sac",
    timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    gamma: float = 0.99,
    output_dir: str = "output",
    seed: int = 42,
    penalty_scale: float = 10.0,
    max_episode_steps: int = 1000,
    eval_freq: int = 5000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
) -> Path:
    """Train an RL agent on the DeltaThetaEnv.

    Args:
        algo: Algorithm to use ('sac' or 'ppo').
        timesteps: Total training timesteps.
        learning_rate: Learning rate.
        batch_size: Batch size for training.
        gamma: Discount factor.
        output_dir: Directory for outputs (models, logs, etc.).
        seed: Random seed.
        penalty_scale: Margin penalty scale for the environment.
        max_episode_steps: Maximum steps per episode.
        eval_freq: Evaluation frequency in timesteps.
        n_eval_episodes: Number of evaluation episodes.
        verbose: Verbosity level.

    Returns:
        Path to the saved model.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_{timestamp}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = str(run_dir / "logs")
    tb_dir = str(run_dir / "tensorboard")

    # Create engine config
    config = EngineConfig(
        seed=seed,
        max_episode_steps=max_episode_steps,
    )

    # Create training environment
    train_env = create_env(config=config, penalty_scale=penalty_scale, log_dir=log_dir)

    # Create evaluation environment
    eval_config = EngineConfig(
        seed=seed + 1000,
        max_episode_steps=max_episode_steps,
    )
    eval_env = create_env(config=eval_config, penalty_scale=penalty_scale)

    # Set up callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(timesteps // 10, 1000),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=algo,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    theta_cb = ThetaMetricsCallback(verbose=verbose)

    callbacks = [checkpoint_cb, eval_cb, theta_cb]

    # Create the model
    algo_lower = algo.lower()
    if algo_lower == "sac":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            verbose=verbose,
            tensorboard_log=tb_dir,
            seed=seed,
        )
    elif algo_lower == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            verbose=verbose,
            tensorboard_log=tb_dir,
            seed=seed,
            n_steps=2048,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}. Use 'sac' or 'ppo'.")

    # Save training config
    train_config = {
        "algo": algo,
        "timesteps": timesteps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "seed": seed,
        "penalty_scale": penalty_scale,
        "max_episode_steps": max_episode_steps,
        "run_name": run_name,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    if verbose:
        print(f"Training {algo.upper()} for {timesteps} timesteps...")
        print(f"Output directory: {run_dir}")

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    model_path = run_dir / "final_model"
    model.save(str(model_path))

    if verbose:
        print(f"Model saved to {model_path}")

    # Clean up
    train_env.close()
    eval_env.close()

    return model_path


def evaluate(
    model_path: str,
    algo: str = "sac",
    n_episodes: int = 50,
    seed: int = 99,
    output_path: str | None = None,
    max_episode_steps: int = 1000,
) -> list[dict[str, Any]]:
    """Run evaluation episodes and collect data for visualization.

    Args:
        model_path: Path to the saved model.
        algo: Algorithm used ('sac' or 'ppo').
        n_episodes: Number of evaluation episodes.
        seed: Random seed.
        output_path: Path to save evaluation data as JSON.
        max_episode_steps: Maximum steps per episode.

    Returns:
        List of episode data dictionaries.
    """
    config = EngineConfig(seed=seed, max_episode_steps=max_episode_steps)
    env = DeltaThetaEnv(config=config)

    # Load model
    algo_lower = algo.lower()
    if algo_lower == "sac":
        model = SAC.load(model_path)
    elif algo_lower == "ppo":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    episodes_data: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_record: dict[str, Any] = {
            "episode": ep,
            "steps": [],
            "total_reward": 0.0,
            "final_pnl": 0.0,
        }

        done = False
        truncated = False
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            step_data = {
                "tte": info.get("tte", 0.0),
                "iv": info.get("iv", 0.0),
                "strike_width": info.get("action_strike_width", 0.0),
                "delta_position": info.get("action_delta_position", 0.0),
                "reward": reward,
                "underlying_price": info.get("underlying_price", 0.0),
                "margin_utilization": info.get("margin_utilization", 0.0),
                "pnl": info.get("episode_pnl", 0.0),
            }
            episode_record["steps"].append(step_data)
            episode_record["total_reward"] += reward

        episode_record["final_pnl"] = info.get("episode_pnl", 0.0)
        episodes_data.append(episode_record)

    env.close()

    # Save evaluation data
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(episodes_data, f, indent=2)
        print(f"Evaluation data saved to {output_file}")

    return episodes_data


def main() -> None:
    """CLI entry point for training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train a delta-theta matrix RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="RL algorithm to use",
    )
    train_parser.add_argument(
        "--timesteps", type=int, default=100_000, help="Total training timesteps"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    train_parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    train_parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    train_parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--penalty-scale",
        type=float,
        default=10.0,
        help="Margin expansion penalty scale",
    )
    train_parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    train_parser.add_argument(
        "--eval-freq", type=int, default=5000, help="Evaluation frequency"
    )
    train_parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained agent")
    eval_parser.add_argument("model_path", type=str, help="Path to the saved model")
    eval_parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm used for training",
    )
    eval_parser.add_argument(
        "--n-episodes", type=int, default=50, help="Number of evaluation episodes"
    )
    eval_parser.add_argument("--seed", type=int, default=99, help="Random seed")
    eval_parser.add_argument(
        "--output", type=str, default="output/eval_data.json", help="Output file"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            algo=args.algo,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            output_dir=args.output_dir,
            seed=args.seed,
            penalty_scale=args.penalty_scale,
            max_episode_steps=args.max_episode_steps,
            eval_freq=args.eval_freq,
            verbose=args.verbose,
        )
    elif args.command == "evaluate":
        evaluate(
            model_path=args.model_path,
            algo=args.algo,
            n_episodes=args.n_episodes,
            seed=args.seed,
            output_path=args.output,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
