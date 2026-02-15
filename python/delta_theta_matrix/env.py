"""
Gymnasium environment wrapping the Rust BacktestEngine.

Provides a standard Gymnasium interface for RL training, with:
- Action space: Box([-1, -1], [1, 1]) for [strike_width_normalized, delta_position_normalized]
- Observation space: Box matching the 20-dimensional state vector
- Reward: theta_captured - (margin_expansion_ratio^3) * penalty_scale
"""

from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from delta_theta_matrix import BacktestEngine, EngineConfig, get_state_dim


class DeltaThetaEnv(gym.Env):
    """Gymnasium environment for delta-theta matrix optimization.

    The agent controls two continuous actions:
    1. strike_width_normalized [-1, 1] → mapped to [2.5, 15.0] dollar spread width
    2. delta_position_normalized [-1, 1] → mapped to [-0.50, -0.05] target short delta

    The reward function balances theta capture against capital efficiency:
        reward = theta_captured - (margin_utilization^3) * penalty_scale + pnl_component

    Episodes terminate on:
    - Margin call (large negative reward)
    - Option expiration
    - Maximum steps reached (truncation)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: EngineConfig | None = None,
        penalty_scale: float = 10.0,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            config: Engine configuration. Uses defaults if None.
            penalty_scale: Multiplier for the margin expansion penalty.
            render_mode: Rendering mode ('human', 'ansi', or None).
        """
        super().__init__()

        self.config = config or EngineConfig()
        self.penalty_scale = penalty_scale
        self.render_mode = render_mode

        # Create the Rust engine
        self.engine = BacktestEngine(self.config)

        # Action space: [strike_width_normalized, delta_position_normalized]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # Observation space: matches state vector dimensionality
        state_dim = get_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float64,
        )

        # Episode tracking
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._last_info: dict[str, Any] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Random seed (unused, seed is set in EngineConfig).
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info_dict).
        """
        super().reset(seed=seed)

        obs_list = self.engine.reset()
        obs = np.array(obs_list, dtype=np.float64)

        self._episode_reward = 0.0
        self._episode_steps = 0

        info = {
            "episode_reward": 0.0,
            "episode_steps": 0,
            "underlying_price": self.engine.underlying_price,
            "tte": self.engine.tte,
            "iv": self.engine.iv,
        }
        self._last_info = info

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Array of [strike_width_normalized, delta_position_normalized],
                    each in [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        strike_width = float(action[0])
        delta_position = float(action[1])

        # Call the Rust engine
        result = self.engine.step(strike_width, delta_position)

        obs = np.array(result.observation, dtype=np.float64)
        reward = float(result.reward)

        self._episode_reward += reward
        self._episode_steps += 1

        # Parse info from JSON
        try:
            engine_info = json.loads(result.info)
        except json.JSONDecodeError:
            engine_info = {}

        info: dict[str, Any] = {
            "episode_reward": self._episode_reward,
            "episode_steps": self._episode_steps,
            "underlying_price": self.engine.underlying_price,
            "tte": self.engine.tte,
            "iv": self.engine.iv,
            "action_strike_width": strike_width,
            "action_delta_position": delta_position,
            **engine_info,
        }
        self._last_info = info

        return obs, reward, result.done, result.truncated, info

    def render(self) -> str | None:
        """Render the current state.

        Returns:
            Text summary if render_mode is 'ansi' or 'human', None otherwise.
        """
        if self.render_mode not in ("human", "ansi"):
            return None

        info = self._last_info
        text = (
            f"═══ Delta-Theta Matrix ═══\n"
            f"  Step:       {info.get('episode_steps', '?')}\n"
            f"  Underlying: ${info.get('underlying_price', 0.0):.2f}\n"
            f"  TTE:        {info.get('tte', 0.0):.4f} yr\n"
            f"  IV:         {info.get('iv', 0.0):.2%}\n"
            f"  Reward:     {info.get('episode_reward', 0.0):.4f}\n"
            f"  Margin:     {info.get('margin_utilization', 0.0):.2%}\n"
            f"  Positions:  {info.get('positions', 0)}\n"
            f"  P&L:        ${info.get('episode_pnl', 0.0):.2f}\n"
            f"══════════════════════════\n"
        )

        if self.render_mode == "human":
            print(text)

        return text

    def close(self) -> None:
        """Clean up resources."""
        pass
