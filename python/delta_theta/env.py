"""
Gymnasium-compatible RL environment for Delta-Theta matrix trading.

The environment simulates options theta decay across a strike × expiration
surface and rewards the agent for capturing theta while penalising
inefficient capital usage.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# The Rust extension is imported at runtime after `maturin develop`.
# During type-checking / linting without the built extension, this will
# fail gracefully.
try:
    from delta_theta import DeltaThetaMatrix
except ImportError:  # pragma: no cover
    DeltaThetaMatrix = None  # type: ignore[assignment,misc]


class DeltaThetaEnv(gym.Env):
    """Custom Gymnasium environment for delta-theta surface trading.

    Observation:
        Flattened concatenation of the theta and delta surfaces.

    Action:
        Continuous Box with three dimensions:
        [strike_index (normalised), expiration_index (normalised), position_size]

    Reward:
        ``theta_captured - capital_penalty``
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_strikes: int = 10,
        n_expirations: int = 5,
        dt: float = 1.0,
        capital_penalty_weight: float = 0.1,
        max_steps: int = 252,
    ) -> None:
        super().__init__()

        self.n_strikes = n_strikes
        self.n_expirations = n_expirations
        self.dt = dt
        self.capital_penalty_weight = capital_penalty_weight
        self.max_steps = max_steps

        # Observation: flattened theta + delta surfaces
        obs_size = 2 * n_strikes * n_expirations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: [strike_idx_norm, expiry_idx_norm, position_size]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        self._matrix: DeltaThetaMatrix | None = None
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        strikes = np.linspace(80, 120, self.n_strikes).tolist()
        expirations = list(range(1, self.n_expirations + 1))  # simplified

        assert DeltaThetaMatrix is not None, (
            "Rust extension not available — run `maturin develop` first."
        )
        self._matrix = DeltaThetaMatrix(strikes, expirations)

        # Initialise surfaces with random values (TODO: use real market data)
        rng = self.np_random
        self._matrix.theta_surface = (
            (-rng.random((self.n_strikes, self.n_expirations)) * 0.05).tolist()
        )
        self._matrix.delta_surface = (
            (rng.random((self.n_strikes, self.n_expirations)) * 0.8 + 0.1).tolist()
        )

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._matrix is not None

        # Decode action
        strike_idx = int(np.clip(action[0] * self.n_strikes, 0, self.n_strikes - 1))
        expiry_idx = int(
            np.clip(action[1] * self.n_expirations, 0, self.n_expirations - 1)
        )
        position_size = float(action[2])

        # Theta captured at the chosen cell
        theta_value = self._matrix.theta_surface[strike_idx][expiry_idx]
        theta_captured = -theta_value * position_size  # theta is negative for longs

        # Capital penalty via efficiency score
        capital_penalty = (
            self.capital_penalty_weight
            * (1.0 - min(self._matrix.capital_efficiency_score(), 1.0))
        )

        reward = float(theta_captured - capital_penalty)

        # Advance time: apply theta decay to the surface
        decayed = self._matrix.compute_theta_decay(self.dt)
        self._matrix.theta_surface = decayed

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        assert self._matrix is not None
        theta_flat = [v for row in self._matrix.theta_surface for v in row]
        delta_flat = [v for row in self._matrix.delta_surface for v in row]
        return np.array(theta_flat + delta_flat, dtype=np.float32)
