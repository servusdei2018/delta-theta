"""
3D Plotly visualization for delta-theta matrix analysis.

Generates:
- 3D surface plot: X=time-to-expiry, Y=implied vol, Z=agent's chosen strike width
- Training reward curve
- Saves as interactive HTML files

Usage:
    python -m delta_theta_matrix.visualize --data output/eval_data.json --output output/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_evaluation_data(data_path: str) -> list[dict[str, Any]]:
    """Load evaluation episode data from JSON.

    Args:
        data_path: Path to the evaluation data JSON file.

    Returns:
        List of episode data dictionaries.
    """
    with open(data_path) as f:
        return json.load(f)


def create_3d_surface(
    episodes_data: list[dict[str, Any]],
    output_path: str | None = None,
) -> go.Figure:
    """Create a 3D Plotly surface: X=time-to-expiry, Y=IV, Z=strike width.

    Aggregates step-level data across all episodes to build a surface showing
    how the agent adjusts strike width based on time-to-expiry and implied vol.

    Args:
        episodes_data: List of episode data from evaluation.
        output_path: Optional path to save the HTML file.

    Returns:
        Plotly Figure object.
    """
    # Collect all step data
    tte_vals: list[float] = []
    iv_vals: list[float] = []
    width_vals: list[float] = []

    for episode in episodes_data:
        for step in episode.get("steps", []):
            tte = step.get("tte", 0.0)
            iv = step.get("iv", 0.0)
            # Denormalize strike width: [-1, 1] -> [2.5, 15.0]
            raw_width = step.get("strike_width", 0.0)
            width = 2.5 + (raw_width + 1.0) / 2.0 * 12.5

            tte_vals.append(tte)
            iv_vals.append(iv)
            width_vals.append(width)

    if not tte_vals:
        print("No step data found in evaluation data.")
        return go.Figure()

    tte_arr = np.array(tte_vals)
    iv_arr = np.array(iv_vals)
    width_arr = np.array(width_vals)

    # Create grid for surface interpolation
    tte_bins = np.linspace(tte_arr.min(), tte_arr.max(), 30)
    iv_bins = np.linspace(iv_arr.min(), iv_arr.max(), 30)

    # Bin the data and compute mean strike width per bin
    z_grid = np.full((len(iv_bins) - 1, len(tte_bins) - 1), np.nan)

    for i in range(len(iv_bins) - 1):
        for j in range(len(tte_bins) - 1):
            mask = (
                (iv_arr >= iv_bins[i])
                & (iv_arr < iv_bins[i + 1])
                & (tte_arr >= tte_bins[j])
                & (tte_arr < tte_bins[j + 1])
            )
            if mask.sum() > 0:
                z_grid[i, j] = width_arr[mask].mean()

    # Interpolate NaN values for smoother surface

    # Simple nearest-neighbor fill for NaN values
    mask = np.isnan(z_grid)
    if mask.any() and not mask.all():
        z_filled = z_grid.copy()
        # Use a simple fill: replace NaN with column/row mean
        col_means = np.nanmean(z_grid, axis=0)
        for i in range(z_grid.shape[0]):
            for j in range(z_grid.shape[1]):
                if np.isnan(z_filled[i, j]):
                    z_filled[i, j] = (
                        col_means[j]
                        if not np.isnan(col_means[j])
                        else np.nanmean(z_grid)
                    )
        z_grid = z_filled

    # Create bin centers for axis labels
    tte_centers = (tte_bins[:-1] + tte_bins[1:]) / 2
    iv_centers = (iv_bins[:-1] + iv_bins[1:]) / 2

    fig = go.Figure(
        data=[
            go.Surface(
                x=tte_centers,
                y=iv_centers,
                z=z_grid,
                colorscale="Viridis",
                colorbar=dict(title="Strike Width ($)"),
                hovertemplate=(
                    "TTE: %{x:.4f} yr<br>"
                    "IV: %{y:.2%}<br>"
                    "Strike Width: $%{z:.1f}<br>"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text="Delta-Theta Matrix: Agent Strike Width Selection",
            font=dict(size=18),
        ),
        scene=dict(
            xaxis_title="Time to Expiry (years)",
            yaxis_title="Implied Volatility",
            zaxis_title="Strike Width ($)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        width=1000,
        height=700,
        template="plotly_dark",
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"3D surface saved to {output_path}")

    return fig


def create_reward_curve(
    episodes_data: list[dict[str, Any]],
    output_path: str | None = None,
) -> go.Figure:
    """Create a training reward curve plot.

    Args:
        episodes_data: List of episode data from evaluation.
        output_path: Optional path to save the HTML file.

    Returns:
        Plotly Figure object.
    """
    episode_rewards = [ep.get("total_reward", 0.0) for ep in episodes_data]
    episode_pnls = [ep.get("final_pnl", 0.0) for ep in episodes_data]
    episode_nums = list(range(len(episodes_data)))

    # Compute rolling average
    window = min(10, len(episode_rewards))
    if window > 0:
        rolling_avg = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        ).tolist()
        rolling_x = list(range(window - 1, len(episode_rewards)))
    else:
        rolling_avg = episode_rewards
        rolling_x = episode_nums

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Episode Rewards", "Episode P&L"),
        vertical_spacing=0.12,
    )

    # Reward scatter
    fig.add_trace(
        go.Scatter(
            x=episode_nums,
            y=episode_rewards,
            mode="markers",
            name="Episode Reward",
            marker=dict(size=5, opacity=0.5, color="#636EFA"),
        ),
        row=1,
        col=1,
    )

    # Rolling average
    fig.add_trace(
        go.Scatter(
            x=rolling_x,
            y=rolling_avg,
            mode="lines",
            name=f"Rolling Avg ({window})",
            line=dict(color="#EF553B", width=2),
        ),
        row=1,
        col=1,
    )

    # P&L bar chart
    colors = ["#00CC96" if p >= 0 else "#EF553B" for p in episode_pnls]
    fig.add_trace(
        go.Bar(
            x=episode_nums,
            y=episode_pnls,
            name="Episode P&L",
            marker_color=colors,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text="Delta-Theta Matrix: Training Performance",
            font=dict(size=18),
        ),
        height=800,
        width=1000,
        template="plotly_dark",
        showlegend=True,
    )

    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        print(f"Reward curve saved to {output_path}")

    return fig


def create_all_visualizations(
    data_path: str,
    output_dir: str = "output/plots",
) -> None:
    """Generate all visualization plots from evaluation data.

    Args:
        data_path: Path to evaluation data JSON.
        output_dir: Directory to save HTML plots.
    """
    episodes_data = load_evaluation_data(data_path)
    print(f"Loaded {len(episodes_data)} episodes from {data_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    create_3d_surface(episodes_data, str(output / "strike_width_surface.html"))
    create_reward_curve(episodes_data, str(output / "reward_curve.html"))

    print(f"\nAll visualizations saved to {output_dir}/")


def main() -> None:
    """CLI entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Generate delta-theta matrix visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation data JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/plots",
        help="Output directory for HTML plots",
    )

    args = parser.parse_args()
    create_all_visualizations(args.data, args.output)


if __name__ == "__main__":
    main()
