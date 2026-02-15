//! Delta-Theta Matrix Integration — Rust core with PyO3 bindings.
//!
//! Provides high-performance matrix operations for options theta/delta
//! surface analysis, exposed to Python via PyO3.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DeltaThetaMatrix
// ---------------------------------------------------------------------------

/// A 2-D surface indexed by (strike, expiration) holding theta and delta values.
///
/// Used by the RL environment to compute decay, find optimal strikes, and
/// score capital efficiency.
#[pyclass]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeltaThetaMatrix {
    /// Strike prices along the strike axis.
    #[pyo3(get, set)]
    pub strikes: Vec<f64>,

    /// Expiration timestamps (unix epoch seconds) along the expiration axis.
    #[pyo3(get, set)]
    pub expirations: Vec<i64>,

    /// Theta values — shape: [strikes.len()][expirations.len()]
    #[pyo3(get, set)]
    pub theta_surface: Vec<Vec<f64>>,

    /// Delta values — shape: [strikes.len()][expirations.len()]
    #[pyo3(get, set)]
    pub delta_surface: Vec<Vec<f64>>,
}

#[pymethods]
impl DeltaThetaMatrix {
    /// Create a new `DeltaThetaMatrix` with zero-initialised surfaces.
    #[new]
    pub fn new(strikes: Vec<f64>, expirations: Vec<i64>) -> Self {
        let rows = strikes.len();
        let cols = expirations.len();
        Self {
            strikes,
            expirations,
            theta_surface: vec![vec![0.0; cols]; rows],
            delta_surface: vec![vec![0.0; cols]; rows],
        }
    }

    /// Compute the theta surface after applying time-decay over `dt` days.
    ///
    /// Returns a new 2-D Vec with each element decayed by `exp(-theta * dt)`.
    /// This is a simplified model — real decay depends on many more factors.
    pub fn compute_theta_decay(&self, dt: f64) -> Vec<Vec<f64>> {
        self.theta_surface
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&theta| {
                        // Exponential decay model: value * e^(-|theta| * dt)
                        // theta is typically negative for long options
                        theta * (-theta.abs() * dt).exp()
                    })
                    .collect()
            })
            .collect()
    }

    /// Find strikes whose delta is within `tolerance` of `delta_target` for
    /// any expiration.
    ///
    /// Returns a deduplicated, sorted list of matching strike prices.
    pub fn optimal_strikes(&self, delta_target: f64, tolerance: f64) -> Vec<f64> {
        let mut result: Vec<f64> = Vec::new();
        for (i, row) in self.delta_surface.iter().enumerate() {
            for &delta in row {
                if (delta - delta_target).abs() <= tolerance {
                    let strike = self.strikes[i];
                    if !result.iter().any(|&s| (s - strike).abs() < f64::EPSILON) {
                        result.push(strike);
                    }
                }
            }
        }
        result.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Capital efficiency score: ratio of total absolute theta captured to
    /// total absolute delta exposure.
    ///
    /// A higher score means more theta per unit of directional risk.
    /// Returns 0.0 when delta exposure is zero to avoid division by zero.
    pub fn capital_efficiency_score(&self) -> f64 {
        let total_theta: f64 = self
            .theta_surface
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.abs())
            .sum();

        let total_delta: f64 = self
            .delta_surface
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.abs())
            .sum();

        if total_delta < f64::EPSILON {
            0.0
        } else {
            total_theta / total_delta
        }
    }
}

// ---------------------------------------------------------------------------
// BacktestBridge
// ---------------------------------------------------------------------------

/// Placeholder bridge for NautilusTrader back-testing integration.
///
/// Actual integration is done on the Python side via `nautilus_trader`.
/// This struct provides a Rust-side hook for future native back-test logic.
#[pyclass]
#[derive(Clone, Debug)]
pub struct BacktestBridge {
    // TODO: Add fields for backtest state, portfolio, etc.
}

#[pymethods]
impl BacktestBridge {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    /// Run a back-test with the given JSON configuration string.
    ///
    /// Currently returns a stub JSON response. Replace with real logic once
    /// NautilusTrader integration is wired up on the Python side.
    pub fn run_backtest(&self, config: &str) -> PyResult<String> {
        // TODO: Implement real backtest logic via NautilusTrader Python bridge
        let response = serde_json::json!({
            "status": "stub",
            "message": "Backtest not yet implemented — wire up NautilusTrader on the Python side.",
            "config_received": config,
        });
        Ok(response.to_string())
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

/// The `delta_theta` Python extension module.
#[pymodule]
fn delta_theta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DeltaThetaMatrix>()?;
    m.add_class::<BacktestBridge>()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_matrix_zero_initialised() {
        let m = DeltaThetaMatrix::new(vec![100.0, 110.0], vec![1000, 2000]);
        assert_eq!(m.theta_surface.len(), 2);
        assert_eq!(m.theta_surface[0].len(), 2);
        assert!(m.theta_surface[0][0].abs() < f64::EPSILON);
    }

    #[test]
    fn test_capital_efficiency_zero_delta() {
        let m = DeltaThetaMatrix::new(vec![100.0], vec![1000]);
        assert!((m.capital_efficiency_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optimal_strikes_empty_when_no_match() {
        let m = DeltaThetaMatrix::new(vec![100.0, 110.0], vec![1000]);
        let result = m.optimal_strikes(0.5, 0.01);
        assert!(result.is_empty());
    }
}
