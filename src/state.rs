//! State-space emitter — zero-copy NumPy array output via `#[pyfunction]`.
//!
//! Provides efficient state vector extraction from the BacktestEngine,
//! using the numpy crate for zero-copy transfer to Python.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::engine::BacktestEngine;

/// Observation vector dimensionality.
pub const STATE_DIM: usize = 20;

/// Extract the current state vector from the engine as a NumPy array.
///
/// The state vector contains:
/// - [0] Underlying mid price (normalized by /100)
/// - [1] ATM bid-ask spread
/// - [2..7] OTM put deltas at various strikes
/// - [7..12] Implied vol surface points
/// - [12] Time-to-expiry
/// - [13] Current implied volatility
/// - [14] Margin utilization ratio
/// - [15] Theta exposure
/// - [16] Buying power (normalized)
/// - [17] Episode progress
/// - [18] Number of positions
/// - [19] Episode P&L (normalized)
///
/// Uses zero-copy transfer via the numpy crate for maximum performance.
#[pyfunction]
pub fn get_state_vector<'py>(
    py: Python<'py>,
    engine: &BacktestEngine,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let obs = engine.get_observation();

    // Create a NumPy array from the observation vector
    let array = PyArray1::from_vec(py, obs);

    Ok(array)
}

/// Get the dimensionality of the state vector.
#[pyfunction]
pub fn get_state_dim() -> usize {
    STATE_DIM
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_dim() {
        assert_eq!(get_state_dim(), 20);
    }
}
