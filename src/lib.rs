//! Q-Learned Delta-Theta Matrix Integration
//!
//! PyO3 module entry point exposing the Rust backtest engine, state emitter,
//! and configuration types to Python.

use pyo3::prelude::*;

pub mod engine;
pub mod options;
pub mod orderbook;
pub mod risk;
pub mod state;

/// The `delta_theta_matrix` Python module, implemented in Rust.
///
/// Exposes:
/// - `BacktestEngine` — core simulation engine with order book, risk, and options pricing
/// - `EngineConfig` — configuration for the engine
/// - `StepResult` — result of a simulation step
/// - `RiskState` — portfolio risk state
/// - `Greeks` — option Greeks
/// - `PutCreditSpread` — put credit spread structure
/// - `get_state_vector()` — zero-copy NumPy state extraction
/// - `get_state_dim()` — dimensionality of the state vector
#[pymodule]
fn delta_theta_matrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<engine::BacktestEngine>()?;
    m.add_class::<engine::EngineConfig>()?;
    m.add_class::<engine::StepResult>()?;
    m.add_class::<risk::RiskState>()?;
    m.add_class::<options::Greeks>()?;
    m.add_class::<options::PutCreditSpread>()?;
    m.add_class::<options::OptionLeg>()?;
    m.add_function(wrap_pyfunction!(state::get_state_vector, m)?)?;
    m.add_function(wrap_pyfunction!(state::get_state_dim, m)?)?;
    Ok(())
}
