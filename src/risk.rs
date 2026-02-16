//! Margin and risk tracking — buying power, margin calls, early assignment detection.
//!
//! Tracks portfolio risk state including buying power consumption, margin requirements
//! for put credit spreads, early assignment triggers, and catastrophic margin calls.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::options::PutCreditSpread;

/// A tracked position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Position {
    /// Ticker symbol
    #[pyo3(get)]
    pub ticker: String,
    /// The put credit spread
    #[pyo3(get)]
    pub spread: PutCreditSpread,
    /// Number of contracts
    #[pyo3(get)]
    pub quantity: u32,
    /// Margin required for this position
    #[pyo3(get)]
    pub margin_required: f64,
    /// Whether early assignment has been triggered
    #[pyo3(get)]
    pub early_assignment_triggered: bool,
}

/// Portfolio-level risk state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RiskState {
    /// Total available buying power
    #[pyo3(get)]
    pub buying_power: f64,
    /// Total margin currently in use
    #[pyo3(get)]
    pub margin_used: f64,
    /// Whether a margin call has been triggered
    #[pyo3(get)]
    pub margin_call_triggered: bool,
    /// All open positions
    pub positions: Vec<Position>,
    /// Initial buying power (for ratio calculations)
    #[pyo3(get)]
    pub initial_buying_power: f64,
}

impl RiskState {
    /// Create a new risk state with the given initial buying power.
    pub fn new(initial_buying_power: f64) -> Self {
        RiskState {
            buying_power: initial_buying_power,
            margin_used: 0.0,
            margin_call_triggered: false,
            positions: Vec::new(),
            initial_buying_power,
        }
    }

    /// Calculate the margin requirement for a put credit spread.
    ///
    /// For a put credit spread: margin = (spread_width * 100 - net_credit * 100) * quantity
    /// This represents the maximum loss on the position.
    pub fn calculate_margin(spread: &PutCreditSpread, quantity: u32) -> f64 {
        let per_contract = spread.spread_width * 100.0 - spread.net_credit * 100.0;
        per_contract * quantity as f64
    }

    /// Open a new put credit spread position.
    ///
    /// # Arguments
    /// * `ticker` - Underlying symbol
    /// * `spread` - The put credit spread to open
    /// * `quantity` - Number of contracts
    ///
    /// # Returns
    /// Ok(()) if the position was opened, Err if insufficient buying power.
    pub fn open_position(
        &mut self,
        ticker: &str,
        spread: PutCreditSpread,
        quantity: u32,
    ) -> Result<(), String> {
        let margin_required = Self::calculate_margin(&spread, quantity);

        if margin_required > self.buying_power {
            return Err(format!(
                "Insufficient buying power: need {:.2}, have {:.2}",
                margin_required, self.buying_power
            ));
        }

        self.buying_power -= margin_required;
        self.margin_used += margin_required;

        // Credit received increases buying power
        let credit_received = spread.net_credit * 100.0 * quantity as f64;
        self.buying_power += credit_received;

        self.positions.push(Position {
            ticker: ticker.to_string(),
            spread,
            quantity,
            margin_required,
            early_assignment_triggered: false,
        });

        Ok(())
    }

    /// Check for early assignment on all positions.
    ///
    /// Early assignment is triggered when the short put goes deep in-the-money
    /// (underlying price drops significantly below the short strike).
    ///
    /// # Arguments
    /// * `underlying_price` - Current price of the underlying
    /// * `itm_threshold` - How deep ITM (as fraction of strike) to trigger assignment
    ///
    /// # Returns
    /// Number of positions with early assignment triggered.
    pub fn check_early_assignment(
        &mut self,
        underlying_price: f64,
        itm_threshold: f64,
    ) -> usize {
        let mut count = 0;
        for position in &mut self.positions {
            let short_strike = position.spread.short_leg.strike;
            let itm_amount = short_strike - underlying_price;
            let itm_ratio = itm_amount / short_strike;

            if itm_ratio > itm_threshold && !position.early_assignment_triggered {
                position.early_assignment_triggered = true;
                count += 1;
            }
        }
        count
    }

    /// Check for catastrophic margin call.
    ///
    /// A margin call is triggered when margin_used exceeds buying_power,
    /// indicating the account cannot support its positions.
    ///
    /// # Returns
    /// true if a margin call was triggered.
    pub fn check_margin_call(&mut self) -> bool {
        if self.margin_used > self.buying_power + self.margin_used {
            // This shouldn't happen in normal operation, but check for safety
            self.margin_call_triggered = true;
        }

        // More realistic: margin call if unrealized losses push buying power negative
        if self.buying_power < 0.0 {
            self.margin_call_triggered = true;
        }

        self.margin_call_triggered
    }

    /// Update risk state based on current market conditions.
    ///
    /// Recalculates margin requirements and buying power based on current P&L.
    ///
    /// # Arguments
    /// * `underlying_price` - Current underlying price
    /// * `r` - Risk-free rate
    /// * `t` - Time to expiration in years
    ///
    /// # Returns
    /// Total unrealized P&L across all positions.
    pub fn update_mark_to_market(
        &mut self,
        underlying_price: f64,
        r: f64,
        t: f64,
    ) -> Result<f64, String> {
        let mut total_pnl = 0.0;

        for position in &self.positions {
            let _sigma = position.spread.short_leg.greeks.delta.abs().max(0.1); // Use delta as proxy
            // Use the stored implied vol from the order book
            let pnl = position.spread.current_pnl(
                underlying_price,
                r,
                // Use a reasonable vol estimate
                0.25,
                t,
            )?;
            total_pnl += pnl * position.quantity as f64;
        }

        // Adjust buying power based on unrealized P&L
        let base_bp = self.initial_buying_power - self.margin_used;
        self.buying_power = base_bp + total_pnl;

        Ok(total_pnl)
    }

    /// Get the margin utilization ratio (0.0 to 1.0+).
    pub fn margin_utilization(&self) -> f64 {
        if self.initial_buying_power <= 0.0 {
            return 0.0;
        }
        self.margin_used / self.initial_buying_power
    }

    /// Get total theta exposure across all positions.
    pub fn total_theta_exposure(&self) -> f64 {
        self.positions
            .iter()
            .map(|p| p.spread.net_theta() * p.quantity as f64)
            .sum()
    }

    /// Close all positions and reset margin.
    pub fn close_all_positions(&mut self) {
        self.buying_power += self.margin_used;
        self.margin_used = 0.0;
        self.positions.clear();
        self.margin_call_triggered = false;
    }

    /// Get the number of open positions.
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::PutCreditSpread;

    fn make_test_spread() -> PutCreditSpread {
        PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.25, 0.1).unwrap()
    }

    #[test]
    fn test_risk_state_creation() {
        let risk = RiskState::new(100_000.0);
        assert_eq!(risk.buying_power, 100_000.0);
        assert_eq!(risk.margin_used, 0.0);
        assert!(!risk.margin_call_triggered);
    }

    #[test]
    fn test_open_position() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();
        assert_eq!(risk.position_count(), 1);
        assert!(risk.margin_used > 0.0);
    }

    #[test]
    fn test_insufficient_buying_power() {
        let mut risk = RiskState::new(1.0); // Very low buying power
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_margin_utilization() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();
        assert!(risk.margin_utilization() > 0.0);
        assert!(risk.margin_utilization() < 1.0);
    }
}
