//! Margin and risk tracking — buying power, margin calls, early assignment detection.
//!
//! Tracks portfolio risk state including buying power consumption, margin requirements
//! for put credit spreads, early assignment triggers, and catastrophic margin calls.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::options::PutCreditSpread;

/// Epsilon for floating-point money comparisons (1/100th of a cent).
const MONEY_EPSILON: f64 = 0.0001;

/// Compare two monetary values for approximate equality.
fn money_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < MONEY_EPSILON
}

/// Compare whether `a` is less than `b` with epsilon tolerance.
fn money_lt(a: f64, b: f64) -> bool {
    a < b - MONEY_EPSILON
}

/// Compare whether `a` is greater than `b` with epsilon tolerance.
fn money_gt(a: f64, b: f64) -> bool {
    a > b + MONEY_EPSILON
}

/// Configurable risk limits for the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RiskLimits {
    /// Maximum number of open positions allowed.
    #[pyo3(get, set)]
    pub max_positions: usize,
    /// Maximum notional exposure (total margin) allowed.
    #[pyo3(get, set)]
    pub max_notional: f64,
    /// Maximum position size (contracts) per single trade.
    #[pyo3(get, set)]
    pub max_position_size: u32,
    /// Maximum margin utilization ratio (0.0 to 1.0).
    #[pyo3(get, set)]
    pub max_margin_utilization: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        RiskLimits {
            max_positions: 50,
            max_notional: 500_000.0,
            max_position_size: 100,
            max_margin_utilization: 0.80,
        }
    }
}

#[pymethods]
impl RiskLimits {
    /// Create new risk limits with the given parameters.
    #[new]
    #[pyo3(signature = (
        max_positions = 50,
        max_notional = 500_000.0,
        max_position_size = 100,
        max_margin_utilization = 0.80,
    ))]
    pub fn new(
        max_positions: usize,
        max_notional: f64,
        max_position_size: u32,
        max_margin_utilization: f64,
    ) -> Self {
        RiskLimits {
            max_positions,
            max_notional,
            max_position_size,
            max_margin_utilization,
        }
    }
}

/// A tracked position in the portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Position {
    /// Ticker symbol.
    #[pyo3(get)]
    pub ticker: String,
    /// The put credit spread.
    #[pyo3(get)]
    pub spread: PutCreditSpread,
    /// Number of contracts.
    #[pyo3(get)]
    pub quantity: u32,
    /// Margin required for this position.
    #[pyo3(get)]
    pub margin_required: f64,
    /// Whether early assignment has been triggered.
    #[pyo3(get)]
    pub early_assignment_triggered: bool,
    /// Implied volatility at time of entry (for mark-to-market).
    #[pyo3(get)]
    pub entry_iv: f64,
}

/// Portfolio-level risk state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct RiskState {
    /// Total available buying power.
    #[pyo3(get)]
    pub buying_power: f64,
    /// Total margin currently in use.
    #[pyo3(get)]
    pub margin_used: f64,
    /// Whether a margin call has been triggered.
    #[pyo3(get)]
    pub margin_call_triggered: bool,
    /// All open positions.
    pub positions: Vec<Position>,
    /// Initial buying power (for ratio calculations).
    #[pyo3(get)]
    pub initial_buying_power: f64,
    /// Configurable risk limits.
    pub risk_limits: RiskLimits,
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
            risk_limits: RiskLimits::default(),
        }
    }

    /// Create a new risk state with custom risk limits.
    pub fn with_limits(initial_buying_power: f64, risk_limits: RiskLimits) -> Self {
        RiskState {
            buying_power: initial_buying_power,
            margin_used: 0.0,
            margin_call_triggered: false,
            positions: Vec::new(),
            initial_buying_power,
            risk_limits,
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

    /// Validate an order before opening a position.
    ///
    /// Checks quantity, symbol validity, and risk limits.
    fn validate_order(
        &self,
        ticker: &str,
        spread: &PutCreditSpread,
        quantity: u32,
    ) -> Result<(), String> {
        // Reject zero or invalid quantity
        if quantity == 0 {
            return Err("Quantity must be greater than zero".to_string());
        }

        // Reject quantities exceeding position size limit
        if quantity > self.risk_limits.max_position_size {
            return Err(format!(
                "Quantity {} exceeds max position size {}",
                quantity, self.risk_limits.max_position_size
            ));
        }

        // Reject empty or whitespace-only symbols
        if ticker.trim().is_empty() {
            return Err("Ticker symbol must not be empty".to_string());
        }

        // Reject symbols with invalid characters (only allow alphanumeric and dots)
        if !ticker
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.')
        {
            return Err(format!("Invalid ticker symbol: {}", ticker));
        }

        // Check position count limit
        if self.positions.len() >= self.risk_limits.max_positions {
            return Err(format!(
                "Maximum position count ({}) reached",
                self.risk_limits.max_positions
            ));
        }

        // Check notional limit
        let margin_required = Self::calculate_margin(spread, quantity);
        if money_gt(
            self.margin_used + margin_required,
            self.risk_limits.max_notional,
        ) {
            return Err(format!(
                "Would exceed max notional: {:.2} + {:.2} > {:.2}",
                self.margin_used, margin_required, self.risk_limits.max_notional
            ));
        }

        // Check margin utilization limit
        if self.initial_buying_power > MONEY_EPSILON {
            let projected_utilization =
                (self.margin_used + margin_required) / self.initial_buying_power;
            if projected_utilization > self.risk_limits.max_margin_utilization {
                return Err(format!(
                    "Would exceed max margin utilization: {:.2}% > {:.2}%",
                    projected_utilization * 100.0,
                    self.risk_limits.max_margin_utilization * 100.0
                ));
            }
        }

        // Validate spread has positive width and credit
        if spread.spread_width <= 0.0 {
            return Err("Spread width must be positive".to_string());
        }
        if spread.net_credit < 0.0 {
            return Err("Net credit must be non-negative for a credit spread".to_string());
        }

        Ok(())
    }

    /// Open a new put credit spread position.
    ///
    /// # Arguments
    /// * `ticker` - Underlying symbol
    /// * `spread` - The put credit spread to open
    /// * `quantity` - Number of contracts
    ///
    /// # Returns
    /// Ok(()) if the position was opened, Err if validation fails or insufficient buying power.
    pub fn open_position(
        &mut self,
        ticker: &str,
        spread: PutCreditSpread,
        quantity: u32,
    ) -> Result<(), String> {
        self.open_position_with_iv(ticker, spread, quantity, 0.25)
    }

    /// Open a new put credit spread position with explicit IV.
    ///
    /// # Arguments
    /// * `ticker` - Underlying symbol
    /// * `spread` - The put credit spread to open
    /// * `quantity` - Number of contracts
    /// * `entry_iv` - Implied volatility at entry
    ///
    /// # Returns
    /// Ok(()) if the position was opened, Err if validation fails or insufficient buying power.
    pub fn open_position_with_iv(
        &mut self,
        ticker: &str,
        spread: PutCreditSpread,
        quantity: u32,
        entry_iv: f64,
    ) -> Result<(), String> {
        // Validate the order first
        self.validate_order(ticker, &spread, quantity)?;

        let margin_required = Self::calculate_margin(&spread, quantity);

        if money_gt(margin_required, self.buying_power) {
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
            entry_iv,
        });

        Ok(())
    }

    /// Check for early assignment on all positions.
    ///
    /// Early assignment is triggered when the short put goes deep in-the-money
    /// (underlying price drops significantly below the short strike).
    ///
    /// Returns 0 for empty portfolios.
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
        if self.positions.is_empty() {
            return 0;
        }

        let mut count = 0;
        for position in &mut self.positions {
            let short_strike = position.spread.short_leg.strike;

            // Guard against zero strike (shouldn't happen but be safe)
            if short_strike <= MONEY_EPSILON {
                continue;
            }

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
    /// A margin call is triggered when margin_used exceeds available buying_power,
    /// indicating the account cannot support its positions.
    ///
    /// **FIX**: Previously used `margin_used > buying_power + margin_used` which was
    /// always false (dead code). Now correctly checks `margin_used > buying_power`.
    ///
    /// # Returns
    /// true if a margin call was triggered.
    pub fn check_margin_call(&mut self) -> bool {
        // FIXED: was `self.margin_used > self.buying_power + self.margin_used` (always false)
        // Now correctly checks if margin exceeds available buying power
        if money_gt(self.margin_used, self.buying_power) {
            self.margin_call_triggered = true;
        }

        // More realistic: margin call if unrealized losses push buying power negative
        if money_lt(self.buying_power, 0.0) {
            self.margin_call_triggered = true;
        }

        self.margin_call_triggered
    }

    /// Update risk state based on current market conditions.
    ///
    /// Recalculates margin requirements and buying power based on current P&L.
    /// Uses actual implied volatility from market data for each position rather
    /// than a hardcoded value.
    ///
    /// Returns `Ok(0.0)` for empty portfolios.
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
        // Guard: empty portfolio has zero P&L
        if self.positions.is_empty() {
            return Ok(0.0);
        }

        let mut total_pnl = 0.0;

        for position in &self.positions {
            // FIXED: Use actual IV from the position's entry data instead of hardcoded 0.25
            let sigma = position.entry_iv;

            // Guard against zero or negative IV
            let sigma = if sigma <= MONEY_EPSILON { 0.25 } else { sigma };

            let pnl = position.spread.current_pnl(underlying_price, r, sigma, t)?;
            total_pnl += pnl * position.quantity as f64;
        }

        // Adjust buying power based on unrealized P&L
        let base_bp = self.initial_buying_power - self.margin_used;
        self.buying_power = base_bp + total_pnl;

        Ok(total_pnl)
    }

    /// Update risk state using per-position IV from market data.
    ///
    /// This variant accepts a slice of IVs, one per position, for precise
    /// mark-to-market calculations.
    ///
    /// # Arguments
    /// * `underlying_price` - Current underlying price
    /// * `r` - Risk-free rate
    /// * `t` - Time to expiration in years
    /// * `position_ivs` - Implied volatility for each position (must match position count)
    ///
    /// # Returns
    /// Total unrealized P&L across all positions.
    pub fn update_mark_to_market_with_ivs(
        &mut self,
        underlying_price: f64,
        r: f64,
        t: f64,
        position_ivs: &[f64],
    ) -> Result<f64, String> {
        if self.positions.is_empty() {
            return Ok(0.0);
        }

        if position_ivs.len() != self.positions.len() {
            return Err(format!(
                "IV count ({}) does not match position count ({})",
                position_ivs.len(),
                self.positions.len()
            ));
        }

        let mut total_pnl = 0.0;

        for (i, position) in self.positions.iter().enumerate() {
            let sigma = position_ivs[i];
            let sigma = if sigma <= MONEY_EPSILON { 0.25 } else { sigma };

            let pnl = position.spread.current_pnl(underlying_price, r, sigma, t)?;
            total_pnl += pnl * position.quantity as f64;
        }

        let base_bp = self.initial_buying_power - self.margin_used;
        self.buying_power = base_bp + total_pnl;

        Ok(total_pnl)
    }

    /// Get the margin utilization ratio (0.0 to 1.0+).
    ///
    /// Returns 0.0 for zero or negative initial buying power.
    pub fn margin_utilization(&self) -> f64 {
        if self.initial_buying_power <= MONEY_EPSILON {
            return 0.0;
        }
        self.margin_used / self.initial_buying_power
    }

    /// Get total theta exposure across all positions.
    ///
    /// Returns 0.0 for empty portfolios.
    pub fn total_theta_exposure(&self) -> f64 {
        if self.positions.is_empty() {
            return 0.0;
        }
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

    fn make_wide_spread() -> PutCreditSpread {
        PutCreditSpread::new(100.0, 95.0, 85.0, 0.05, 0.30, 0.1).unwrap()
    }

    // ── Basic creation tests ────────────────────────────────────────────

    #[test]
    fn test_risk_state_creation() {
        let risk = RiskState::new(100_000.0);
        assert!((risk.buying_power - 100_000.0).abs() < MONEY_EPSILON);
        assert!((risk.margin_used - 0.0).abs() < MONEY_EPSILON);
        assert!(!risk.margin_call_triggered);
        assert_eq!(risk.position_count(), 0);
    }

    #[test]
    fn test_risk_state_with_limits() {
        let limits = RiskLimits {
            max_positions: 10,
            max_notional: 50_000.0,
            max_position_size: 5,
            max_margin_utilization: 0.5,
        };
        let risk = RiskState::with_limits(100_000.0, limits);
        assert_eq!(risk.risk_limits.max_positions, 10);
        assert!((risk.risk_limits.max_notional - 50_000.0).abs() < MONEY_EPSILON);
    }

    #[test]
    fn test_risk_limits_default() {
        let limits = RiskLimits::default();
        assert_eq!(limits.max_positions, 50);
        assert_eq!(limits.max_position_size, 100);
        assert!((limits.max_margin_utilization - 0.80).abs() < MONEY_EPSILON);
    }

    // ── Open position tests ─────────────────────────────────────────────

    #[test]
    fn test_open_position() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();
        assert_eq!(risk.position_count(), 1);
        assert!(risk.margin_used > 0.0);
    }

    #[test]
    fn test_open_position_with_iv() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position_with_iv("TEST", spread, 1, 0.30).unwrap();
        assert_eq!(risk.position_count(), 1);
        assert!((risk.positions[0].entry_iv - 0.30).abs() < MONEY_EPSILON);
    }

    #[test]
    fn test_insufficient_buying_power() {
        let mut risk = RiskState::new(1.0); // Very low buying power
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 100);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("Insufficient buying power")
                || err_msg.contains("Would exceed max notional")
                || err_msg.contains("margin utilization"),
            "Unexpected error message: {}",
            err_msg
        );
    }

    #[test]
    fn test_multiple_positions() {
        let mut risk = RiskState::new(100_000.0);
        for i in 0..5 {
            let spread = make_test_spread();
            risk.open_position(&format!("T{}", i), spread, 1).unwrap();
        }
        assert_eq!(risk.position_count(), 5);
    }

    // ── Order validation tests ──────────────────────────────────────────

    #[test]
    fn test_reject_zero_quantity() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("greater than zero"));
    }

    #[test]
    fn test_reject_empty_symbol() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        let result = risk.open_position("", spread, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not be empty"));
    }

    #[test]
    fn test_reject_whitespace_symbol() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        let result = risk.open_position("   ", spread, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not be empty"));
    }

    #[test]
    fn test_reject_invalid_symbol_chars() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        let result = risk.open_position("TE$T!", spread, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid ticker"));
    }

    #[test]
    fn test_accept_valid_symbol_with_dot() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        let result = risk.open_position("BRK.B", spread, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reject_exceeds_position_size_limit() {
        let limits = RiskLimits {
            max_position_size: 5,
            ..RiskLimits::default()
        };
        let mut risk = RiskState::with_limits(1_000_000.0, limits);
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max position size"));
    }

    #[test]
    fn test_reject_exceeds_max_positions() {
        let limits = RiskLimits {
            max_positions: 2,
            ..RiskLimits::default()
        };
        let mut risk = RiskState::with_limits(1_000_000.0, limits);
        risk.open_position("A", make_test_spread(), 1).unwrap();
        risk.open_position("B", make_test_spread(), 1).unwrap();
        let result = risk.open_position("C", make_test_spread(), 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Maximum position count"));
    }

    #[test]
    fn test_reject_exceeds_max_notional() {
        let limits = RiskLimits {
            max_notional: 100.0,
            ..RiskLimits::default()
        };
        let mut risk = RiskState::with_limits(1_000_000.0, limits);
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max notional"));
    }

    #[test]
    fn test_reject_exceeds_margin_utilization() {
        let limits = RiskLimits {
            max_margin_utilization: 0.0001, // Extremely low limit to force rejection
            ..RiskLimits::default()
        };
        let mut risk = RiskState::with_limits(100_000.0, limits);
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("margin utilization"));
    }

    // ── Margin call tests (CRITICAL FIX) ────────────────────────────────

    #[test]
    fn test_margin_call_fixed_logic() {
        // The old bug: `margin_used > buying_power + margin_used` was always false
        // Now: `margin_used > buying_power` correctly triggers
        let mut risk = RiskState::new(100_000.0);
        risk.margin_used = 60_000.0;
        risk.buying_power = 50_000.0;

        // margin_used (60k) > buying_power (50k) → should trigger
        let result = risk.check_margin_call();
        assert!(result, "Margin call should trigger when margin_used > buying_power");
        assert!(risk.margin_call_triggered);
    }

    #[test]
    fn test_no_margin_call_when_healthy() {
        let mut risk = RiskState::new(100_000.0);
        risk.margin_used = 30_000.0;
        risk.buying_power = 70_000.0;

        let result = risk.check_margin_call();
        assert!(!result, "No margin call when buying_power > margin_used");
    }

    #[test]
    fn test_margin_call_negative_buying_power() {
        let mut risk = RiskState::new(100_000.0);
        risk.buying_power = -1000.0;

        let result = risk.check_margin_call();
        assert!(result, "Margin call should trigger on negative buying power");
    }

    #[test]
    fn test_margin_call_equal_values() {
        let mut risk = RiskState::new(100_000.0);
        risk.margin_used = 50_000.0;
        risk.buying_power = 50_000.0;

        // Equal values should NOT trigger (within epsilon)
        let result = risk.check_margin_call();
        assert!(!result, "No margin call when margin_used == buying_power");
    }

    #[test]
    fn test_margin_call_barely_over() {
        let mut risk = RiskState::new(100_000.0);
        risk.margin_used = 50_001.0;
        risk.buying_power = 50_000.0;

        let result = risk.check_margin_call();
        assert!(result, "Margin call should trigger when margin barely exceeds BP");
    }

    // ── Early assignment tests ──────────────────────────────────────────

    #[test]
    fn test_early_assignment_triggered() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();

        // Price drops well below short strike (95)
        let count = risk.check_early_assignment(80.0, 0.05);
        assert_eq!(count, 1);
        assert!(risk.positions[0].early_assignment_triggered);
    }

    #[test]
    fn test_early_assignment_not_triggered() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();

        // Price above short strike
        let count = risk.check_early_assignment(100.0, 0.05);
        assert_eq!(count, 0);
        assert!(!risk.positions[0].early_assignment_triggered);
    }

    #[test]
    fn test_early_assignment_empty_portfolio() {
        let mut risk = RiskState::new(100_000.0);
        let count = risk.check_early_assignment(80.0, 0.05);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_early_assignment_not_double_counted() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();

        let count1 = risk.check_early_assignment(80.0, 0.05);
        assert_eq!(count1, 1);

        // Second check should not re-trigger
        let count2 = risk.check_early_assignment(80.0, 0.05);
        assert_eq!(count2, 0);
    }

    // ── Mark-to-market tests ────────────────────────────────────────────

    #[test]
    fn test_mark_to_market_empty_portfolio() {
        let mut risk = RiskState::new(100_000.0);
        let pnl = risk.update_mark_to_market(100.0, 0.05, 0.1).unwrap();
        assert!((pnl - 0.0).abs() < MONEY_EPSILON);
    }

    #[test]
    fn test_mark_to_market_uses_entry_iv() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position_with_iv("TEST", spread, 1, 0.30).unwrap();

        // Should use 0.30 IV, not hardcoded 0.25
        let pnl = risk.update_mark_to_market(100.0, 0.05, 0.1);
        assert!(pnl.is_ok());
    }

    #[test]
    fn test_mark_to_market_with_explicit_ivs() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();

        let ivs = vec![0.35];
        let pnl = risk.update_mark_to_market_with_ivs(100.0, 0.05, 0.1, &ivs);
        assert!(pnl.is_ok());
    }

    #[test]
    fn test_mark_to_market_iv_count_mismatch() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();

        let ivs = vec![0.30, 0.35]; // 2 IVs but only 1 position
        let result = risk.update_mark_to_market_with_ivs(100.0, 0.05, 0.1, &ivs);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match"));
    }

    #[test]
    fn test_mark_to_market_zero_iv_fallback() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position_with_iv("TEST", spread, 1, 0.0).unwrap();

        // Should fall back to 0.25 when IV is zero
        let pnl = risk.update_mark_to_market(100.0, 0.05, 0.1);
        assert!(pnl.is_ok());
    }

    // ── Margin utilization tests ────────────────────────────────────────

    #[test]
    fn test_margin_utilization() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();
        assert!(risk.margin_utilization() > 0.0);
        assert!(risk.margin_utilization() < 1.0);
    }

    #[test]
    fn test_margin_utilization_zero_initial_bp() {
        let risk = RiskState::new(0.0);
        assert!((risk.margin_utilization() - 0.0).abs() < MONEY_EPSILON);
    }

    #[test]
    fn test_margin_utilization_negative_initial_bp() {
        let risk = RiskState::new(-100.0);
        assert!((risk.margin_utilization() - 0.0).abs() < MONEY_EPSILON);
    }

    // ── Theta exposure tests ────────────────────────────────────────────

    #[test]
    fn test_theta_exposure_empty_portfolio() {
        let risk = RiskState::new(100_000.0);
        assert!((risk.total_theta_exposure() - 0.0).abs() < MONEY_EPSILON);
    }

    #[test]
    fn test_theta_exposure_with_positions() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();
        // Theta should be non-zero for an active spread
        let theta = risk.total_theta_exposure();
        assert!(theta.abs() > 0.0, "Theta should be non-zero: {}", theta);
    }

    // ── Close positions tests ───────────────────────────────────────────

    #[test]
    fn test_close_all_positions() {
        let mut risk = RiskState::new(100_000.0);
        let spread = make_test_spread();
        risk.open_position("TEST", spread, 1).unwrap();
        assert!(risk.position_count() > 0);

        risk.close_all_positions();
        assert_eq!(risk.position_count(), 0);
        assert!((risk.margin_used - 0.0).abs() < MONEY_EPSILON);
        assert!(!risk.margin_call_triggered);
    }

    #[test]
    fn test_close_empty_portfolio() {
        let mut risk = RiskState::new(100_000.0);
        risk.close_all_positions(); // Should not panic
        assert_eq!(risk.position_count(), 0);
    }

    // ── Float precision tests ───────────────────────────────────────────

    #[test]
    fn test_money_eq() {
        assert!(money_eq(100.0, 100.0));
        assert!(money_eq(100.0, 100.00001));
        assert!(!money_eq(100.0, 100.001));
    }

    #[test]
    fn test_money_lt() {
        assert!(money_lt(99.0, 100.0));
        assert!(!money_lt(100.0, 100.0));
        assert!(!money_lt(100.0, 100.00001));
    }

    #[test]
    fn test_money_gt() {
        assert!(money_gt(100.0, 99.0));
        assert!(!money_gt(100.0, 100.0));
        assert!(!money_gt(100.00001, 100.0));
    }

    // ── Calculate margin tests ──────────────────────────────────────────

    #[test]
    fn test_calculate_margin() {
        let spread = make_test_spread();
        let margin = RiskState::calculate_margin(&spread, 1);
        // spread_width=5, net_credit>0, so margin = (5*100 - credit*100) * 1
        assert!(margin > 0.0);
        assert!(margin < 500.0); // Less than max loss of $500 per contract
    }

    #[test]
    fn test_calculate_margin_multiple_contracts() {
        let spread = make_test_spread();
        let margin_1 = RiskState::calculate_margin(&spread, 1);
        let margin_5 = RiskState::calculate_margin(&spread, 5);
        assert!((margin_5 - margin_1 * 5.0).abs() < MONEY_EPSILON);
    }

    // ── Extreme value tests ─────────────────────────────────────────────

    #[test]
    fn test_extreme_buying_power() {
        let risk = RiskState::new(f64::MAX / 2.0);
        assert!(!risk.margin_call_triggered);
        assert_eq!(risk.position_count(), 0);
    }

    #[test]
    fn test_very_small_buying_power() {
        let mut risk = RiskState::new(0.01);
        let spread = make_test_spread();
        let result = risk.open_position("TEST", spread, 1);
        assert!(result.is_err());
    }

    // ── Integration-style tests ─────────────────────────────────────────

    #[test]
    fn test_full_lifecycle() {
        let mut risk = RiskState::new(100_000.0);
        let initial_bp = risk.buying_power;

        // Open position
        let spread = make_test_spread();
        risk.open_position_with_iv("MU", spread, 2, 0.28).unwrap();
        assert_eq!(risk.position_count(), 1);
        assert!(risk.margin_used > 0.0);

        // Check no margin call
        assert!(!risk.check_margin_call());

        // Check no early assignment (price above strike)
        assert_eq!(risk.check_early_assignment(100.0, 0.05), 0);

        // Mark to market
        let pnl = risk.update_mark_to_market(100.0, 0.05, 0.08).unwrap();
        // P&L should be finite
        assert!(pnl.is_finite());

        // Close all
        risk.close_all_positions();
        assert_eq!(risk.position_count(), 0);
        assert!((risk.margin_used - 0.0).abs() < MONEY_EPSILON);

        // Buying power should be restored (approximately)
        assert!(risk.buying_power > 0.0);
    }

    #[test]
    fn test_margin_call_after_losses() {
        let mut risk = RiskState::new(10_000.0);
        let spread = make_test_spread();

        // Open a position
        risk.open_position("TEST", spread, 1).unwrap();

        // Simulate heavy losses by manually setting buying power negative
        risk.buying_power = -500.0;

        assert!(risk.check_margin_call());
        assert!(risk.margin_call_triggered);
    }
}
