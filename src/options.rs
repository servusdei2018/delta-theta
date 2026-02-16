//! Options pricing via Black-Scholes, Greeks computation, and multi-leg spread structures.
//!
//! Provides put option pricing, Greeks (delta, gamma, theta, vega), implied volatility
//! estimation via Newton-Raphson, and put credit spread P&L calculations.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};

/// Standard normal cumulative distribution function (CDF).
fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Standard normal probability density function (PDF).
fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Error function approximation (Abramowitz & Stegun).
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Black-Scholes d1 parameter.
fn bs_d1(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// Black-Scholes d2 parameter.
fn bs_d2(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    bs_d1(s, k, r, sigma, t) - sigma * t.sqrt()
}

/// Price a European put option using Black-Scholes.
///
/// # Arguments
/// * `s` - Current underlying price
/// * `k` - Strike price
/// * `r` - Risk-free interest rate (annualized)
/// * `sigma` - Implied volatility (annualized)
/// * `t` - Time to expiration in years
///
/// # Returns
/// Put option price, or an error if inputs are invalid.
pub fn black_scholes_put(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> Result<f64, String> {
    if s <= 0.0 {
        return Err("Underlying price must be positive".to_string());
    }
    if k <= 0.0 {
        return Err("Strike price must be positive".to_string());
    }
    if sigma <= 0.0 {
        return Err("Volatility must be positive".to_string());
    }
    if t <= 0.0 {
        // At expiration, return intrinsic value
        return Ok((k - s).max(0.0));
    }

    let d1 = bs_d1(s, k, r, sigma, t);
    let d2 = bs_d2(s, k, r, sigma, t);

    let put_price = k * (-r * t).exp() * norm_cdf(-d2) - s * norm_cdf(-d1);
    Ok(put_price)
}

/// Greeks for a European put option.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct Greeks {
    /// Rate of change of option price w.r.t. underlying price
    #[pyo3(get)]
    pub delta: f64,
    /// Rate of change of delta w.r.t. underlying price
    #[pyo3(get)]
    pub gamma: f64,
    /// Rate of change of option price w.r.t. time (per day)
    #[pyo3(get)]
    pub theta: f64,
    /// Rate of change of option price w.r.t. volatility
    #[pyo3(get)]
    pub vega: f64,
}

/// Compute Greeks for a European put option.
///
/// # Arguments
/// * `s` - Current underlying price
/// * `k` - Strike price
/// * `r` - Risk-free interest rate
/// * `sigma` - Implied volatility
/// * `t` - Time to expiration in years
pub fn put_greeks(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> Result<Greeks, String> {
    if s <= 0.0 || k <= 0.0 || sigma <= 0.0 {
        return Err("Price, strike, and volatility must be positive".to_string());
    }
    if t <= 0.0 {
        // At expiration, delta is -1 if ITM, 0 if OTM
        let delta = if s < k { -1.0 } else { 0.0 };
        return Ok(Greeks {
            delta,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
        });
    }

    let d1 = bs_d1(s, k, r, sigma, t);
    let d2 = bs_d2(s, k, r, sigma, t);

    // Put delta = N(d1) - 1
    let delta = norm_cdf(d1) - 1.0;

    // Gamma (same for put and call)
    let gamma = norm_pdf(d1) / (s * sigma * t.sqrt());

    // Put theta (per year, we convert to per day by dividing by 365)
    let theta_annual = -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt())
        + r * k * (-r * t).exp() * norm_cdf(-d2);
    let theta = theta_annual / 365.0;

    // Vega (per 1% move in vol)
    let vega = s * t.sqrt() * norm_pdf(d1) / 100.0;

    Ok(Greeks {
        delta,
        gamma,
        theta,
        vega,
    })
}

/// Implied volatility estimation using Newton-Raphson method.
///
/// # Arguments
/// * `market_price` - Observed market price of the put
/// * `s` - Underlying price
/// * `k` - Strike price
/// * `r` - Risk-free rate
/// * `t` - Time to expiration in years
/// * `max_iterations` - Maximum Newton-Raphson iterations
///
/// # Returns
/// Estimated implied volatility, or error if convergence fails.
pub fn implied_volatility(
    market_price: f64,
    s: f64,
    k: f64,
    r: f64,
    t: f64,
    max_iterations: usize,
) -> Result<f64, String> {
    if market_price <= 0.0 {
        return Err("Market price must be positive".to_string());
    }
    if t <= 0.0 {
        return Err("Time to expiry must be positive for IV calculation".to_string());
    }

    let mut sigma = 0.3; // Initial guess
    let tolerance = 1e-8;

    for _ in 0..max_iterations {
        let price = black_scholes_put(s, k, r, sigma, t)?;
        let diff = price - market_price;

        if diff.abs() < tolerance {
            return Ok(sigma);
        }

        // Vega for Newton-Raphson step (not scaled by 100)
        let d1 = bs_d1(s, k, r, sigma, t);
        let vega = s * t.sqrt() * norm_pdf(d1);

        if vega.abs() < 1e-12 {
            return Err("Vega too small, cannot converge".to_string());
        }

        sigma -= diff / vega;

        // Clamp sigma to reasonable bounds
        sigma = sigma.clamp(0.001, 10.0);
    }

    Err(format!(
        "IV did not converge after {} iterations (last sigma={:.6})",
        max_iterations, sigma
    ))
}

/// A single option leg in a spread structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct OptionLeg {
    /// Strike price
    #[pyo3(get)]
    pub strike: f64,
    /// True if this leg is a short position
    #[pyo3(get)]
    pub is_short: bool,
    /// Option price (premium)
    #[pyo3(get)]
    pub premium: f64,
    /// Greeks for this leg
    #[pyo3(get)]
    pub greeks: Greeks,
}

/// Put credit spread: short a higher-strike put, long a lower-strike put.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PutCreditSpread {
    /// The short put leg (higher strike)
    #[pyo3(get)]
    pub short_leg: OptionLeg,
    /// The long put leg (lower strike)
    #[pyo3(get)]
    pub long_leg: OptionLeg,
    /// Net credit received (short premium - long premium)
    #[pyo3(get)]
    pub net_credit: f64,
    /// Maximum possible loss (spread width * 100 - net credit * 100)
    #[pyo3(get)]
    pub max_loss: f64,
    /// Maximum possible profit (net credit * 100)
    #[pyo3(get)]
    pub max_profit: f64,
    /// Width between strikes
    #[pyo3(get)]
    pub spread_width: f64,
}

impl PutCreditSpread {
    /// Construct a put credit spread given market parameters.
    ///
    /// # Arguments
    /// * `s` - Underlying price
    /// * `short_strike` - Strike of the short put (higher)
    /// * `long_strike` - Strike of the long put (lower)
    /// * `r` - Risk-free rate
    /// * `sigma` - Implied volatility
    /// * `t` - Time to expiration in years
    pub fn new(
        s: f64,
        short_strike: f64,
        long_strike: f64,
        r: f64,
        sigma: f64,
        t: f64,
    ) -> Result<Self, String> {
        if short_strike <= long_strike {
            return Err("Short strike must be higher than long strike for a put credit spread".to_string());
        }

        let short_premium = black_scholes_put(s, short_strike, r, sigma, t)?;
        let long_premium = black_scholes_put(s, long_strike, r, sigma, t)?;

        let short_greeks = put_greeks(s, short_strike, r, sigma, t)?;
        let long_greeks = put_greeks(s, long_strike, r, sigma, t)?;

        let net_credit = short_premium - long_premium;
        let spread_width = short_strike - long_strike;
        let max_loss = spread_width * 100.0 - net_credit * 100.0;
        let max_profit = net_credit * 100.0;

        Ok(PutCreditSpread {
            short_leg: OptionLeg {
                strike: short_strike,
                is_short: true,
                premium: short_premium,
                greeks: short_greeks,
            },
            long_leg: OptionLeg {
                strike: long_strike,
                is_short: false,
                premium: long_premium,
                greeks: long_greeks,
            },
            net_credit,
            max_loss,
            max_profit,
            spread_width,
        })
    }

    /// Calculate current P&L of the spread given new market conditions.
    ///
    /// P&L = (initial net credit - current cost to close) * 100
    pub fn current_pnl(&self, s: f64, r: f64, sigma: f64, t: f64) -> Result<f64, String> {
        let short_price = black_scholes_put(s, self.short_leg.strike, r, sigma, t)?;
        let long_price = black_scholes_put(s, self.long_leg.strike, r, sigma, t)?;

        // Cost to close: buy back short, sell long
        let close_cost = short_price - long_price;
        let pnl = (self.net_credit - close_cost) * 100.0;
        Ok(pnl)
    }

    /// Net delta of the spread.
    pub fn net_delta(&self) -> f64 {
        // Short leg delta is negated (we're short), long leg delta is as-is
        -self.short_leg.greeks.delta + self.long_leg.greeks.delta
    }

    /// Net theta of the spread (positive theta = time decay benefits us).
    pub fn net_theta(&self) -> f64 {
        // Short leg theta benefits us (negated), long leg theta costs us
        -self.short_leg.greeks.theta + self.long_leg.greeks.theta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Black-Scholes pricing tests ─────────────────────────────────────

    #[test]
    fn test_black_scholes_put_atm() {
        let price = black_scholes_put(100.0, 100.0, 0.05, 0.2, 1.0).unwrap();
        // ATM put with 20% vol, 1 year, should be roughly 5-6
        assert!(price > 4.0 && price < 8.0, "ATM put price: {}", price);
    }

    #[test]
    fn test_black_scholes_put_deep_otm() {
        let price = black_scholes_put(100.0, 50.0, 0.05, 0.2, 0.1).unwrap();
        // Deep OTM put should be nearly worthless
        assert!(price < 0.01, "Deep OTM put should be near zero: {}", price);
    }

    #[test]
    fn test_black_scholes_put_deep_itm() {
        let price = black_scholes_put(50.0, 100.0, 0.05, 0.2, 0.1).unwrap();
        // Deep ITM put should be close to intrinsic value
        assert!(price > 45.0, "Deep ITM put should be near intrinsic: {}", price);
    }

    #[test]
    fn test_black_scholes_put_at_expiry() {
        // At expiry, should return intrinsic value
        let itm = black_scholes_put(90.0, 100.0, 0.05, 0.2, 0.0).unwrap();
        assert!((itm - 10.0).abs() < 0.01, "ITM at expiry: {}", itm);

        let otm = black_scholes_put(110.0, 100.0, 0.05, 0.2, 0.0).unwrap();
        assert!((otm - 0.0).abs() < 0.01, "OTM at expiry: {}", otm);
    }

    #[test]
    fn test_black_scholes_put_invalid_inputs() {
        assert!(black_scholes_put(0.0, 100.0, 0.05, 0.2, 1.0).is_err());
        assert!(black_scholes_put(-1.0, 100.0, 0.05, 0.2, 1.0).is_err());
        assert!(black_scholes_put(100.0, 0.0, 0.05, 0.2, 1.0).is_err());
        assert!(black_scholes_put(100.0, -1.0, 0.05, 0.2, 1.0).is_err());
        assert!(black_scholes_put(100.0, 100.0, 0.05, 0.0, 1.0).is_err());
        assert!(black_scholes_put(100.0, 100.0, 0.05, -0.1, 1.0).is_err());
    }

    #[test]
    fn test_black_scholes_put_higher_vol_higher_price() {
        let low_vol = black_scholes_put(100.0, 100.0, 0.05, 0.1, 1.0).unwrap();
        let high_vol = black_scholes_put(100.0, 100.0, 0.05, 0.5, 1.0).unwrap();
        assert!(high_vol > low_vol, "Higher vol should mean higher put price");
    }

    #[test]
    fn test_black_scholes_put_longer_time_higher_price() {
        let short = black_scholes_put(100.0, 100.0, 0.05, 0.2, 0.1).unwrap();
        let long = black_scholes_put(100.0, 100.0, 0.05, 0.2, 1.0).unwrap();
        assert!(long > short, "Longer time should mean higher put price");
    }

    // ── Greeks tests ────────────────────────────────────────────────────

    #[test]
    fn test_put_greeks_delta_range() {
        let greeks = put_greeks(100.0, 100.0, 0.05, 0.2, 1.0).unwrap();
        assert!(greeks.delta > -1.0 && greeks.delta < 0.0);
        assert!(greeks.gamma > 0.0);
    }

    #[test]
    fn test_put_greeks_deep_itm_delta() {
        let greeks = put_greeks(50.0, 100.0, 0.05, 0.2, 0.1).unwrap();
        // Deep ITM put delta should be close to -1
        assert!(greeks.delta < -0.9, "Deep ITM delta: {}", greeks.delta);
    }

    #[test]
    fn test_put_greeks_deep_otm_delta() {
        let greeks = put_greeks(200.0, 100.0, 0.05, 0.2, 0.1).unwrap();
        // Deep OTM put delta should be close to 0
        assert!(greeks.delta > -0.1, "Deep OTM delta: {}", greeks.delta);
    }

    #[test]
    fn test_put_greeks_at_expiry() {
        let itm = put_greeks(90.0, 100.0, 0.05, 0.2, 0.0).unwrap();
        assert!((itm.delta - (-1.0)).abs() < 0.01);
        assert!((itm.gamma - 0.0).abs() < 0.01);

        let otm = put_greeks(110.0, 100.0, 0.05, 0.2, 0.0).unwrap();
        assert!((otm.delta - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_put_greeks_invalid_inputs() {
        assert!(put_greeks(0.0, 100.0, 0.05, 0.2, 1.0).is_err());
        assert!(put_greeks(100.0, 0.0, 0.05, 0.2, 1.0).is_err());
        assert!(put_greeks(100.0, 100.0, 0.05, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_put_greeks_vega_positive() {
        let greeks = put_greeks(100.0, 100.0, 0.05, 0.2, 1.0).unwrap();
        assert!(greeks.vega > 0.0, "Vega should be positive: {}", greeks.vega);
    }

    #[test]
    fn test_put_greeks_theta_negative_for_otm() {
        let greeks = put_greeks(100.0, 95.0, 0.05, 0.2, 0.1).unwrap();
        // OTM put theta should be negative (time decay hurts long puts)
        assert!(greeks.theta < 0.0, "OTM put theta should be negative: {}", greeks.theta);
    }

    // ── Implied volatility tests ────────────────────────────────────────

    #[test]
    fn test_implied_volatility_roundtrip() {
        let true_vol = 0.25;
        let price = black_scholes_put(100.0, 100.0, 0.05, true_vol, 0.5).unwrap();
        let iv = implied_volatility(price, 100.0, 100.0, 0.05, 0.5, 100).unwrap();
        assert!((iv - true_vol).abs() < 0.001, "IV roundtrip failed: {}", iv);
    }

    #[test]
    fn test_implied_volatility_various_vols() {
        for &true_vol in &[0.10, 0.20, 0.30, 0.50, 1.0] {
            let price = black_scholes_put(100.0, 100.0, 0.05, true_vol, 0.5).unwrap();
            let iv = implied_volatility(price, 100.0, 100.0, 0.05, 0.5, 200).unwrap();
            assert!(
                (iv - true_vol).abs() < 0.01,
                "IV roundtrip failed for vol={}: got {}",
                true_vol,
                iv
            );
        }
    }

    #[test]
    fn test_implied_volatility_invalid_inputs() {
        assert!(implied_volatility(0.0, 100.0, 100.0, 0.05, 0.5, 100).is_err());
        assert!(implied_volatility(-1.0, 100.0, 100.0, 0.05, 0.5, 100).is_err());
        assert!(implied_volatility(5.0, 100.0, 100.0, 0.05, 0.0, 100).is_err());
    }

    // ── Put credit spread tests ─────────────────────────────────────────

    #[test]
    fn test_put_credit_spread() {
        let spread = PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.2, 0.1).unwrap();
        assert!(spread.net_credit > 0.0, "Credit spread should receive net credit");
        assert!((spread.spread_width - 5.0).abs() < 1e-10);
        assert!(spread.max_loss > 0.0);
    }

    #[test]
    fn test_put_credit_spread_invalid_strikes() {
        // Short strike must be higher than long strike
        let result = PutCreditSpread::new(100.0, 90.0, 95.0, 0.05, 0.2, 0.1);
        assert!(result.is_err());

        // Equal strikes
        let result = PutCreditSpread::new(100.0, 95.0, 95.0, 0.05, 0.2, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_put_credit_spread_max_profit_equals_credit() {
        let spread = PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.2, 0.1).unwrap();
        assert!(
            (spread.max_profit - spread.net_credit * 100.0).abs() < 0.01,
            "Max profit should equal net credit * 100"
        );
    }

    #[test]
    fn test_put_credit_spread_pnl_at_entry() {
        let spread = PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.25, 0.1).unwrap();
        // P&L at entry conditions should be approximately zero
        let pnl = spread.current_pnl(100.0, 0.05, 0.25, 0.1).unwrap();
        assert!(pnl.abs() < 1.0, "P&L at entry should be near zero: {}", pnl);
    }

    #[test]
    fn test_put_credit_spread_pnl_favorable() {
        let spread = PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.25, 0.1).unwrap();
        // Price moves up → favorable for put credit spread
        let pnl = spread.current_pnl(110.0, 0.05, 0.25, 0.05).unwrap();
        assert!(pnl > 0.0, "P&L should be positive when price rises: {}", pnl);
    }

    #[test]
    fn test_put_credit_spread_net_delta() {
        let spread = PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.25, 0.1).unwrap();
        let delta = spread.net_delta();
        // Net delta of a put credit spread should be positive (bullish)
        assert!(delta.is_finite(), "Net delta should be finite: {}", delta);
    }

    #[test]
    fn test_put_credit_spread_net_theta() {
        let spread = PutCreditSpread::new(100.0, 95.0, 90.0, 0.05, 0.25, 0.1).unwrap();
        let theta = spread.net_theta();
        assert!(theta.is_finite(), "Net theta should be finite: {}", theta);
    }

    // ── Edge case tests ─────────────────────────────────────────────────

    #[test]
    fn test_very_high_volatility() {
        let price = black_scholes_put(100.0, 100.0, 0.05, 5.0, 1.0).unwrap();
        assert!(price.is_finite() && price > 0.0);
    }

    #[test]
    fn test_very_short_time() {
        let price = black_scholes_put(100.0, 100.0, 0.05, 0.2, 0.001).unwrap();
        assert!(price.is_finite() && price >= 0.0);
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((norm_cdf(1.0) + norm_cdf(-1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_pdf_symmetry() {
        assert!((norm_pdf(1.0) - norm_pdf(-1.0)).abs() < 1e-10);
        assert!(norm_pdf(0.0) > norm_pdf(1.0));
    }
}
