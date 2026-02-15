//! BacktestEngine — order book simulation with nanosecond precision and episodic resets.
//!
//! Core simulation engine that manages the order book state, executes put credit spread
//! trades, simulates theta decay and volatility spikes, computes P&L, and checks margin.

use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::options::PutCreditSpread;
use crate::orderbook::UnderlyingOrderBook;
use crate::risk::RiskState;

/// Configuration for the backtest engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct EngineConfig {
    /// Ticker symbols to simulate
    #[pyo3(get, set)]
    pub tickers: Vec<String>,
    /// Initial underlying prices for each ticker
    #[pyo3(get, set)]
    pub initial_prices: Vec<f64>,
    /// Initial buying power
    #[pyo3(get, set)]
    pub initial_buying_power: f64,
    /// Risk-free interest rate
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    /// Initial time to expiration in years
    #[pyo3(get, set)]
    pub initial_tte: f64,
    /// Time decay per tick (in years)
    #[pyo3(get, set)]
    pub time_decay_per_tick: f64,
    /// Maximum episode steps before forced termination
    #[pyo3(get, set)]
    pub max_episode_steps: u64,
    /// Random seed (0 for random)
    #[pyo3(get, set)]
    pub seed: u64,
    /// Number of strikes to generate per underlying
    #[pyo3(get, set)]
    pub num_strikes: usize,
    /// Strike spacing (e.g., 5.0 for $5 wide strikes)
    #[pyo3(get, set)]
    pub strike_spacing: f64,
    /// Base implied volatility
    #[pyo3(get, set)]
    pub base_iv: f64,
    /// Probability of a vol spike per tick
    #[pyo3(get, set)]
    pub vol_spike_prob: f64,
    /// Magnitude of vol spikes (multiplier)
    #[pyo3(get, set)]
    pub vol_spike_magnitude: f64,
    /// Early assignment ITM threshold (fraction of strike)
    #[pyo3(get, set)]
    pub early_assignment_threshold: f64,
}

#[pymethods]
impl EngineConfig {
    /// Create a new engine configuration with sensible defaults.
    #[new]
    #[pyo3(signature = (
        tickers = None,
        initial_prices = None,
        initial_buying_power = 100_000.0,
        risk_free_rate = 0.05,
        initial_tte = 0.0833,
        max_episode_steps = 1000,
        seed = 0,
    ))]
    pub fn new(
        tickers: Option<Vec<String>>,
        initial_prices: Option<Vec<f64>>,
        initial_buying_power: f64,
        risk_free_rate: f64,
        initial_tte: f64,
        max_episode_steps: u64,
        seed: u64,
    ) -> Self {
        let tickers = tickers.unwrap_or_else(|| vec!["MU".to_string(), "AMD".to_string()]);
        let initial_prices = initial_prices.unwrap_or_else(|| vec![85.0, 120.0]);

        EngineConfig {
            tickers,
            initial_prices,
            initial_buying_power,
            risk_free_rate,
            initial_tte,
            time_decay_per_tick: 1.0 / (252.0 * 390.0), // ~1 minute per tick
            max_episode_steps,
            seed,
            num_strikes: 11,
            strike_spacing: 2.5,
            base_iv: 0.30,
            vol_spike_prob: 0.02,
            vol_spike_magnitude: 1.5,
            early_assignment_threshold: 0.05,
        }
    }
}

/// Result of a single simulation step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct StepResult {
    /// Observation vector (state)
    #[pyo3(get)]
    pub observation: Vec<f64>,
    /// Reward for this step
    #[pyo3(get)]
    pub reward: f64,
    /// Whether the episode is done
    #[pyo3(get)]
    pub done: bool,
    /// Whether the episode was truncated (max steps reached)
    #[pyo3(get)]
    pub truncated: bool,
    /// Additional info as JSON string
    #[pyo3(get)]
    pub info: String,
}

/// The core backtest engine, exposed as a Python class.
#[pyclass]
pub struct BacktestEngine {
    /// Engine configuration
    config: EngineConfig,
    /// Order books for each underlying
    order_books: Vec<UnderlyingOrderBook>,
    /// Risk/margin state
    risk_state: RiskState,
    /// Current timestamp in nanoseconds
    current_timestamp_ns: u64,
    /// Episode step counter
    episode_step: u64,
    /// Current time to expiration
    current_tte: f64,
    /// Random number generator
    rng: StdRng,
    /// Cross-reset cache: cumulative P&L across episodes
    cumulative_pnl: f64,
    /// Current episode's realized P&L
    episode_pnl: f64,
    /// Current implied volatility (can spike)
    current_iv: f64,
}

#[pymethods]
impl BacktestEngine {
    /// Create a new BacktestEngine with the given configuration.
    #[new]
    pub fn new(config: EngineConfig) -> Result<Self, PyErr> {
        let rng = if config.seed == 0 {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(config.seed)
        };

        let mut engine = BacktestEngine {
            config: config.clone(),
            order_books: Vec::new(),
            risk_state: RiskState::new(config.initial_buying_power),
            current_timestamp_ns: 0,
            episode_step: 0,
            current_tte: config.initial_tte,
            rng,
            cumulative_pnl: 0.0,
            episode_pnl: 0.0,
            current_iv: config.base_iv,
        };

        engine.initialize_order_books().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to initialize: {}", e))
        })?;

        Ok(engine)
    }

    /// Reset the episode: regenerate order book state, return initial observation.
    pub fn reset(&mut self) -> Result<Vec<f64>, PyErr> {
        // Accumulate P&L from previous episode
        self.cumulative_pnl += self.episode_pnl;

        // Reset episode state
        self.episode_step = 0;
        self.episode_pnl = 0.0;
        self.current_tte = self.config.initial_tte;
        self.current_iv = self.config.base_iv;
        self.current_timestamp_ns = 0;

        // Reset risk state
        self.risk_state = RiskState::new(self.config.initial_buying_power);

        // Regenerate order books
        self.initialize_order_books().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to reset: {}", e))
        })?;

        Ok(self.get_observation())
    }

    /// Advance simulation by one tick: execute trade, simulate decay, compute P&L.
    ///
    /// # Arguments
    /// * `strike_width` - Width of the put credit spread (distance between strikes)
    /// * `delta_position` - Target delta for the short put (e.g., -0.3 for 30-delta)
    pub fn step(&mut self, strike_width: f64, delta_position: f64) -> Result<StepResult, PyErr> {
        self.episode_step += 1;
        self.current_timestamp_ns += 60_000_000_000; // 1 minute in nanoseconds

        // Simulate potential volatility spike
        self.simulate_vol_spike();

        // Advance order books by one tick
        for book in &mut self.order_books {
            book.tick(&mut self.rng);
        }

        // Decay time to expiration
        self.current_tte -= self.config.time_decay_per_tick;
        self.current_tte = self.current_tte.max(0.001);

        // Execute the put credit spread trade on the first underlying
        let trade_result = self.execute_spread_trade(strike_width, delta_position);

        // Calculate step reward
        let (reward, info) = match trade_result {
            Ok(pnl) => {
                self.episode_pnl += pnl;

                // Check for early assignment
                if let Some(book) = self.order_books.first() {
                    self.risk_state.check_early_assignment(
                        book.underlying_price,
                        self.config.early_assignment_threshold,
                    );
                }

                // Update mark-to-market
                if let Some(book) = self.order_books.first() {
                    let _ = self.risk_state.update_mark_to_market(
                        book.underlying_price,
                        self.config.risk_free_rate,
                        self.current_tte,
                    );
                }

                let theta_captured = self.risk_state.total_theta_exposure().abs();
                let margin_ratio = self.risk_state.margin_utilization();

                // Reward = theta captured - cubic penalty for margin expansion
                let reward = theta_captured - margin_ratio.powi(3) * 10.0 + pnl * 0.01;

                let info = serde_json::json!({
                    "pnl": pnl,
                    "episode_pnl": self.episode_pnl,
                    "margin_utilization": margin_ratio,
                    "theta_exposure": theta_captured,
                    "current_iv": self.current_iv,
                    "tte": self.current_tte,
                    "positions": self.risk_state.position_count(),
                });

                (reward, info.to_string())
            }
            Err(e) => {
                let info = serde_json::json!({
                    "error": e,
                    "episode_pnl": self.episode_pnl,
                });
                (-0.1, info.to_string()) // Small penalty for failed trades
            }
        };

        // Check termination conditions
        let margin_call = self.risk_state.check_margin_call();
        let max_steps_reached = self.episode_step >= self.config.max_episode_steps;
        let expired = self.current_tte <= 0.002;

        let (done, truncated, final_reward) = if margin_call {
            // Catastrophic margin call: large negative reward
            (true, false, reward - 100.0)
        } else if expired {
            // Expiration reached: settle positions
            (true, false, reward + self.episode_pnl * 0.1)
        } else if max_steps_reached {
            (false, true, reward)
        } else {
            (false, false, reward)
        };

        Ok(StepResult {
            observation: self.get_observation(),
            reward: final_reward,
            done,
            truncated,
            info,
        })
    }

    /// Advance order book by n synthetic ticks without trading.
    pub fn run_ticks(&mut self, n: usize) {
        for _ in 0..n {
            self.current_timestamp_ns += 60_000_000_000;
            self.current_tte -= self.config.time_decay_per_tick;
            self.current_tte = self.current_tte.max(0.001);

            for book in &mut self.order_books {
                book.tick(&mut self.rng);
            }
        }
    }

    /// Get the current timestamp in nanoseconds.
    #[getter]
    pub fn timestamp_ns(&self) -> u64 {
        self.current_timestamp_ns
    }

    /// Get the current episode step.
    #[getter]
    pub fn step_count(&self) -> u64 {
        self.episode_step
    }

    /// Get the current time to expiration.
    #[getter]
    pub fn tte(&self) -> f64 {
        self.current_tte
    }

    /// Get the current implied volatility.
    #[getter]
    pub fn iv(&self) -> f64 {
        self.current_iv
    }

    /// Get the underlying price for the first ticker.
    #[getter]
    pub fn underlying_price(&self) -> f64 {
        self.order_books
            .first()
            .map(|b| b.underlying_price)
            .unwrap_or(0.0)
    }

    /// Get the current risk state.
    #[getter]
    pub fn risk(&self) -> RiskState {
        self.risk_state.clone()
    }
}

impl BacktestEngine {
    /// Initialize order books for all configured underlyings.
    fn initialize_order_books(&mut self) -> Result<(), String> {
        self.order_books.clear();

        for (i, ticker) in self.config.tickers.iter().enumerate() {
            let price = self.config.initial_prices.get(i).copied().unwrap_or(100.0);
            let num_strikes = self.config.num_strikes;
            let spacing = self.config.strike_spacing;

            // Generate strikes centered around the current price
            let center_strike = (price / spacing).round() * spacing;
            let half = num_strikes as f64 / 2.0;

            let strikes: Vec<f64> = (0..num_strikes)
                .map(|j| center_strike - half * spacing + j as f64 * spacing)
                .collect();

            // Generate IV smile (higher IV for OTM puts)
            let ivs: Vec<f64> = strikes
                .iter()
                .map(|&k| {
                    let moneyness = (k - price) / price;
                    // IV smile: higher vol for OTM puts
                    self.config.base_iv + 0.1 * (-moneyness).max(0.0) + 0.05 * moneyness.powi(2)
                })
                .collect();

            let book = UnderlyingOrderBook::new(
                ticker,
                price,
                &strikes,
                &ivs,
                self.config.risk_free_rate,
                self.current_tte,
            )?;

            self.order_books.push(book);
        }

        Ok(())
    }

    /// Simulate a potential volatility spike.
    fn simulate_vol_spike(&mut self) {
        let roll: f64 = rand::Rng::gen(&mut self.rng);
        if roll < self.config.vol_spike_prob {
            self.current_iv *= self.config.vol_spike_magnitude;
            // Also spike the order book IVs
            for book in &mut self.order_books {
                for opt_book in &mut book.option_books {
                    opt_book.implied_vol *= self.config.vol_spike_magnitude;
                }
            }
        } else {
            // Mean revert IV back toward base
            self.current_iv += 0.01 * (self.config.base_iv - self.current_iv);
        }
    }

    /// Execute a put credit spread trade.
    ///
    /// Finds appropriate strikes based on the requested width and delta,
    /// prices the spread, and opens the position.
    fn execute_spread_trade(
        &mut self,
        strike_width: f64,
        delta_position: f64,
    ) -> Result<f64, String> {
        let book = self
            .order_books
            .first()
            .ok_or_else(|| "No order books available".to_string())?;

        let underlying_price = book.underlying_price;

        // Find the short strike: closest to the target delta
        // delta_position is normalized [-1, 1], map to actual delta [-0.5, -0.05]
        let target_delta = -0.05 - (delta_position.clamp(-1.0, 1.0) + 1.0) / 2.0 * 0.45;

        // Find the strike with delta closest to target
        let mut best_strike = underlying_price - 5.0; // Default OTM
        let mut best_delta_diff = f64::MAX;

        for opt_book in &book.option_books {
            let greeks = crate::options::put_greeks(
                underlying_price,
                opt_book.strike,
                self.config.risk_free_rate,
                opt_book.implied_vol,
                self.current_tte,
            );

            if let Ok(g) = greeks {
                let diff = (g.delta - target_delta).abs();
                if diff < best_delta_diff {
                    best_delta_diff = diff;
                    best_strike = opt_book.strike;
                }
            }
        }

        // Determine spread width: normalize from [-1, 1] to [2.5, 15.0]
        let actual_width = 2.5 + (strike_width.clamp(-1.0, 1.0) + 1.0) / 2.0 * 12.5;
        let long_strike = best_strike - actual_width;

        // Ensure long strike is positive and reasonable
        if long_strike <= 0.0 {
            return Err("Long strike would be non-positive".to_string());
        }

        // Get IV for the short strike
        let sigma = book
            .book_for_strike(best_strike)
            .map(|b| b.implied_vol)
            .unwrap_or(self.current_iv);

        // Create the spread
        let spread = PutCreditSpread::new(
            underlying_price,
            best_strike,
            long_strike,
            self.config.risk_free_rate,
            sigma,
            self.current_tte,
        )?;

        let credit = spread.net_credit;

        // Open the position (1 contract)
        let ticker = book.ticker.clone();
        self.risk_state.open_position(&ticker, spread, 1)?;

        Ok(credit)
    }

    /// Build the observation vector for the current state.
    pub fn get_observation(&self) -> Vec<f64> {
        let mut obs = Vec::with_capacity(32);

        if let Some(book) = self.order_books.first() {
            // Underlying mid price (normalized)
            obs.push(book.underlying_price / 100.0);

            // Bid-ask spread of underlying options
            if let Some(atm_book) = book.option_books.get(book.option_books.len() / 2) {
                obs.push(atm_book.spread());
            } else {
                obs.push(0.0);
            }

            // OTM put deltas at various strikes (up to 5)
            for opt_book in book.option_books.iter().take(5) {
                let greeks = crate::options::put_greeks(
                    book.underlying_price,
                    opt_book.strike,
                    self.config.risk_free_rate,
                    opt_book.implied_vol,
                    self.current_tte,
                );
                obs.push(greeks.map(|g| g.delta).unwrap_or(0.0));
            }

            // Implied vol surface points (up to 5)
            for opt_book in book.option_books.iter().take(5) {
                obs.push(opt_book.implied_vol);
            }

            // Underlying price for second ticker if available
            if let Some(book2) = self.order_books.get(1) {
                obs.push(book2.underlying_price / 100.0);
            } else {
                obs.push(0.0);
            }
        } else {
            // No order books: fill with zeros
            for _ in 0..12 {
                obs.push(0.0);
            }
        }

        // Time to expiry
        obs.push(self.current_tte);

        // Current IV
        obs.push(self.current_iv);

        // Margin utilization ratio
        obs.push(self.risk_state.margin_utilization());

        // Theta exposure
        obs.push(self.risk_state.total_theta_exposure());

        // Buying power (normalized)
        obs.push(self.risk_state.buying_power / self.config.initial_buying_power);

        // Episode progress
        obs.push(self.episode_step as f64 / self.config.max_episode_steps as f64);

        // Number of positions
        obs.push(self.risk_state.position_count() as f64);

        // Episode P&L (normalized)
        obs.push(self.episode_pnl / 1000.0);

        // Pad to fixed size (20 elements)
        while obs.len() < 20 {
            obs.push(0.0);
        }
        obs.truncate(20);

        obs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> EngineConfig {
        EngineConfig::new(None, None, 100_000.0, 0.05, 0.0833, 100, 42)
    }

    #[test]
    fn test_engine_creation() {
        let config = make_config();
        let engine = BacktestEngine::new(config).unwrap();
        assert_eq!(engine.episode_step, 0);
        assert!(engine.order_books.len() >= 1);
    }

    #[test]
    fn test_engine_reset() {
        let config = make_config();
        let mut engine = BacktestEngine::new(config).unwrap();
        let obs = engine.reset().unwrap();
        assert_eq!(obs.len(), 20);
    }

    #[test]
    fn test_engine_step() {
        let config = make_config();
        let mut engine = BacktestEngine::new(config).unwrap();
        let _ = engine.reset().unwrap();
        let result = engine.step(0.0, 0.0).unwrap();
        assert_eq!(result.observation.len(), 20);
    }
}
