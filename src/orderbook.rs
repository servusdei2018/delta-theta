//! L2 order book simulation with bid-ask spreads.
//!
//! Simulates realistic Level 2 market depth for options chains with 10 levels,
//! synthetic tick updates using random walks with mean reversion, and tracking
//! of multiple option strikes per underlying.

use pyo3::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Number of price levels on each side of the book.
const NUM_LEVELS: usize = 10;

/// A single price level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct PriceLevel {
    /// Price at this level
    #[pyo3(get)]
    pub price: f64,
    /// Quantity (number of contracts) at this level
    #[pyo3(get)]
    pub quantity: u32,
}

/// L2 order book for a single option contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionOrderBook {
    /// Strike price this book represents
    pub strike: f64,
    /// Bid levels (sorted descending by price)
    pub bids: Vec<PriceLevel>,
    /// Ask levels (sorted ascending by price)
    pub asks: Vec<PriceLevel>,
    /// Theoretical mid price (from Black-Scholes)
    pub theo_mid: f64,
    /// Current implied volatility for this strike
    pub implied_vol: f64,
}

impl OptionOrderBook {
    /// Create a new order book centered around a theoretical price.
    ///
    /// # Arguments
    /// * `strike` - The option strike price
    /// * `theo_price` - Theoretical mid price from Black-Scholes
    /// * `implied_vol` - Implied volatility for this strike
    /// * `base_spread` - Base bid-ask spread as a fraction of theo price
    pub fn new(strike: f64, theo_price: f64, implied_vol: f64, base_spread: f64) -> Self {
        let half_spread = (theo_price * base_spread / 2.0).max(0.01);
        let tick_size = 0.01;

        let best_bid = theo_price - half_spread;
        let best_ask = theo_price + half_spread;

        let mut rng = rand::thread_rng();

        let bids: Vec<PriceLevel> = (0..NUM_LEVELS)
            .map(|i| PriceLevel {
                price: (best_bid - i as f64 * tick_size).max(0.01),
                quantity: rng.gen_range(10..200),
            })
            .collect();

        let asks: Vec<PriceLevel> = (0..NUM_LEVELS)
            .map(|i| PriceLevel {
                price: best_ask + i as f64 * tick_size,
                quantity: rng.gen_range(10..200),
            })
            .collect();

        OptionOrderBook {
            strike,
            bids,
            asks,
            theo_mid: theo_price,
            implied_vol,
        }
    }

    /// Get the best bid price.
    pub fn best_bid(&self) -> f64 {
        self.bids.first().map(|l| l.price).unwrap_or(0.0)
    }

    /// Get the best ask price.
    pub fn best_ask(&self) -> f64 {
        self.asks.first().map(|l| l.price).unwrap_or(f64::MAX)
    }

    /// Get the mid price (average of best bid and best ask).
    pub fn mid_price(&self) -> f64 {
        (self.best_bid() + self.best_ask()) / 2.0
    }

    /// Get the bid-ask spread.
    pub fn spread(&self) -> f64 {
        self.best_ask() - self.best_bid()
    }

    /// Apply a synthetic tick update with mean-reverting random walk.
    ///
    /// # Arguments
    /// * `rng` - Random number generator
    /// * `mean_reversion_strength` - How strongly prices revert to theo (0.0 to 1.0)
    /// * `volatility_scale` - Scale of random price movements
    pub fn tick_update<R: Rng>(
        &mut self,
        rng: &mut R,
        mean_reversion_strength: f64,
        volatility_scale: f64,
    ) {
        let normal = Normal::new(0.0, volatility_scale).unwrap_or(Normal::new(0.0, 0.01).unwrap());

        // Mean-reverting price movement
        let current_mid = self.mid_price();
        let reversion = mean_reversion_strength * (self.theo_mid - current_mid);
        let noise: f64 = normal.sample(rng);
        let price_change = reversion + noise;

        // Update all bid levels
        for level in &mut self.bids {
            level.price = (level.price + price_change).max(0.01);
            // Randomly adjust quantities
            let qty_change: i32 = rng.gen_range(-20..20);
            level.quantity = (level.quantity as i32 + qty_change).clamp(1, 500) as u32;
        }

        // Update all ask levels
        for level in &mut self.asks {
            level.price = (level.price + price_change).max(0.01);
            let qty_change: i32 = rng.gen_range(-20..20);
            level.quantity = (level.quantity as i32 + qty_change).clamp(1, 500) as u32;
        }

        // Ensure bid < ask ordering is maintained
        if let (Some(best_bid), Some(best_ask)) = (self.bids.first(), self.asks.first()) {
            if best_bid.price >= best_ask.price {
                let mid = (best_bid.price + best_ask.price) / 2.0;
                if let Some(b) = self.bids.first_mut() {
                    b.price = mid - 0.01;
                }
                if let Some(a) = self.asks.first_mut() {
                    a.price = mid + 0.01;
                }
            }
        }

        // Small IV perturbation
        let iv_noise: f64 = Normal::new(0.0, 0.002)
            .unwrap_or(Normal::new(0.0, 0.001).unwrap())
            .sample(rng);
        self.implied_vol = (self.implied_vol + iv_noise).clamp(0.05, 3.0);
    }
}

/// Full order book state for an underlying, containing books for multiple strikes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderlyingOrderBook {
    /// Ticker symbol
    pub ticker: String,
    /// Current underlying mid price
    pub underlying_price: f64,
    /// Order books for each option strike
    pub option_books: Vec<OptionOrderBook>,
    /// Mean reversion strength parameter
    pub mean_reversion: f64,
    /// Volatility scale for tick updates
    pub vol_scale: f64,
}

impl UnderlyingOrderBook {
    /// Create a new underlying order book with option chains.
    ///
    /// # Arguments
    /// * `ticker` - Symbol name
    /// * `underlying_price` - Current price of the underlying
    /// * `strikes` - List of option strike prices
    /// * `implied_vols` - Implied volatility for each strike
    /// * `r` - Risk-free rate
    /// * `t` - Time to expiration in years
    pub fn new(
        ticker: &str,
        underlying_price: f64,
        strikes: &[f64],
        implied_vols: &[f64],
        r: f64,
        t: f64,
    ) -> Result<Self, String> {
        if strikes.len() != implied_vols.len() {
            return Err("Strikes and implied_vols must have the same length".to_string());
        }

        let mut option_books = Vec::with_capacity(strikes.len());
        for (i, &strike) in strikes.iter().enumerate() {
            let sigma = implied_vols[i];
            let theo_price = crate::options::black_scholes_put(underlying_price, strike, r, sigma, t)?;
            let base_spread = 0.05 + 0.02 * (strike - underlying_price).abs() / underlying_price;
            option_books.push(OptionOrderBook::new(strike, theo_price, sigma, base_spread));
        }

        Ok(UnderlyingOrderBook {
            ticker: ticker.to_string(),
            underlying_price,
            option_books,
            mean_reversion: 0.1,
            vol_scale: 0.005,
        })
    }

    /// Advance all option books by one tick.
    pub fn tick<R: Rng>(&mut self, rng: &mut R) {
        // Update underlying price with random walk
        let underlying_noise: f64 = Normal::new(0.0, self.underlying_price * 0.001)
            .unwrap_or(Normal::new(0.0, 0.01).unwrap())
            .sample(rng);
        self.underlying_price += underlying_noise;
        self.underlying_price = self.underlying_price.max(0.01);

        // Update each option book
        for book in &mut self.option_books {
            book.tick_update(rng, self.mean_reversion, self.vol_scale);
        }
    }

    /// Find the order book for a given strike, if it exists.
    pub fn book_for_strike(&self, strike: f64) -> Option<&OptionOrderBook> {
        self.option_books
            .iter()
            .find(|b| (b.strike - strike).abs() < 1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_order_book_creation() {
        let book = OptionOrderBook::new(100.0, 2.50, 0.25, 0.05);
        assert_eq!(book.bids.len(), NUM_LEVELS);
        assert_eq!(book.asks.len(), NUM_LEVELS);
        assert!(book.best_bid() < book.best_ask());
    }

    #[test]
    fn test_tick_update_maintains_ordering() {
        let mut book = OptionOrderBook::new(100.0, 2.50, 0.25, 0.05);
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            book.tick_update(&mut rng, 0.1, 0.005);
            assert!(book.best_bid() < book.best_ask(), "Bid-ask crossed after tick");
        }
    }

    #[test]
    fn test_underlying_order_book() {
        let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
        let ivs = vec![0.30, 0.27, 0.25, 0.27, 0.30];
        let book = UnderlyingOrderBook::new("TEST", 100.0, &strikes, &ivs, 0.05, 0.1).unwrap();
        assert_eq!(book.option_books.len(), 5);
        assert!(book.book_for_strike(100.0).is_some());
    }
}
