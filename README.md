# Quantitative Derivatives Monte Carlo Lab

A modular Python project implementing Monte Carlo simulation for lognormal (GBM) dynamics, validation against Black–Scholes, and pricing of a bespoke up-and-in Asian option. The code is written to be clean, vectorized, and easy to extend (e.g., term structures, stochastic volatility, variance reduction).

---

## 1. Model Overview

Under the risk-neutral measure, the underlying follows geometric Brownian motion (GBM):

\[
dS_t = (r - q) S_t \, dt + \sigma S_t \, dW_t
\]

with baseline parameters:

- \(S_0 = 100\)
- \(r = 3\%\)
- \(q = 0\%\)
- \(\sigma = 20\%\)
- \(T = 1\) year

Paths are simulated using the **exact discretization**:

\[
S_{t+\Delta t} = S_t \exp\Big((r - q - 0.5 \sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\Big),
\]

with \(Z \sim \mathcal{N}(0, 1)\).

---

## 2. File Structure

- `quant_mc_test.py`  
  Core module containing:
  - GBM path generator (`simulate_gbm_paths`)
  - Monte Carlo pricing for European calls (`price_european_call_mc`)
  - Black–Scholes closed-form call price (`bs_call_price`)
  - Convergence study helper (`convergence_study`)
  - Up-and-in arithmetic-average Asian call pricer (`price_up_and_in_asian_call_mc`)
  - Utility functions for standard error and confidence intervals

---

## 3. Installation

```bash
git clone https://github.com/nmadagi/quantitative-derivatives-monte-carlo-lab.git
cd quantitative-derivatives-monte-carlo-lab
pip install -r requirements.txt
