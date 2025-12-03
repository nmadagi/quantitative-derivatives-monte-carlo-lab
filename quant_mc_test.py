"""
quant_mc_test.py

Monte Carlo + Black Scholes toolkit for:
- GBM path simulation
- European call pricing (MC vs BS)
- Convergence study utilities
- Up-and-in Asian arithmetic-average Asian call

Baseline risk-neutral GBM:
    dS_t = (r - q) S_t dt + sigma S_t dW_t
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt, erf
from time import perf_counter
from typing import Tuple, Optional, Iterable

import numpy as np


# ============================================================
# Helpers: statistics
# ============================================================

def mc_standard_error(samples: np.ndarray) -> float:
    """
    Compute standard error of the mean for a 1D array of samples.
    """
    n = samples.shape[0]
    if n < 2:
        return 0.0
    return samples.std(ddof=1) / sqrt(n)


def mc_confidence_interval(
    mean: float,
    std_error: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Symmetric (1 - alpha) confidence interval using normal approximation.
    For alpha = 0.05 -> 95% CI (mean ± 1.96 * SE).
    """
    # For 95% CI we use 1.96; for general alpha we could invert Phi,
    # but here a constant is enough (95% is what the assignment wants).
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    return mean - z * std_error, mean + z * std_error


# ============================================================
# 1. GBM Path Generator (Task 1A)
# ============================================================

def simulate_gbm_paths(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int] = None,
    include_initial: bool = True,
) -> np.ndarray:
    """
    Simulate GBM price paths under the risk-neutral measure using the
    exact discretization:

        S_{t+Δt} = S_t * exp((r - q - 0.5 * σ^2) * Δt + σ * sqrt(Δt) * Z)

    Parameters
    ----------
    S0 : float
        Initial price.
    r : float
        Continuously compounded risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity in years.
    n_paths : int
        Number of Monte Carlo paths (N).
    n_steps : int
        Number of time steps (M).
    seed : Optional[int]
        Random seed for reproducibility. If None, RNG is not seeded.
    include_initial : bool
        If True, returned array has shape (N, M+1) and includes S0 at t=0
        as the first column. If False, shape is (N, M) and starts at t=Δt.

    Returns
    -------
    paths : np.ndarray
        Simulated paths of shape (n_paths, n_steps + 1) if include_initial,
        otherwise (n_paths, n_steps).
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    # Draw all standard normals at once for vectorization
    z = rng.standard_normal(size=(n_paths, n_steps))

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * sqrt(dt) * z

    # log S evolution: log S_t = log S0 + cumsum(drift + diffusion)
    log_S0 = log(S0)
    log_increments = drift + diffusion
    log_paths = log_S0 + np.cumsum(log_increments, axis=1)
    paths = np.exp(log_paths)

    if include_initial:
        S0_column = np.full((n_paths, 1), S0, dtype=float)
        paths = np.concatenate([S0_column, paths], axis=1)

    return paths


# ============================================================
# 2. Black–Scholes Closed-Form (Task 2B)
# ============================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
) -> float:
    """
    Black–Scholes price of a European call option.

    Parameters
    ----------
    S0 : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    q : float
        Continuous dividend yield.
    sigma : float
        Volatility.
    T : float
        Time to maturity in years.
    """
    if T <= 0 or sigma <= 0:
        # Handle trivial edge cases
        return max(S0 * exp(-q * T) - K * exp(-r * T), 0.0)

    sqrtT = sqrt(T)
    d1 = (log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    discounted_S0 = S0 * exp(-q * T)
    discounted_K = K * exp(-r * T)

    return discounted_S0 * Nd1 - discounted_K * Nd2


# ============================================================
# 3. European Call via Monte Carlo (Task 2A)
# ============================================================

@dataclass
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int
    runtime_sec: float


def price_european_call_mc(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> MCResult:
    """
    Monte Carlo pricing of a European call using GBM paths.

    Returns discounted price estimate, standard error, 95% CI, and runtime.
    """
    start = perf_counter()

    paths = simulate_gbm_paths(
        S0=S0,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        include_initial=True,
    )
    # S_T is the last column
    ST = paths[:, -1]
    payoff = np.maximum(ST - K, 0.0)

    disc_factor = exp(-r * T)
    discounted_payoff = disc_factor * payoff

    price = discounted_payoff.mean()
    se = mc_standard_error(discounted_payoff)
    ci_low, ci_high = mc_confidence_interval(price, se)

    runtime = perf_counter() - start

    return MCResult(
        price=price,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paths=n_paths,
        n_steps=n_steps,
        runtime_sec=runtime,
    )


# ============================================================
# 4. Convergence Study Helper (Task 2C)
# ============================================================

def convergence_study(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_steps: int,
    path_list: Iterable[int],
    seed: Optional[int] = 1234,
):
    """
    Run a convergence study for a single strike K using different
    path counts. Returns a list of MCResult plus the BS price.
    """
    bs_price = bs_call_price(S0, K, r, q, sigma, T)
    results = []

    # If we want reproducibility across N, we can vary the seed or not.
    for i, n_paths in enumerate(path_list):
        # Optionally vary seed slightly per run
        run_seed = None if seed is None else seed + i
        res = price_european_call_mc(
            S0=S0,
            K=K,
            r=r,
            q=q,
            sigma=sigma,
            T=T,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=run_seed,
        )
        results.append(res)

    return bs_price, results


# ============================================================
# 5. Up-and-In Asian Call (Task 3A)
# ============================================================

@dataclass
class AsianBarrierResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    n_steps: int
    runtime_sec: float
    barrier_activation_pct: float


def price_up_and_in_asian_call_mc(
    S0: float,
    K: float,
    H: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: Optional[int] = None,
) -> AsianBarrierResult:
    """
    Monte Carlo pricing of an up-and-in Asian call with arithmetic average.

        Barrier H (up-and-in):
            if max_t S_t >= H:
                payoff = max(average(S_t) - K, 0)
            else:
                payoff = 0

    Discrete monitoring at the simulated time grid.

    Returns price, SE, 95% CI, runtime, and % of paths that hit the barrier.
    """
    start = perf_counter()

    paths = simulate_gbm_paths(
        S0=S0,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        include_initial=True,
    )

    # Barrier check (max over time dimension)
    max_path = paths.max(axis=1)
    barrier_hit = max_path >= H

    # Arithmetic average over time dimension
    # (includes S0; you can also exclude t=0 if you prefer – but be consistent)
    S_avg = paths.mean(axis=1)

    # Payoff only for activated paths
    intrinsic = np.maximum(S_avg - K, 0.0)
    payoff = np.where(barrier_hit, intrinsic, 0.0)

    disc_factor = exp(-r * T)
    discounted_payoff = disc_factor * payoff

    price = discounted_payoff.mean()
    se = mc_standard_error(discounted_payoff)
    ci_low, ci_high = mc_confidence_interval(price, se)
    runtime = perf_counter() - start

    activation_pct = float(barrier_hit.mean()) * 100.0

    return AsianBarrierResult(
        price=price,
        std_error=se,
        ci_low=ci_low,
        ci_high=ci_high,
        n_paths=n_paths,
        n_steps=n_steps,
        runtime_sec=runtime,
        barrier_activation_pct=activation_pct,
    )


# ============================================================
# Example driver (optional, not required by assignment)
# ============================================================

if __name__ == "__main__":
    # Baseline parameters
    S0 = 100.0
    r = 0.03
    q = 0.0
    sigma = 0.20
    T = 1.0
    n_steps = 52  # weekly

    strikes = [70, 80, 90, 100, 110, 120, 130]
    n_paths = 50_000

    print("=== MC vs Black–Scholes (European Call) ===")
    for K in strikes:
        mc_res = price_european_call_mc(
            S0, K, r, q, sigma, T,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=42,
        )
        bs = bs_call_price(S0, K, r, q, sigma, T)
        abs_err = abs(mc_res.price - bs)
        rel_err = abs_err / bs if bs != 0 else float("nan")
        inside_ci = mc_res.ci_low <= bs <= mc_res.ci_high

        print(
            f"K={K:3.0f} | MC={mc_res.price:8.4f} "
            f"(CI [{mc_res.ci_low:8.4f}, {mc_res.ci_high:8.4f}]) | "
            f"BS={bs:8.4f} | abs_err={abs_err:7.4f} "
            f"rel_err={rel_err:7.4f} | BS in CI? {inside_ci}"
        )

    print("\n=== Convergence Study (K=100) ===")
    bs_price, conv_results = convergence_study(
        S0=S0,
        K=100.0,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        path_list=[2_000, 20_000, 200_000],
        seed=1234,
    )
    print(f"Black–Scholes reference price (K=100): {bs_price:.6f}")
    for res in conv_results:
        width = res.ci_high - res.ci_low
        print(
            f"N={res.n_paths:7d} | MC={res.price:10.6f} | "
            f"SE={res.std_error:10.6f} | CI width={width:10.6f} | "
            f"runtime={res.runtime_sec:6.3f}s"
        )

    print("\n=== Up-and-In Asian Call (K=100, H=120) ===")
    asian_res = price_up_and_in_asian_call_mc(
        S0=S0,
        K=100.0,
        H=120.0,
        r=r,
        q=q,
        sigma=sigma,
        T=T,
        n_paths=100_000,
        n_steps=n_steps,
        seed=2025,
    )
    print(
        f"Asian up-and-in price={asian_res.price:.6f} | "
        f"SE={asian_res.std_error:.6f} | "
        f"95% CI=[{asian_res.ci_low:.6f}, {asian_res.ci_high:.6f}] | "
        f"barrier activation={asian_res.barrier_activation_pct:.2f}% | "
        f"runtime={asian_res.runtime_sec:.3f}s"
    )
