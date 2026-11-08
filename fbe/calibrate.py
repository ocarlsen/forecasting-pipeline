"""Platt scaling: calibrated p = sigmoid(a * logit(raw p) + b).

Damped (Armijo-backtracked) Newton's method on an L2-penalized log-likelihood,
with the penalty pulling (a, b) toward the IDENTITY (a=1, b=0) rather than toward (0, 0): a
starved fit should decay toward "trust the raw model," not collapse to a constant. Fit
separately for market and dataset questions; raw p is clipped to [0.01, 0.99] before the logit
so a single training point with a flush 0 or 1 doesn't dominate or break the fit.

Below 50 training rows for a type, BUILD_SPEC.md says Platt is the identity for that type --
reported as such, not silently fit on too little data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

CALIBRATOR_VERSION = "platt-l2-v1-py"
CLAMP_MIN = 0.01
CLAMP_MAX = 0.99
LAMBDA = 4.0
MAX_ITERATIONS = 50
GRADIENT_TOLERANCE = 1e-7
ARMIJO_C1 = 1e-4
BACKTRACK_SHRINK = 0.5
MAX_BACKTRACK_STEPS = 40
MIN_TRAINING_ROWS = 50


def clamp(p: float) -> float:
    return min(CLAMP_MAX, max(CLAMP_MIN, p))


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class PlattFit:
    a: float
    b: float
    n: int
    identity: bool  # True if this fit is the untrained a=1,b=0 identity (n==0, or n<50 override)


def apply_platt_fit(fit: PlattFit, raw_p: float) -> float:
    """Apply a fit to one raw probability -- the only place this formula is implemented, so a
    fit is always applied exactly how it was fit."""
    return sigmoid(fit.a * logit(clamp(raw_p)) + fit.b)


def _mean_objective(a: float, b: float, x: list[float], y: list[float]) -> float:
    n = len(x)
    total = 0.0
    for xi, yi in zip(x, y):
        z = a * xi + b
        log_one_plus_exp_z = z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))
        total += log_one_plus_exp_z - yi * z
    n_eff = max(n, 1)
    return total / n_eff + (LAMBDA / (2 * n_eff)) * ((a - 1) ** 2 + b**2)


def fit_platt(points: list[tuple[float, int]]) -> PlattFit:
    """points: list of (raw_p, outcome in {0,1}). No threshold enforcement here -- see
    fit_platt_for_type() for the <50-rows-means-identity policy."""
    n = len(points)
    x = [logit(clamp(p)) for p, _ in points]
    y = [float(o) for _, o in points]
    n_eff = max(n, 1)

    a, b = 1.0, 0.0
    converged = n == 0

    for _ in range(MAX_ITERATIONS):
        if converged:
            break
        grad_a = LAMBDA * (a - 1)
        grad_b = LAMBDA * b
        h_aa = LAMBDA
        h_bb = LAMBDA
        h_ab = 0.0
        for xi, yi in zip(x, y):
            z = a * xi + b
            pred = sigmoid(z)
            err = pred - yi
            grad_a += err * xi
            grad_b += err
            w = pred * (1 - pred)
            h_aa += w * xi * xi
            h_bb += w
            h_ab += w * xi
        grad_a /= n_eff
        grad_b /= n_eff
        h_aa /= n_eff
        h_bb /= n_eff
        h_ab /= n_eff

        if math.hypot(grad_a, grad_b) < GRADIENT_TOLERANCE:
            converged = True
            break

        det = h_aa * h_bb - h_ab * h_ab
        delta_a = (h_bb * grad_a - h_ab * grad_b) / det
        delta_b = (h_aa * grad_b - h_ab * grad_a) / det

        current = _mean_objective(a, b, x, y)
        directional_derivative = -(grad_a * delta_a + grad_b * delta_b)
        t = 1.0
        for _step in range(MAX_BACKTRACK_STEPS):
            nxt = _mean_objective(a - t * delta_a, b - t * delta_b, x, y)
            if nxt <= current + ARMIJO_C1 * t * directional_derivative:
                break
            t *= BACKTRACK_SHRINK
        a -= t * delta_a
        b -= t * delta_b

    if not converged:
        raise RuntimeError(
            f"Platt fit did not converge in {MAX_ITERATIONS} iterations (n={n}); "
            "this should be unreachable given the objective's strict convexity"
        )
    if not (math.isfinite(a) and math.isfinite(b)):
        raise RuntimeError(f"Platt fit produced non-finite (a={a}, b={b})")
    return PlattFit(a=a, b=b, n=n, identity=(n == 0))


def fit_platt_for_type(points: list[tuple[float, int]]) -> tuple[PlattFit, str]:
    """Returns (fit, note). Below MIN_TRAINING_ROWS, the fit is forced to identity -- reported
    as such -- rather than fit on too little data to trust."""
    n = len(points)
    if n < MIN_TRAINING_ROWS:
        return (
            PlattFit(a=1.0, b=0.0, n=n, identity=True),
            f"identity (only {n} training rows, below the {MIN_TRAINING_ROWS}-row minimum)",
        )
    fit = fit_platt(points)
    return fit, f"fit on {n} training rows (a={fit.a:.4f}, b={fit.b:.4f})"
