import math
import random

from fbe.calibrate import apply_platt_fit, clamp, fit_platt, fit_platt_for_type, logit, sigmoid


def test_apply_platt_fit_identity_is_passthrough():
    from fbe.calibrate import PlattFit

    identity = PlattFit(a=1.0, b=0.0, n=0, identity=True)
    for p in (0.1, 0.5, 0.9):
        assert math.isclose(apply_platt_fit(identity, p), p, abs_tol=1e-9)


def test_apply_platt_fit_matches_formula():
    from fbe.calibrate import PlattFit

    fit = PlattFit(a=2.0, b=-0.3, n=100, identity=False)
    p = 0.7
    expected = sigmoid(2.0 * logit(clamp(p)) - 0.3)
    assert math.isclose(apply_platt_fit(fit, p), expected, abs_tol=1e-12)


def test_fit_platt_recovers_known_a_b_on_synthetic_data():
    true_a, true_b = 2.0, -0.5
    rng = random.Random(1234)
    points = []
    for _ in range(3000):
        raw_p = rng.uniform(0.02, 0.98)
        z = true_a * logit(raw_p) + true_b
        outcome = 1 if rng.random() < sigmoid(z) else 0
        points.append((raw_p, outcome))

    fit = fit_platt(points)
    assert abs(fit.a - true_a) < 0.25
    assert abs(fit.b - true_b) < 0.25
    assert not fit.identity


def test_fit_platt_identity_below_minimum_rows():
    points = [(0.9, 1)] * 10 + [(0.1, 0)] * 10  # n=20 < 50
    fit, note = fit_platt_for_type(points)
    assert fit.identity
    assert fit.a == 1.0 and fit.b == 0.0
    assert "identity" in note
    assert "20" in note


def test_fit_platt_for_type_fits_when_enough_rows():
    rng = random.Random(7)
    points = []
    for _ in range(200):
        raw_p = rng.uniform(0.05, 0.95)
        outcome = 1 if rng.random() < sigmoid(1.5 * logit(raw_p) + 0.2) else 0
        points.append((raw_p, outcome))
    fit, note = fit_platt_for_type(points)
    assert not fit.identity
    assert "fit on 200" in note
