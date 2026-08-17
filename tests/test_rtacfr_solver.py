import builtins

import numpy as np
from scipy.stats import gamma

import methods


def _constant_signal_data(t=32, level=0.04):
    cases = np.full(t, 1000.0)
    template = {"cases": cases, "deaths": np.zeros(t)}
    _, _, _, xmat = methods._rtacfr_design_matrix(template)
    return {"cases": cases, "deaths": xmat @ np.full(t, level)}, xmat


def test_default_admm_is_finite_bounded_and_does_not_import_cvxpy(monkeypatch):
    data, _ = _constant_signal_data(t=24)
    real_import = builtins.__import__

    def reject_cvxpy(name, *args, **kwargs):
        if name == "cvxpy" or name.startswith("cvxpy."):
            raise AssertionError("default ADMM path must not import cvxpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_cvxpy)
    first = methods._run_rtacfr_fusedlasso_internal(data)
    second = methods._run_rtacfr_fusedlasso_internal(data, solver="admm")

    assert np.isfinite(first).all()
    assert np.all((first >= 0.0) & (first <= 1.0))
    np.testing.assert_allclose(first, second, rtol=0.0, atol=1e-12)


def test_admm_reports_stable_primal_and_dual_residuals():
    data, xmat = _constant_signal_data(t=24)
    dt = data["deaths"]
    difference = methods._rtacfr_difference_matrix(len(dt))
    x_scale = np.sum(xmat ** 2) / len(dt)
    rho = max(1.0, 0.05 * float(x_scale))
    normal_matrix = 2.0 * (xmat.T @ xmat)
    normal_matrix += rho * (difference.T @ difference + np.eye(len(dt)))
    factor = methods.linalg.cho_factor(normal_matrix, lower=True, check_finite=False)

    _, _, diagnostics = methods._rtacfr_admm_solve(
        xmat, dt, difference, factor, rho, lambda_val=1.0
    )

    assert diagnostics["iterations"] <= 5000
    assert diagnostics["converged"]
    assert diagnostics["primal_residual"] < 1e-3
    assert diagnostics["dual_residual"] < 1e-3
    assert np.isfinite(diagnostics["objective_history"]).all()


def test_constant_signal_is_reasonably_recovered():
    data, _ = _constant_signal_data(t=32, level=0.04)
    estimate = methods._run_rtacfr_fusedlasso_internal(data)

    assert np.all((estimate >= 0.0) & (estimate <= 1.0))
    assert abs(float(np.mean(estimate)) - 0.04) < 0.005


def test_cache_tags_keep_assumed_delays_separate(tmp_path, monkeypatch):
    data, _ = _constant_signal_data(t=20)
    monkeypatch.setattr(methods, "SIGNAL_CACHE_DIR", str(tmp_path))
    short_delay = gamma(a=2.0, scale=4.0)
    long_delay = gamma(a=2.0, scale=12.0)

    short_signal = methods.get_rtacfr_signal(
        data, "cache test", 0, delay_dist=short_delay, cache_tag="short"
    )
    long_signal = methods.get_rtacfr_signal(
        data, "cache test", 0, delay_dist=long_delay, delay_name="long"
    )
    files = list(tmp_path.glob("*.npz"))

    assert short_signal.shape == long_signal.shape == (20,)
    assert len(files) == 2
    assert any("short" in file.name for file in files)
    assert any("long" in file.name for file in files)

    cached_short = methods.get_rtacfr_signal(
        data, "cache test", 0, delay_dist=short_delay, cache_tag="short"
    )
    np.testing.assert_allclose(short_signal, cached_short)
    assert len(list(tmp_path.glob("*.npz"))) == 2
