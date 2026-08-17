import numpy as np
import pytest

import methods


def _small_data(T=32):
    cases = np.full(T, 1000.0)
    delay = np.exp(-np.arange(T, dtype=float) / 5.0)
    delay /= delay.sum()
    expected = np.convolve(cases * 0.04, delay, mode="full")[:T]
    rng = np.random.default_rng(1234)
    return {"cases": cases, "deaths": rng.poisson(expected).astype(float)}


def test_boundary_updates_are_possible_and_diagnostics_are_exposed():
    data = _small_data()
    result = methods.run_rjmcmc(
        data, backend="numba", seed=7, iterations=500, burn_in=100,
        initial_state=(0, [], [-3.5]), diagnostics=True,
    )
    diagnostics = result["sampler_diagnostics"]
    update_idx = diagnostics["move_names"].index("update")
    assert diagnostics["proposed"][update_idx] > 0
    assert np.isfinite(result["p_t_hat"]).all()

    max_state = (methods.K_MAX, list(range(2, methods.K_MAX + 2)),
                 [-3.5] * (methods.K_MAX + 1))
    result_max = methods.run_rjmcmc(
        data, backend="numba", seed=8, iterations=100, burn_in=20,
        initial_state=max_state, diagnostics=True,
    )
    assert result_max["sampler_diagnostics"]["proposed"][update_idx] > 0


@pytest.mark.skipif(not methods.c_backend_available(), reason="C extension not built")
def test_backends_have_statistically_consistent_posterior_summaries():
    data = _small_data()
    numba_k, c_k, numba_pip, c_pip = [], [], [], []
    numba_runs, c_runs = [], []
    for seed in range(5):
        n = methods.run_rjmcmc(data, backend="numba", seed=seed,
                               iterations=1200, burn_in=300,
                               return_samples=True)
        c = methods.run_rjmcmc(data, backend="c", seed=seed,
                               iterations=1200, burn_in=300,
                               return_samples=True)
        numba_runs.append(n)
        c_runs.append(c)
        numba_k.append(n["k_est"])
        c_k.append(c["k_est"])
        numba_pip.append(n["pip_array"])
        c_pip.append(c["pip_array"])

    numba_k_draws = np.concatenate([run["samples"]["k"] for run in numba_runs])
    c_k_draws = np.concatenate([run["samples"]["k"] for run in c_runs])
    bins = np.arange(methods.K_MAX + 2) - 0.5
    numba_k_mass = np.histogram(numba_k_draws, bins=bins, density=True)[0]
    c_k_mass = np.histogram(c_k_draws, bins=bins, density=True)[0]
    k_total_variation = 0.5 * np.abs(numba_k_mass - c_k_mass).sum()

    numba_p_t = np.mean([run["p_t_hat"] for run in numba_runs], axis=0)
    c_p_t = np.mean([run["p_t_hat"] for run in c_runs], axis=0)
    numba_lower = np.mean([run["p_t_lower_ci"] for run in numba_runs], axis=0)
    c_lower = np.mean([run["p_t_lower_ci"] for run in c_runs], axis=0)
    numba_upper = np.mean([run["p_t_upper_ci"] for run in numba_runs], axis=0)
    c_upper = np.mean([run["p_t_upper_ci"] for run in c_runs], axis=0)

    # The kernels use different random-number generators, so consistency is
    # assessed at the posterior-distribution level rather than draw by draw.
    assert abs(np.mean(numba_k) - np.mean(c_k)) <= 1.0
    assert abs(numba_k_draws.mean() - c_k_draws.mean()) <= 0.5
    assert k_total_variation <= 0.15
    assert np.mean(np.abs(np.mean(numba_pip, axis=0) - np.mean(c_pip, axis=0))) <= 0.08
    assert np.mean(np.abs(numba_p_t - c_p_t)) <= 0.012
    assert np.mean(np.abs(numba_lower - c_lower)) <= 0.02
    assert np.mean(np.abs(numba_upper - c_upper)) <= 0.02


@pytest.mark.parametrize("backend", ["numba", "c"])
def test_backends_support_realdata_tuning_controls(backend):
    if backend == "c" and not methods.c_backend_available():
        pytest.skip("C extension not built")
    result = methods.run_rjmcmc(
        _small_data(),
        backend=backend,
        seed=19,
        iterations=120,
        burn_in=30,
        theta_prop_sigma=0.2,
        u_prop_sigma=0.1,
        move_window=4,
        global_move_prob=0.1,
        max_k=5,
        diagnostics=True,
        return_samples=True,
    )
    assert result["samples"]["taus"].shape == (120, 5)
    assert np.isfinite(result["p_t_hat"]).all()
    assert np.sum(result["sampler_diagnostics"]["proposed"]) > 0


def test_explicit_c_backend_fails_with_build_hint_when_unavailable():
    if methods.c_backend_available():
        pytest.skip("C extension is available")
    with pytest.raises(RuntimeError, match="build_ext --inplace"):
        methods.run_rjmcmc(_small_data(), backend="c", iterations=1, burn_in=0)
