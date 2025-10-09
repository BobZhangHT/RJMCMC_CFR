from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from methods import get_rtacfr_signal


def test_get_rtacfr_signal_creates_cache_dir(tmp_path, monkeypatch):
    data = {
        "cases": np.array([10.0, 12.0, 11.0]),
        "deaths": np.array([1.0, 0.0, 2.0]),
    }

    def fake_solver(_):
        return np.array([0.1, 0.2, 0.3])

    monkeypatch.setattr("methods._run_rtacfr_fusedlasso_internal", fake_solver)

    cache_dir = tmp_path / "rtacfr_cache"
    monkeypatch.setattr("methods.SIGNAL_CACHE_DIR", str(cache_dir), raising=False)

    result = get_rtacfr_signal(data, "Demo", rep_idx=0)

    expected = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(result, expected)

    cache_file = cache_dir / "signal_scen=Demo_rep=0.npz"
    assert cache_file.exists(), "Cache file should be created on first call."

    call_count = {"n": 0}

    def failing_solver(_):
        call_count["n"] += 1
        raise AssertionError("Solver should not be called when cache exists")

    monkeypatch.setattr("methods._run_rtacfr_fusedlasso_internal", failing_solver)

    cached = get_rtacfr_signal(data, "Demo", rep_idx=0)
    np.testing.assert_allclose(cached, expected)
    assert call_count["n"] == 0
