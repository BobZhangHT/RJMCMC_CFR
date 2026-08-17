from pathlib import Path

import methods
import workflow_runtime


def test_workflows_share_the_canonical_rjmcmc_entrypoint():
    assert workflow_runtime.load_rjmcmc() is methods.run_rjmcmc

    root = Path(__file__).resolve().parents[1]
    for script_name in ("Simulation_Analysis.py", "Realdata_Analysis_JP.py"):
        source = (root / script_name).read_text(encoding="utf-8")
        assert "load_rjmcmc()" in source
        assert "_rjmcmc_sampler_numba" not in source
        assert "_rjmcmc_sampler_c" not in source
