import csv
import json
from pathlib import Path

import numpy as np
import pytest

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


def test_simulation_diagnostic_cache_replaces_old_draw_count(tmp_path, monkeypatch):
    import Simulation_Analysis as simulation

    checkpoint = tmp_path / "chain.pkl"
    workflow_runtime.atomic_pickle_dump({
        "result": {"backend": "numba", "samples": {"k": np.zeros(20_000)}},
    }, checkpoint)
    calls = []

    def sample(run, data, **kwargs):
        calls.append(kwargs)
        n = kwargs["iterations"]
        return {"backend": "numba", "samples": {
            "k": np.zeros(n), "taus": np.zeros((n, 10)), "theta": np.zeros((n, 11)),
            "p_t": np.zeros((n, len(data["cases"]))),
        }}

    monkeypatch.setattr(simulation, "_configure_project", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulation, "call_rjmcmc", sample)
    state = {"k": 0, "taus": [], "theta_values": [-5.5]}
    task = ("No Change", 0, state, {"seed": 2025, "backend": "numba"}, 31, checkpoint, True)
    assert simulation._diagnostic_worker(task).startswith("completed:")
    assert simulation._diagnostic_worker(task).startswith("cached:")
    assert len(calls) == 1
    assert calls[0]["burn_in"] == 5000
    assert calls[0]["seed"] == 753464599
    record = workflow_runtime.load_pickle(checkpoint)
    assert record["signature"]["iterations"] == 31
    assert record["signature"]["initial_state"] == state
    assert simulation.build_parser().parse_args([]).diagnostic_iterations == 200_000

    monkeypatch.setattr(methods, "c_backend_available", lambda: False)
    c_task = (*task[:3], {"seed": 2025, "backend": "c"}, *task[4:])
    with pytest.raises(RuntimeError, match="C backend is required"):
        simulation._diagnostic_worker(c_task)


def test_simulation_gate_keeps_unreported_daily_coordinate(tmp_path, monkeypatch):
    import config
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    import Simulation_Analysis as simulation

    monkeypatch.setattr(config, "SCENARIOS", {"No Change": config.SCENARIOS["No Change"]})
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(PdfPages, "savefig", lambda *args, **kwargs: None)

    def create_checkpoints(tasks, *unused):
        for _, chain, initial, _, draws, checkpoint, _ in tasks:
            rng = np.random.default_rng(chain)
            p_t = np.repeat(rng.uniform(size=(draws, 1)), 200, axis=1)
            p_t[:, 72] = 0.125
            workflow_runtime.atomic_pickle_dump({
                "chain": chain, "initial_state": initial, "runtime_seconds": 0.0,
                "result": {"samples": {"p_t": p_t, "k": rng.integers(0, 10, size=draws)}},
            }, checkpoint)

    monkeypatch.setattr(simulation, "run_parallel", create_checkpoints)
    args = simulation.build_parser().parse_args(["--diagnostic-iterations", "101"])
    simulation.run_diagnostics(args, {"results_root": tmp_path, "plots_dir": tmp_path})
    root = tmp_path / "reviewer_simulation_diagnostics"
    gate = json.loads((root / "convergence_gate.json").read_text())
    assert gate["monitored_coordinates"] == gate["expected_coordinates"] == 202
    assert gate["passed"] is False
    with (root / "coordinate_diagnostics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    daily = [row for row in rows if row["variable"] == "p_t"]
    assert [int(row["day"]) for row in daily] == list(range(1, 201))
    assert np.isnan(float(daily[72]["rank_normalized_rhat"]))


def test_figure6_reference_draws_without_fits(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    import Realdata_Analysis_JP as japan

    reference = tmp_path / "reference_r2.json"
    fits = [{"p_g": pg, "pip": [0.0] * 699 + [pg], "max_pip": pg,
             "modal_locations": list(range(2, 12)), "cap_mass": 0.9}
            for pg in (0.1, 0.3, 0.5, 0.7, 0.9)]
    reference.write_text(json.dumps({"figure6": {"cap": 10, "days": 700, "fits": fits}}))
    results, plots = tmp_path / "results", tmp_path / "plots"

    def no_fit(*args, **kwargs):
        pytest.fail("Figure 6 redraw attempted to access the fitting workflow")

    calls = []

    def inspect(figure, path, **kwargs):
        calls.append(path.suffix)
        heatmap, summaries = figure.axes[:2]
        image = heatmap.images[0]
        assert image.get_cmap().name == "Greys"
        assert image.get_clim() == (0.0, 1.0)
        assert image.get_array().shape == (5, 700)
        np.testing.assert_array_equal(heatmap.collections[0].get_offsets()[:, 0], np.arange(3, 13))
        np.testing.assert_array_equal(heatmap.collections[0].get_facecolors()[0], [1, 1, 1, 1])
        np.testing.assert_array_equal(heatmap.collections[0].get_edgecolors()[0], [0, 0, 0, 1])
        assert [line.get_color() for line in summaries.lines] == ["#C44E52", "#4C72B0"]
        assert [line.get_marker() for line in summaries.lines] == ["o", "s"]

    monkeypatch.setattr(japan, "_runtime_paths", no_fit)
    monkeypatch.setattr(japan, "_load_data", no_fit)
    monkeypatch.setattr(japan, "run_parallel", no_fit)
    monkeypatch.setattr(japan, "load_pickle", no_fit)
    monkeypatch.setattr(Figure, "savefig", inspect)
    japan.main(["--pg-reference", str(reference), "--data", str(tmp_path / "absent.csv"),
                "--results-root", str(results), "--plots-dir", str(plots)])
    assert calls == [".png", ".pdf"]
    csv_path = results / "realdata/pg_sensitivity/p_g_sensitivity.csv"
    assert "Historical R1 sensitivity snapshot for the R2 redraw" in csv_path.read_text()


def test_reviewer_only_runs_diagnostics_then_figure6_reference(tmp_path, monkeypatch):
    import Realdata_Analysis_JP as japan

    reference = tmp_path / "reference_r2.json"
    runtime = {"realdata_root": tmp_path / "realdata", "plots_dir": tmp_path / "plots"}
    loaded_data = {"cases": [1], "deaths": [0]}
    calls = []

    def diagnostics(data, paths, args):
        assert data is loaded_data and paths is runtime
        assert args.backend == "numba-fast"
        assert (args.chains, args.chain_iterations, args.chain_burn_in, args.chain_storage_stride) == (
            4, 20_000_000, 2_000_000, 200
        )
        calls.append("diagnostics")

    def redraw(data, paths, args):
        assert data is loaded_data and paths is runtime
        assert args.pg_reference == reference
        calls.append("figure6")

    def no_main_fit(*args, **kwargs):
        pytest.fail("Reviewer-only reproduction attempted a main analysis fit")

    monkeypatch.setattr(japan, "_runtime_paths", lambda args: runtime)
    monkeypatch.setattr(japan, "_configure_project", lambda *args: None)
    monkeypatch.setattr(japan, "_load_data", lambda *args: (None, loaded_data))
    monkeypatch.setattr(japan, "run_reviewer_diagnostics", diagnostics)
    monkeypatch.setattr(japan, "run_pg_sensitivity", redraw)
    monkeypatch.setattr(japan, "_fit_method", no_main_fit)

    japan.main(["--reviewer-only", "--backend", "numba-fast", "--pg-reference", str(reference)])
    assert calls == ["diagnostics", "figure6"]
