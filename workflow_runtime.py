"""Small runtime helpers shared by the converted analysis scripts.

The helpers deliberately depend only on the Python standard library and keep
all transient artifacts in the caller-selected, ignored output directories.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import multiprocessing as mp
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

from tqdm import tqdm


T = TypeVar("T")
R = TypeVar("R")


def load_rjmcmc():
    """Return the single canonical RJMCMC entry point used by all workflows."""
    from methods import run_rjmcmc

    return run_rjmcmc


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a reproducible 32-bit seed for a logical task."""
    payload = repr((int(base_seed), parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def call_rjmcmc(
    run_fn: Callable[..., R],
    data: Any,
    *,
    p_geom: float | None = None,
    theta_sigma: float | None = None,
    seed: int | None = None,
    backend: str | None = None,
    iterations: int | None = None,
    burn_in: int | None = None,
    initial_state: Any = None,
    return_samples: bool = False,
    summary_method: str | None = None,
    delay_dist: Any = None,
    diagnostics: bool = False,
    move_window: int | None = None,
    theta_prop_sigma: float | None = None,
    global_move_prob: float | None = None,
    max_k: int | None = None,
    u_prop_sigma: float | None = None,
) -> R:
    """Call old or extended RJMCMC APIs without changing ``methods.py``."""
    if seed is not None:
        import numpy as np

        np.random.seed(int(seed))
    signature = inspect.signature(run_fn)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs: dict[str, Any] = {}
    values = {
        "p_geom": p_geom,
        "theta_sigma": theta_sigma,
        "seed": seed,
        "backend": backend,
        "iterations": iterations,
        "burn_in": burn_in,
        "initial_state": initial_state,
        "return_samples": return_samples,
        "summary_method": summary_method,
        "delay_dist": delay_dist,
        "diagnostics": diagnostics,
        "move_window": move_window,
        "theta_prop_sigma": theta_prop_sigma,
        "global_move_prob": global_move_prob,
        "max_k": max_k,
        "u_prop_sigma": u_prop_sigma,
    }
    for name, value in values.items():
        if value is not None and (name in signature.parameters or accepts_kwargs):
            kwargs[name] = value
    return run_fn(data, **kwargs)


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> None:
    """Write bytes beside ``path`` and replace the destination atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def atomic_pickle_dump(value: Any, path: str | os.PathLike[str]) -> None:
    """Atomically pickle ``value`` to ``path``."""
    atomic_write_bytes(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path: str | os.PathLike[str]) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def atomic_json_dump(value: Any, path: str | os.PathLike[str]) -> None:
    """Atomically write JSON, allowing pathlib values in small manifests."""
    payload = json.dumps(value, indent=2, sort_keys=True, default=str).encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_text_write(text: str, path: str | os.PathLike[str]) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def run_parallel(
    tasks: Sequence[T],
    worker: Callable[[T], R],
    max_workers: int,
    description: str,
) -> list[R]:
    """Run tasks with a Windows-safe spawn pool and a progress bar.

    Serial execution remains available for debugging and smoke tests. In the
    pool case results are returned in completion order; each task writes its
    own checkpoint, so ordering is immaterial for resumability.
    """
    if not tasks:
        return []
    if max_workers <= 1:
        output: list[R] = []
        for task in tqdm(tasks, desc=description):
            output.append(worker(task))
        return output

    context = mp.get_context("spawn")
    output = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        with tqdm(total=len(futures), desc=description) as progress:
            for future in as_completed(futures):
                output.append(future.result())
                progress.update(1)
    return output


def ensure_ignored_output(path: str | os.PathLike[str], expected_name: str) -> Path:
    """Validate that a user-supplied output path is under an ignored area.

    The repository ignores directories named ``results`` and ``plots``. This
    guard prevents an accidental CLI path from putting generated artifacts in
    the source tree while still allowing nested run directories.
    """
    candidate = Path(path)
    parts = {part.lower() for part in candidate.parts}
    if expected_name.lower() not in parts:
        raise ValueError(
            f"Generated output must be under an ignored '{expected_name}' directory: {candidate}"
        )
    return candidate
