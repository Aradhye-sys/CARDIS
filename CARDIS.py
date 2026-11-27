"""CARDIS: standalone distribution of the current paper implementation.

Fast SMASH reconstruction, locally gated diffusion and adaptively reweighted
Whittaker baseline fitting. Default maximum: 60 weighted fits.

Usage
-----
    from CARDIS import CARDIS
    baseline = CARDIS(y, rscript_path="/absolute/path/to/Rscript")
    corrected = y - baseline

For a batch, CARDIS_batch(Y) expects columns: [channels, spectra].
Supply Rscript explicitly, through CARDIS_RSCRIPT, or on PATH. A nearby
Runtime/Rscript launcher is also recognised when present.

Dependencies
------------
Python 3.12+ with NumPy 2.5.1 and SciPy 1.18.0 for the recorded environment.
R with smashr 1.3.12, ashr 2.2.63 and wavethresh 4.7.3.

All CARDIS-specific source and native scripts are embedded below as readable
source strings. On import they are verified and materialised in a private
temporary directory, which is removed at interpreter exit. No separate CARDIS
source folder or installed Python CARDIS package is required. The embedded
scientific Python and R files are unchanged from the paper implementation.
Only the temporary runtime configuration is adapted to discover an R executable.
R and its installed dependencies are not embedded.

SMASH attribution: Xing, Carbonetto and Stephens (2021), JMLR 22(93),
https://jmlr.org/papers/v22/19-042.html.
Third-party notices are embedded in THIRD_PARTY_NOTICES.md below.
"""
from __future__ import annotations

import atexit as _atexit
import hashlib as _hashlib
import importlib.util as _importlib_util
import json as _json
from pathlib import Path as _Path
import shutil as _shutil
import sys as _sys
import tempfile as _tempfile

STANDALONE_PACKAGED_DATE = "2026-11-27"

# The following strings retain the original source bytes, including all
# numerical safeguards, local diffusion rules and native SMASH optimisations.
_BUNDLED_SOURCES = {
    # ---- CARDIS.py ----
    'CARDIS.py': r'''"""CARDIS: one native SMASH fit, optional local diffusion, guarded baseline.

The canonical October 7 workflow is numerically identical to the validated
round40 one-SMASH local-MAD rule. The input is a uniformly sampled real vector.
Default maximum60; explicit operating range30–90, with existing early guards.
The final baseline is always subtracted from the original measurement.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from threading import RLock
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT
RELEASE = "CARDIS_2026-10-07"
PUBLIC_METHOD = "CARDIS"
PUBLIC_RELEASE_DATE = "2026-10-07"
EXPERIMENTAL_RELEASE = RELEASE  # Compatibility metadata alias.
CONDITIONING_RULE = "shape_local_MAD_diffusion2"
DEFAULT_ITERATIONS, MIN_ITERATIONS, MAX_ITERATIONS = 60, 30, 90
_SUPPORT_PATH = ROOT / "_support.py"
_GATE_PATH = ROOT / "diffusion.py"
_PINS = {
    _SUPPORT_PATH: "c909a52642008ac5a06c46ac56bfcaf0d2e309473e3e0a8b441d22c9db26a9ed",
    _GATE_PATH: "0d5f0cfbf3d17dca1de285f79243315ef3b798df0f67e3f40f3abeabb59a1ddc",
    ROOT / "_engine/baseline_bridge.py": "5a70207878c0bad08d5c532c0f2c5ccd79bdce733a23db6c394910012539e28e",
    ROOT / "_engine/baseline_numerics.py": "a3d711534e3aea58ad1a4df600e99cdec2964d17605b8f0089d0d4b38db60234",
    ROOT / "_engine/CARDIS_paper_reference.py": "c333a9e4830e3f2817403f0fa5002fe03b84a98f02ae5f538eaf6dc5a6d9b0f1",
    ROOT / "_native/fast_smash.R": "df71cb8b69d71b2ece273254350181a2763fee1308170d6c07a85d2cbf4206a5",
    ROOT / "_native/fft_variance.R": "3da701dd0823726e71d92659052cc8f2fb0937d4be15d7ec10e1b395875b47d8",
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_MODULE_SHA = _sha(__file__)
_LOAD_LOCK = RLock()


def _load(path, role):
    path = Path(path).resolve(strict=True)
    location = hashlib.sha256(str(path).encode()).hexdigest()
    name = "_cardis_20261007_" + role + "_" + location + "_" + _sha(path)
    with _LOAD_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load CARDIS dependency: " + str(path))
            value = importlib.util.module_from_spec(spec)
            sys.modules[name] = value
            try:
                spec.loader.exec_module(value)
            except BaseException:
                sys.modules.pop(name, None)
                raise
    return sys.modules[name]


def _dependencies():
    for path, expected in _PINS.items():
        if _sha(path) != expected:
            raise RuntimeError("Frozen CARDIS dependency changed: " + str(path))
    if _sha(__file__) != _MODULE_SHA:
        raise RuntimeError("CARDIS module changed after import")
    return _load(_SUPPORT_PATH, "guarded_support"), _load(_GATE_PATH, "gate")


def _source_guards():
    _dependencies()
    sources = {str(path): value for path, value in _PINS.items()}
    sources[str(Path(__file__).resolve())] = _MODULE_SHA
    return sources


def _digest(value):
    data = np.ascontiguousarray(value, dtype="<f8")
    hashed = hashlib.sha256()
    hashed.update(str(data.shape).encode())
    hashed.update(data.tobytes())
    return hashed.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PreparedFastDiffusor:
    raw: np.ndarray
    mean: np.ndarray
    smash: np.ndarray
    gate: np.ndarray
    native_input_variance: np.ndarray
    metadata: dict
    array_sha256: dict
    metadata_sha256: str
    source_manifest_sha256: str


PreparedConditioning = PreparedFastDiffusor


def _prepared(raw, mean, smash, gate, variance, metadata, sources):
    arrays = {}
    for name, value in (("raw", raw), ("mean", mean), ("smash", smash),
                        ("gate", gate), ("native_input_variance", variance)):
        value = np.asarray(value, dtype=float).copy()
        if value.shape != raw.shape or not np.isfinite(value).all():
            raise RuntimeError("Invalid prepared " + name)
        value.setflags(write=False)
        arrays[name] = value
    if np.any(arrays["native_input_variance"] < 0) or np.any((arrays["gate"] < 0) | (arrays["gate"] > 1)):
        raise RuntimeError("Invalid native input variance or gate")
    np.testing.assert_array_equal(arrays["mean"][arrays["gate"] == 0], arrays["smash"][arrays["gate"] == 0])
    info = deepcopy(metadata)
    info.update(release=EXPERIMENTAL_RELEASE, source_method=CONDITIONING_RULE,
        gate_rule_changed_from_full_probe_diffusor=True, original_conditioning_rule_retained=False,
        nonlinear_probe_response_scale_used=False, probe_fits=0,
        native_input_variance_role="Native SMASH input-noise diagnostic only; not used by this gate",
        native_input_variance_is_conditioned_mean_uncertainty=False,
        baseline_equations_changed=False, iteration_automation_claimed=False,
        parameter_free_claimed=False, universal_peak_preservation_claimed=False,
        production_replacement=True)
    return PreparedFastDiffusor(**arrays, metadata=info,
        array_sha256={name: _digest(value) for name, value in arrays.items()},
        metadata_sha256=_json_digest(info), source_manifest_sha256=_json_digest(sources))


def _validate(prepared, raw, sources):
    if not isinstance(prepared, PreparedFastDiffusor):
        raise TypeError("prepared must come from this CARDIS release")
    if prepared.source_manifest_sha256 != _json_digest(sources):
        raise RuntimeError("Prepared conditioning source version differs")
    for name in ("raw", "mean", "smash", "gate", "native_input_variance"):
        value = getattr(prepared, name)
        if value.shape != raw.shape or not np.isfinite(value).all() or _digest(value) != prepared.array_sha256.get(name):
            raise RuntimeError("Prepared " + name + " integrity check failed")
    if _digest(raw) != prepared.array_sha256["raw"]:
        raise ValueError("Prepared conditioning belongs to different raw intensities")
    if _json_digest(prepared.metadata) != prepared.metadata_sha256:
        raise RuntimeError("Prepared metadata integrity check failed")
    if prepared.metadata.get("source_method") != CONDITIONING_RULE or prepared.metadata.get("release") != EXPERIMENTAL_RELEASE:
        raise RuntimeError("Prepared conditioning identifies a different rule")
    if np.any(prepared.native_input_variance < 0) or np.any((prepared.gate < 0) | (prepared.gate > 1)):
        raise RuntimeError("Invalid prepared native variance or gate")
    np.testing.assert_array_equal(prepared.mean[prepared.gate == 0], prepared.smash[prepared.gate == 0])


class ConditioningRuntimeError(RuntimeError):
    def __init__(self, message, diagnostics_dir):
        self.diagnostics_dir = Path(diagnostics_dir)
        super().__init__(str(message) + "\nDiagnostics retained: " + str(self.diagnostics_dir))


def prepare_conditioning(y, *, timeout_seconds=120., diagnostics_root=None,
                         keep_diagnostics=False, rscript_path=None):
    """One native SMASH fit, followed by local-MAD-triggered diffusion."""
    support, _ = _dependencies()
    raw = support._observed(y)
    return prepare_conditioning_batch(raw[:, None], timeout_seconds=timeout_seconds,
        diagnostics_root=diagnostics_root, keep_diagnostics=keep_diagnostics,
        rscript_path=rscript_path)[0]


def prepare_conditioning_batch(spectra, *, timeout_seconds=120., diagnostics_root=None,
                              keep_diagnostics=False, rscript_path=None):
    """Independent columns in one native process; shape [channels, spectra]."""
    if isinstance(timeout_seconds, (bool, np.bool_)) or not isinstance(timeout_seconds, (int, float, np.integer, np.floating)) or not math.isfinite(float(timeout_seconds)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not isinstance(keep_diagnostics, (bool, np.bool_)):
        raise TypeError("keep_diagnostics must be boolean")
    support, gates = _dependencies()
    raw = support._observed_batch(spectra)
    sources = _source_guards()
    n, m = raw.shape
    nonconstant = np.flatnonzero(np.any(raw != raw[0:1], axis=0))
    smash, variance = raw.copy(), np.zeros_like(raw)
    records, job, native_seconds = [], None, 0.
    started = time.perf_counter()
    parent = None if diagnostics_root is None else Path(diagnostics_root).expanduser().resolve()

    def new_job(prefix):
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()

    try:
        if len(nonconstant):
            batch = raw[:, nonconstant].T.copy()
            allowed = float(timeout_seconds) * len(batch)
            if not math.isfinite(allowed):
                raise ValueError("Native process timeout is not representable")
            runtime = support._runtime(rscript_path)
            job = new_job("cardis_fast_diffusor_")
            np.savetxt(job / "input.csv", batch, delimiter=",", fmt="%.17g")
            native_start = time.perf_counter()
            result = subprocess.run([str(runtime), str(RELEASE_ROOT / "_native/fast_smash.R"),
                str(job / "input.csv"), str(job)], capture_output=True, text=True,
                timeout=allowed, check=False)
            native_seconds = time.perf_counter() - native_start
            (job / "run.log").write_text(result.stdout + "\nSTDERR\n" + result.stderr)
            if result.returncode:
                raise RuntimeError("Native SMASH failed with exit code " + str(result.returncode))
            base = np.loadtxt(job / "smash_mean.csv", delimiter=",", ndmin=2)
            native_variance = np.loadtxt(job / "smash_variance.csv", delimiter=",", ndmin=2)
            if base.shape != batch.shape or native_variance.shape != batch.shape or not np.isfinite(base).all() or not np.isfinite(native_variance).all() or np.any(native_variance < 0):
                raise RuntimeError("Native SMASH returned invalid dimensions, means or input variance")
            with (job / "r_diagnostics.csv").open() as handle:
                records = list(csv.DictReader(handle))
            if len(records) != len(batch):
                raise RuntimeError("Incomplete native SMASH diagnostics")
            for position, row in enumerate(records):
                if row.get("row") != str(position + 1) or row.get("success") != "TRUE" or row.get("smashr_version") != "1.3.12" or row.get("ashr_version") != "2.2.63" or row.get("wavethresh_version") != "4.7.3":
                    raise RuntimeError("Native SMASH failed or dependency versions differ")
            smash[:, nonconstant] = base.T
            variance[:, nonconstant] = native_variance.T
        values = []
        for i in range(m):
            # One selected gate only, with no unused proxies or repeated fits.
            candidate = gates.condition(smash[:, i])
            info = deepcopy(candidate["metadata"])
            info.update(native_method="smash_DaubLeAsymm_8", native_versions={
                name: records[0][name + "_version"] for name in ("smashr", "ashr", "wavethresh")} if records else {},
                native_fits_per_input=int(i in nonconstant), native_batch_fits=len(nonconstant),
                constant_identity=bool(i not in nonconstant), native_runtime_invoked=bool(i in nonconstant),
                native_process_seconds=native_seconds, batch_columns=m, column_index=i,
                diagnostics_dir=str(job) if keep_diagnostics and job is not None else None,
                same_output_independent_of_batch_order=True, no_probe_keys_required=True,
                uniform_sampling_assumed=True, correlated_noise_guarantee=False,
                manual_smoothing_strength=False, source_sha256=sources,
                elapsed_batch_seconds=time.perf_counter() - started)
            values.append(_prepared(raw[:, i], candidate["mean"], smash[:, i], candidate["gate"],
                variance[:, i], info, sources))
        if _source_guards() != sources:
            raise RuntimeError("Frozen sources changed during conditioning")
        if job is not None and keep_diagnostics:
            np.savez_compressed(job / "prepared_outputs.npz", raw=raw, smash=smash,
                native_input_variance=variance, mean=np.column_stack([v.mean for v in values]),
                gate=np.column_stack([v.gate for v in values]))
            (job / "python_metadata.json").write_text(json.dumps([v.metadata for v in values], indent=2, allow_nan=False) + "\n")
    except Exception as error:
        if job is None:
            job = new_job("cardis_fast_diffusor_failure_")
        if isinstance(error, subprocess.TimeoutExpired):
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            (job / "run.log").write_text(stdout + "\nSTDERR\n" + stderr)
        (job / "failure.json").write_text(json.dumps(dict(error=type(error).__name__ + ": " + str(error),
            conditioning_rule=CONDITIONING_RULE, truth_used=False, diagnostics_dir=str(job)), indent=2) + "\n")
        raise ConditioningRuntimeError(error, job) from error
    if job is not None and not keep_diagnostics:
        # Delete only the unique private job from this successful call.
        shutil.rmtree(job)
    return values


def CARDIS(y, max_iter=DEFAULT_ITERATIONS, return_debug=False, *, prepared=None,
           return_history=False, timeout_seconds=120., diagnostics_root=None,
           keep_diagnostics=False, rscript_path=None):
    """Return baseline; subtract it from the unchanged raw measurement."""
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or not MIN_ITERATIONS <= max_iter <= MAX_ITERATIONS:
        raise ValueError("max_iter must be an integer 30–90; default maximum 60")
    if not isinstance(return_debug, (bool, np.bool_)) or not isinstance(return_history, (bool, np.bool_)):
        raise TypeError("return_debug and return_history must be boolean")
    if return_history and not return_debug:
        raise ValueError("return_history requires return_debug=True")
    support, _ = _dependencies()
    raw = support._observed(y)
    reused = prepared is not None
    if prepared is None:
        prepared = prepare_conditioning(raw, timeout_seconds=timeout_seconds,
            diagnostics_root=diagnostics_root, keep_diagnostics=keep_diagnostics,
            rscript_path=rscript_path)
    sources = _source_guards()
    _validate(prepared, raw, sources)
    api = support._api()
    core = ModuleType("cardis_fast_private_guarded_baseline")
    exec(api._core_code(), core.__dict__)
    solves = []

    def guarded_solve(working, weights, penalty):
        baseline, info = api.solve_weighted_whittaker(working, weights, penalty, return_diagnostics=True)
        solves.append(info)
        return baseline

    core._solve_weighted_whittaker = guarded_solve
    fit = core.CARDIS(raw, max_iter=int(max_iter), _working_override=prepared.mean,
        return_debug=True, return_history=bool(return_history))
    _validate(prepared, raw, sources)
    if _source_guards() != sources:
        raise RuntimeError("Frozen sources changed during baseline fitting")
    baseline = np.asarray(fit["baseline"], dtype=float).copy()
    if baseline.shape != raw.shape or not np.isfinite(baseline).all() or len(solves) != fit["num_solves"] or not 1 <= len(solves) <= max_iter:
        raise RuntimeError("Invalid baseline or solve diagnostics")
    np.testing.assert_array_equal(fit["y_work"], prepared.mean)
    if not return_debug:
        return baseline
    info = deepcopy(prepared.metadata)
    info["conditioning_reused"] = reused
    return dict(baseline=baseline, corrected=raw-baseline, y_raw=raw.copy(), y_work=prepared.mean.copy(),
        smash_mean=prepared.smash.copy(), diffusion_gate=prepared.gate.copy(),
        native_input_noise_variance=prepared.native_input_variance.copy(),
        input_noise_variance=prepared.native_input_variance.copy(),
        input_noise_variance_role="Native input-noise diagnostic; not conditioned-mean uncertainty",
        smoothing=info,
        baseline_debug=fit, numerical_solve_diagnostics=solves, release=EXPERIMENTAL_RELEASE,
        conditioning_rule=CONDITIONING_RULE, conditioning_statistical_rule_changed=True,
        original_conditioning_rule_retained=False, gate_rule_changed_from_full_probe_diffusor=True,
        baseline_equations_changed=False, numerical_safeguards_retained=True,
        maximum_iterations=int(max_iter), operating_iteration_range=[30, 90],
        stopping_status={"max_iter": "explicit_iteration_budget_reached",
            "converged": "local_numerical_stabilisation_criterion_met",
            "weight_collapse_guard": "data_fidelity_safety_guard_before_next_update"}[fit["stop_reason"]],
        iteration_automation_claimed=False, parameter_free_claimed=False,
        global_convergence_claimed=False, peak_safe_stopping_claimed=False,
        batch_optimisation=False, production_replacement=True, manuscript_or_figures_updated=False,
        method=PUBLIC_METHOD, public_release_date=PUBLIC_RELEASE_DATE, execution_optimised=True)


def CARDIS_batch(spectra, max_iter=DEFAULT_ITERATIONS, return_debug=False, *,
                 prepared=None, chunk_size=128, timeout_seconds=120.,
                 diagnostics_root=None, keep_diagnostics=False, rscript_path=None):
    """Independent baseline fits for columns [channels, spectra].

    Native conditioning shares process startup across each bounded chunk;
    every nonconstant spectrum receives one independent SMASH fit. This helper
    does not impose a common peak template or run the separate batch extension.
    """
    support, _ = _dependencies()
    raw = support._observed_batch(spectra)
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or not MIN_ITERATIONS <= max_iter <= MAX_ITERATIONS:
        raise ValueError("max_iter must be an integer 30–90; default maximum 60")
    if not isinstance(return_debug, (bool, np.bool_)):
        raise TypeError("return_debug must be boolean")
    if prepared is not None:
        if isinstance(prepared, (str, bytes, dict)):
            raise TypeError("prepared must contain one conditioning object per spectrum")
        prepared = list(prepared)
        if len(prepared) != raw.shape[1]:
            raise ValueError("prepared must contain one conditioning result per column")
        for i, value in enumerate(prepared):
            _validate(value, raw[:, i], _source_guards())
    fits = []
    for start in range(0, raw.shape[1], int(chunk_size)):
        stop = min(start + int(chunk_size), raw.shape[1])
        block = (prepare_conditioning_batch(raw[:, start:stop], timeout_seconds=timeout_seconds,
                    diagnostics_root=diagnostics_root, keep_diagnostics=keep_diagnostics,
                    rscript_path=rscript_path)
                 if prepared is None else prepared[start:stop])
        fits.extend(CARDIS(raw[:, start + j], max_iter=max_iter,
            return_debug=return_debug, prepared=value) for j, value in enumerate(block))
    if return_debug:
        variance = np.column_stack([fit["native_input_noise_variance"] for fit in fits])
        return dict(baseline=np.column_stack([fit["baseline"] for fit in fits]),
            corrected=np.column_stack([fit["corrected"] for fit in fits]),
            y_raw=raw.copy(), y_work=np.column_stack([fit["y_work"] for fit in fits]),
            smash_mean=np.column_stack([fit["smash_mean"] for fit in fits]),
            diffusion_gate=np.column_stack([fit["diffusion_gate"] for fit in fits]),
            native_input_noise_variance=variance, input_noise_variance=variance,
            input_noise_variance_role="Native input-noise diagnostic; not conditioned-mean uncertainty",
            smoothing=[fit["smoothing"] for fit in fits],
            fit_diagnostics=[dict(index=i, num_solves=fit["baseline_debug"]["num_solves"],
                stop_reason=fit["baseline_debug"]["stop_reason"], smoothing=fit["smoothing"])
                for i, fit in enumerate(fits)],
            maximum_iterations=int(max_iter), batch_columns=raw.shape[1],
            batch_optimisation=False, conditioning_rule=CONDITIONING_RULE,
            native_input_variance_is_conditioned_mean_uncertainty=False,
            release=RELEASE, method=PUBLIC_METHOD, public_release_date=PUBLIC_RELEASE_DATE,
            noise_shared_between_spectra=False, baseline_equations_changed=False,
            smoothing_rule_changed_from_20261004=True, execution_optimised=True,
            chunk_size=int(chunk_size))
    return np.column_stack(fits)


__all__ = ["CARDIS", "CARDIS_batch", "prepare_conditioning", "prepare_conditioning_batch",
           "PreparedFastDiffusor", "PreparedConditioning", "DEFAULT_ITERATIONS",
           "MIN_ITERATIONS", "MAX_ITERATIONS", "RELEASE"]
''',
    # ---- _support.py ----
    '_support.py': r'''"""Minimal runtime validation and guarded baseline loading for CARDIS."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from threading import RLock
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
_LOCK = RLock()


def _load(path, role):
    path = Path(path).resolve(strict=True)
    name = "_cardis_20261007_" + role + "_" + hashlib.sha256(str(path).encode()).hexdigest()
    with _LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load CARDIS dependency: " + str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        return sys.modules[name]


@lru_cache(maxsize=1)
def _api():
    bridge = _load(ROOT / "_engine/baseline_bridge.py", "bridge")
    numerics = _load(ROOT / "_engine/baseline_numerics.py", "numerics")
    return SimpleNamespace(_core_code=bridge.core_code,
        solve_weighted_whittaker=numerics.solve_weighted_whittaker)


def _observed(y):
    original = np.asarray(y)
    if original.ndim != 1 or original.size < 32:
        raise ValueError("y must be a uniformly sampled 1D spectrum of at least 32 channels")
    if np.iscomplexobj(original) or original.dtype.kind in "bUS":
        raise TypeError("y must contain finite real numeric intensities")
    try:
        value = np.asarray(original, dtype=np.float64).copy()
    except (ValueError, TypeError) as error:
        raise TypeError("y must contain finite real numeric intensities") from error
    if not np.isfinite(value).all():
        raise ValueError("y must contain finite intensities")
    return value


def _observed_batch(spectra):
    value = np.asarray(spectra)
    if value.ndim != 2 or value.shape[0] < 32 or value.shape[1] < 1:
        raise ValueError("spectra must have shape [channels, spectra], with at least 32 channels")
    return np.column_stack([_observed(value[:, i]) for i in range(value.shape[1])])


def _runtime(explicit):
    if explicit is None:
        explicit = os.environ.get("CARDIS_RSCRIPT") or os.environ.get("CARDIS_V11_RSCRIPT")
    if explicit is None:
        settings = ROOT / "runtime_config.json"
        explicit = json.loads(settings.read_text()).get("rscript_path") if settings.is_file() else None
        if explicit is not None and not Path(explicit).is_absolute():
            explicit = ROOT / explicit
    if explicit is None:
        raise FileNotFoundError("Provide rscript_path or the project R runtime configuration")
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Native R runtime unavailable: " + str(path))
    return path
''',
    # ---- diffusion.py ----
    'diffusion.py': r'''"""The selected one-SMASH local-MAD spike-triggered diffusion rule.

This is an exact consolidation of the round40 shape_local_MAD path. It contains
one gate and one optional sigma-two discrete-heat endpoint, with no comparison
proxies, perturbed SMASH fits or Gaussian smoothing of the raw measurement.
The calibrated geometric constants are retained explicitly. A narrow true band
can resemble a spike, so this rule does not guarantee universal peak retention.
"""
import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

MAD_NORMAL = 0.6744897501960817
OFFSETS = np.array([-12, -11, -10, -9, -8, -7, -6, -5, -4, -3,
                    3, 4, 5, 6, 7, 8, 9, 10, 11, 12])


def _pair(a, lag):
    padded = np.pad(a, (lag, lag), mode="reflect")
    return .5 * (padded[:-2 * lag] + padded[2 * lag:])


def _offcentre(a):
    n = a.shape[-1]
    positions = np.arange(n)[:, None] + OFFSETS[None]
    period = 2 * (n - 1)
    positions %= period
    positions = np.where(positions >= n, period - positions, positions)
    return a[positions]


def _ramp(value, lower, upper):
    return np.clip((value - lower) / (upper - lower), 0., 1.)


def _terms(m):
    q1 = m - _pair(m, 1)
    q2 = m - _pair(m, 2)
    q4 = m - _pair(m, 4)
    neighbours = _offcentre(q1)
    centre = np.median(neighbours, axis=-1)
    mad = np.median(np.abs(neighbours - centre[:, None]), axis=-1) / MAD_NORMAL
    floor = np.finfo(float).eps * max(float(np.ptp(m)), np.finfo(float).tiny)
    scale = np.maximum.reduce((mad, np.zeros_like(m), np.full_like(m, floor)))
    sharp, broad, medium = q1 - centre, q4 - 16 * centre, q2 - 4 * centre
    ratio = np.abs(sharp) / np.maximum(np.abs(broad), scale)
    medium_ratio = np.abs(sharp) / np.maximum(np.abs(medium), scale)
    significance = np.abs(sharp) / scale
    valid = np.zeros(len(m), bool)
    valid[4:-4] = True
    padded = np.pad(m, 1, mode="reflect")
    extremum = (m - padded[:-2]) * (m - padded[2:]) > 0
    same = (sharp * broad > 0) & (sharp * medium > 0) & extremum
    sensitive = _ramp(significance, 2.5, 4.) * _ramp(ratio, .22, .50) * same
    padded = np.pad(sharp, 1, mode="reflect")
    opposed = np.maximum(-np.sign(sharp) * padded[:-2], -np.sign(sharp) * padded[2:])
    opposition = opposed / np.maximum(np.abs(sharp), np.finfo(float).tiny)
    alternating = (_ramp(significance, 2.5, 4.) * _ramp(opposition, .20, .50)
                   * _ramp(medium_ratio, .40, .75) * extremum)
    return np.maximum(sensitive, alternating) * valid


def condition(smash):
    m = np.asarray(smash, dtype=float)
    if m.ndim != 1 or len(m) < 32 or not np.isfinite(m).all():
        raise ValueError("Expected a finite real SMASH vector of at least 32 channels")
    seed = np.zeros_like(m) if np.ptp(m) == 0 else _terms(m)
    if np.any(seed):
        core = maximum_filter1d(seed, size=9, mode="reflect")
        shoulder = gaussian_filter1d(core, 1.5, mode="reflect", truncate=4.0)
        gate = np.clip(np.maximum(seed, shoulder), 0., 1.)
        eigen = 1. - np.cos(np.pi * np.arange(len(m)) / len(m))
        centre = m[:1]
        transform = dct(m - centre, type=2, norm="ortho", axis=-1)
        endpoint = idct(transform * np.exp(-2. * 2. * eigen), type=2, norm="ortho", axis=-1) + centre
        mean = m + gate * (endpoint - m)
    else:
        gate, mean = np.zeros_like(m), m.copy()
    return dict(mean=mean, gate=gate, metadata=dict(method="shape_local_MAD",
        truth_used=False, native_fits_per_input=1, probe_fits=0,
        gate_scale_changed_from_full_probe_rule=True,
        shape_thresholds_and_mask_retained=True, endpoint_sigma_channels=2.,
        endpoint="DCT Neumann discrete heat", seed_centres=int(np.count_nonzero(seed)),
        blended_gate_channels=int(np.count_nonzero(gate)), gate_mean=float(np.mean(gate)),
        response_uncertainty_exact=False, SMASH_posterior_uncertainty_used=False,
        noise_proxy_assumes_independent_errors=False,
        universal_peak_preservation_claimed=False, parameter_free_claimed=False,
        exact_full_probe_bypass_certificate=not np.any(gate)))
''',
    # ---- _engine/baseline_bridge.py ----
    '_engine/baseline_bridge.py': r'''"""Reversible observed-only bridge to the unchanged CARDIS baseline equations.

The baseline reference and numerical solver are byte-identical vendor copies.
The three transformations below match the previously validated bridge exactly.
No historical Gaussian conditioning executes when the supplied working trace
is used; its legacy bandwidth remains only an inherited debug diagnostic.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / "CARDIS_paper_reference.py"
REFERENCE_SHA256 = "c333a9e4830e3f2817403f0fa5002fe03b84a98f02ae5f538eaf6dc5a6d9b0f1"
_TRANSFORMS = (
    ("    eps=1e-6,\n):\n",
     "    eps=1e-6,\n    *,\n    _working_override=None,\n"
     "    _lambda_override=None,\n):\n"),
    ("    y_work = _smooth(y_raw, sigma)\n",
     "    y_work = (_smooth(y_raw, sigma) if _working_override is None\n"
     "              else _working_override.copy())\n"),
    ("    lam, lambda_debug = _auto_lambda_from_conditioned_spectrum(y_work)\n",
     "    lam, lambda_debug = _auto_lambda_from_conditioned_spectrum(y_work)\n"
     "    if _lambda_override is not None:\n        lam = _lambda_override\n"),
)


def checked_source():
    content = REFERENCE_PATH.read_bytes()
    if hashlib.sha256(content).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("Frozen CARDIS baseline reference changed")
    source = content.decode("utf-8")
    result = source
    for original, replacement in _TRANSFORMS:
        if result.count(original) != 1:
            raise RuntimeError("Missing or ambiguous CARDIS baseline bridge anchor")
        result = result.replace(original, replacement, 1)
    restored = result
    for original, replacement in reversed(_TRANSFORMS):
        if restored.count(replacement) != 1:
            raise RuntimeError("CARDIS bridge reverse audit failed")
        restored = restored.replace(replacement, original, 1)
    if restored != source:
        raise RuntimeError("CARDIS bridge changed undeclared equations")
    return result


@lru_cache(maxsize=1)
def core_code():
    return compile(checked_source(), str(REFERENCE_PATH) + " [conditioned guarded CARDIS]", "exec")
''',
    # ---- _engine/baseline_numerics.py ----
    '_engine/baseline_numerics.py': r'''"""Validated numerical backend for the unchanged CARDIS Whittaker subproblem.

For strictly positive data weights, this solves

    (diag(w) + lam * D2.T @ D2) z = w * y,

where D2 is the unscaled second-difference operator. Ordinary CARDIS inputs use
the same assembly order and SciPy banded Cholesky solver as the frozen reference.
There is no diagonal jitter, weight renormalisation or altered penalty. A common
power-of-two scaling of both matrix and right-hand side is used only if direct
assembly would risk floating-point overflow. This leaves the mathematical
normal equations unchanged; unrepresentable cases are rejected explicitly.

The backward-error diagnostic concerns the linear solve only. It does not imply
that the fitted baseline is spectroscopically correct or that the changing-weight
outer iteration has converged.
"""

from functools import lru_cache
import math
import warnings

import numpy as np
from scipy.linalg import LinAlgError, solveh_banded


def _sample_count(n):
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("sample count must be an integer of at least 3")
    n = int(n)
    if n < 3:
        raise ValueError("sample count must be an integer of at least 3")
    return n


@lru_cache(maxsize=32)
def difference_penalty_banded(n):
    """Return a read-only lower-band representation of the exact D2.T @ D2.

    Its three rows store the main diagonal and the first and second lower
    diagonals. Padding entries are zero. The n=3 edge case has central diagonal
    4, as follows directly from D2 = [1, -2, 1]. CARDIS requires at least eight
    samples, so this edge correction cannot change a valid CARDIS fit.
    """
    n = _sample_count(n)
    ab = np.zeros((3, n), dtype=float)
    ab[0] = 6.0
    if n == 3:
        ab[0] = (1.0, 4.0, 1.0)
    else:
        ab[0, [0, 1, -2, -1]] = (1.0, 5.0, 5.0, 1.0)
    ab[1, :-1] = -4.0
    ab[1, [0, -2]] = -2.0
    ab[2, :-2] = 1.0
    # An immutable bytes owner prevents consumers from making the cache writable.
    return np.frombuffer(ab.tobytes(), dtype=float).reshape(ab.shape)


def _finite_vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _positive_scalar(value, name):
    if np.iscomplexobj(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar")
    array = np.asarray(value, dtype=float)
    if array.ndim != 0 or not np.isfinite(array) or float(array) <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(array)


def _lower_banded_matvec(ab, x):
    """Multiply the symmetric matrix represented by lower bands by x."""
    result = ab[0] * x
    for offset in (1, 2):
        band = ab[offset, :-offset]
        result[offset:] += band * x[:-offset]
        result[:-offset] += band * x[offset:]
    return result


def _diagnostic_summary(error, sample_count):
    unit_roundoff = np.finfo(float).eps / 2.0
    dimension_roundoff = sample_count * unit_roundoff
    reference_level = (dimension_roundoff / (1.0 - dimension_roundoff)
                       if dimension_roundoff < 1.0 else np.inf)
    # gamma_n is a dimension/precision-derived screening reference. It is not
    # asserted to be a rigorous bound on every banded Cholesky factorisation.
    flagged = error > reference_level
    return {
        "normal_equation_backward_error_inf": error,
        "normal_equation_roundoff_reference_inf": reference_level,
        "normal_equation_warning": bool(flagged),
        "normal_equation_warning_message": (
            "normal-equation backward error exceeds the dimension/precision screening reference"
            if flagged else None
        ),
    }


def normal_equation_diagnostics(ab, rhs, baseline):
    """Measure infinity-norm backward error against the actual solved matrix.

    The returned quantity is ||A z - rhs||_inf divided by
    ||A||_inf ||z||_inf + ||rhs||_inf. The equivalent norm-scaled calculation
    avoids overflow in this diagnostic even for very large finite inputs.
    Lower-band padding must be zero, as in SciPy's lower-band storage convention.
    """
    if np.iscomplexobj(ab):
        raise ValueError("ab must contain real values")
    ab = np.asarray(ab, dtype=float)
    baseline = _finite_vector(baseline, "baseline")
    rhs = _finite_vector(rhs, "rhs")
    if (ab.shape != (3, baseline.size) or baseline.size < 3
            or rhs.shape != baseline.shape or not np.all(np.isfinite(ab))):
        raise ValueError("ab, rhs and baseline must define a finite matching banded system")
    if ab[1, -1] != 0.0 or np.any(ab[2, -2:] != 0.0):
        raise ValueError("lower-band padding entries must be zero")

    a_max = float(np.max(np.abs(ab)))
    z_norm = float(np.max(np.abs(baseline)))
    rhs_norm = float(np.max(np.abs(rhs)))
    if a_max == 0.0:
        return _diagnostic_summary(0.0 if rhs_norm == 0.0 else 1.0, baseline.size)
    a_scaled = ab / a_max
    row_sums = _lower_banded_matvec(np.abs(a_scaled), np.ones(baseline.size))
    row_factor = float(np.max(row_sums))
    a_unit = a_scaled / row_factor
    if z_norm == 0.0:
        error = 0.0 if rhs_norm == 0.0 else 1.0
    elif rhs_norm == 0.0:
        error = float(np.max(np.abs(_lower_banded_matvec(a_unit, baseline / z_norm))))
    else:
        # Let alpha = ||rhs|| / (||A|| ||z||). Logarithms avoid forming the
        # potentially overflowing product in alpha's denominator.
        log_alpha = (math.log2(rhs_norm) - math.log2(a_max)
                     - math.log2(row_factor) - math.log2(z_norm))
        if log_alpha >= 0.0:
            inverse_alpha = math.exp2(-log_alpha)
            rhs_fraction = 1.0 / (1.0 + inverse_alpha)
            az_fraction = inverse_alpha * rhs_fraction
        else:
            alpha = math.exp2(log_alpha)
            az_fraction = 1.0 / (1.0 + alpha)
            rhs_fraction = alpha * az_fraction
        residual = (az_fraction * _lower_banded_matvec(a_unit, baseline / z_norm)
                    - rhs_fraction * (rhs / rhs_norm))
        error = float(np.max(np.abs(residual)))
    return _diagnostic_summary(error, baseline.size)


def _common_scaling_exponent(y, w, lam):
    """Choose zero unless the direct assembly has an overflow-risk exponent."""
    w_exponent = np.frexp(w)[1]
    y_exponent = np.frexp(np.abs(y))[1]
    # The second-difference penalty has largest coefficient six. One extra
    # exponent allows for adding a positive data weight to the main diagonal.
    matrix_bound = max(math.frexp(lam)[1] + 3, int(np.max(w_exponent))) + 1
    rhs_bound = int(np.max(w_exponent + y_exponent))
    maximum_exponent = max(matrix_bound, rhs_bound)
    if maximum_exponent <= 1023:
        return 0
    # Power-of-two scaling itself is exact when representable; 900 leaves ample
    # exponent headroom for the short banded factorisation and diagnostics.
    return maximum_exponent - 900


def solve_weighted_whittaker(y, w, lam, *, return_diagnostics=False):
    """Solve the validated weighted Whittaker system without changing its model.

    Parameters are finite matching one-dimensional arrays y and w with at least
    three elements, strictly positive w, and a finite strictly positive scalar
    lam. The optional diagnostics are returned as ``(baseline, diagnostics)``.
    With the default ``return_diagnostics=False``, only the baseline is returned.
    Inputs and the cached penalty are never modified.
    """
    y = _finite_vector(y, "y")
    w = _finite_vector(w, "w")
    _sample_count(y.size)
    if w.shape != y.shape or np.any(w <= 0.0):
        raise ValueError("w must match y and contain strictly positive weights")
    lam = _positive_scalar(lam, "lam")
    scale_exponent = _common_scaling_exponent(y, w, lam)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        scaled_w = np.ldexp(w, -scale_exponent) if scale_exponent else w
        scaled_lam = math.ldexp(lam, -scale_exponent) if scale_exponent else lam
        if np.any(scaled_w <= 0.0) or scaled_lam <= 0.0:
            raise FloatingPointError("common scaling cannot preserve positive coefficients")
        ab = scaled_lam * difference_penalty_banded(y.size).copy()
        ab[0] += scaled_w
        rhs = scaled_w * y
    if not np.all(np.isfinite(ab)) or not np.all(np.isfinite(rhs)):
        raise FloatingPointError("the weighted normal equations are not representable")
    # Keep the assembled system intact when its actual residual is requested.
    solver_ab = ab.copy() if return_diagnostics else ab
    try:
        baseline = solveh_banded(
            solver_ab, rhs, lower=True, overwrite_ab=True,
            overwrite_b=False, check_finite=False,
        )
    except LinAlgError as error:
        raise LinAlgError(
            "weighted Whittaker factorisation failed; no jitter or model change was applied"
        ) from error
    if not np.all(np.isfinite(baseline)):
        raise FloatingPointError("the weighted solve returned a non-finite baseline")
    if not return_diagnostics:
        return baseline
    diagnostics = normal_equation_diagnostics(ab, rhs, baseline)
    if diagnostics["normal_equation_warning"]:
        warnings.warn(diagnostics["normal_equation_warning_message"], RuntimeWarning, stacklevel=2)
    diagnostics.update({
        "coefficient_scaling_power_of_two": scale_exponent,
        "solver": "scipy.linalg.solveh_banded",
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "lambda": lam,
    })
    return baseline, diagnostics
''',
    # ---- _engine/CARDIS_paper_reference.py ----
    '_engine/CARDIS_paper_reference.py': r'''import numpy as np
from scipy.linalg import solveh_banded
from scipy.ndimage import gaussian_filter1d

EPS = 1e-12
SOLVER_EPS = 1e-6
D_CACHE = {}


def _mad(x):
    x = np.asarray(x, float).ravel()
    return 1.4826 * (np.median(np.abs(x - np.median(x))) + EPS)


def _smooth(x, sigma):
    return gaussian_filter1d(
        np.asarray(x, float).ravel(),
        sigma=max(0.5, float(sigma)),
        mode="reflect",
    )


def _dtd_banded(n):
    """Lower-banded matrix for D2.T @ D2."""
    if n not in D_CACHE:
        ab = np.zeros((3, n), float)
        ab[0] = 6.0
        ab[0, [0, 1, -2, -1]] = (1.0, 5.0, 5.0, 1.0)
        ab[1, :-1] = -4.0
        ab[1, [0, -2]] = -2.0
        ab[2, :-2] = 1.0
        D_CACHE[n] = ab
    return D_CACHE[n]


def _solve_weighted_whittaker(y, w, lam):
    y = np.asarray(y, float).ravel()
    w = np.asarray(w, float).ravel()
    ab = float(lam) * _dtd_banded(y.size).copy()
    ab[0] += w
    return solveh_banded(
        ab,
        w * y,
        lower=True,
        overwrite_ab=True,
        overwrite_b=False,
        check_finite=False,
    )


def _smoothness_penalty(z):
    d2 = np.diff(np.asarray(z, float), n=2)
    return float(np.dot(d2, d2))


def _fft_band_fraction(y, frac, low=True):
    y = np.asarray(y, float).ravel()

    if y.size < 16:
        return 0.0

    spectrum = np.fft.rfft(y - np.mean(y))
    power = spectrum.real**2 + spectrum.imag**2
    total = np.sum(power[1:]) + SOLVER_EPS

    if low:
        n_bins = max(1, int(np.ceil(frac * power.size)))
        band_power = np.sum(power[1:1 + n_bins])
    else:
        start = max(
            1,
            min(int(np.floor((1.0 - frac) * power.size)), power.size - 1),
        )
        band_power = np.sum(power[start:])

    return float(band_power / total)


def _fast_structure_ratio(y):
    return _fft_band_fraction(y, frac=0.25, low=False)


def _conditioning_sigma(y):
    hf = _fast_structure_ratio(y)

    hf_low = 0.0038
    hf_high = 0.0920
    sigma_low = 5.0
    sigma_high = 10.0
    scale_power = 3.0

    t = (hf - hf_low) / (hf_high - hf_low + EPS)
    t = np.clip(t, 0.0, 1.0)
    t = t**scale_power

    return float(sigma_low + t * (sigma_high - sigma_low))


def _baseline_simplicity_score(y_work):
    bottom_0p5 = _fft_band_fraction(y_work, frac=0.005, low=True)
    bottom_5 = _fft_band_fraction(y_work, frac=0.05, low=True)

    return float(bottom_0p5 / (bottom_5 + EPS)), bottom_0p5, bottom_5


def _auto_lambda_from_conditioned_spectrum(y_work):
    base_lambda = 1.0e4
    max_lambda = 7.0e4

    score_low = 0.08
    score_high = 0.70
    boost_power = 10.0

    score, bottom_0p5, bottom_5 = _baseline_simplicity_score(y_work)

    t = (score - score_low) / (score_high - score_low + EPS)
    t = np.clip(t, 0.0, 1.0)

    simplicity = (1.0 - t) ** boost_power
    lam = base_lambda + simplicity * (max_lambda - base_lambda)

    return float(lam), {
        "baseline_simplicity_score": score,
        "conditioned_bottom_0p5_fraction": bottom_0p5,
        "conditioned_bottom_5_fraction": bottom_5,
        "lambda_simplicity": simplicity,
    }


def _curvature_kernel(x, sigma_struct=6.0, alpha=0.70, gamma=2.0):
    x = np.asarray(x, float).ravel()

    if x.size < 8:
        return np.ones_like(x, float)

    d2 = np.diff(x, n=2)
    d2 = np.pad(d2, (1, 1), mode="edge")
    peakness = _smooth(d2 * d2, sigma_struct)

    pmin = float(np.min(peakness))
    pmax = float(np.max(peakness))

    if (pmax - pmin) < 1e-15:
        peak01 = np.zeros_like(peakness)
    else:
        peak01 = (peakness - pmin) / (pmax - pmin + EPS)

    kernel = 1.0 - peak01**gamma
    kernel = alpha * kernel + (1.0 - alpha)
    return np.clip(kernel, 0.0, 1.0)


def _update_weights(
    residual,
    kernel,
    omega,
    zeta,
    mu_neg,
    sig_neg,
    w_floor,
    w_ceiling,
    eps,
):
    residual = np.asarray(residual, float).ravel()
    kernel = np.asarray(kernel, float).ravel()
    w_new = np.empty_like(residual, dtype=float)

    neg = residual < 0
    pos = ~neg

    if np.any(pos):
        d = residual[pos]
        t = (d - (mu_neg + 2.0 * sig_neg)) / (sig_neg + eps)
        t = np.clip(t, -60, 60)
        logistic = 1.0 / (1.0 + np.exp(t))
        w_new[pos] = logistic * (1.0 - omega * (1.0 - kernel[pos]))

    if np.any(neg):
        d = residual[neg]
        t = (-d) / (sig_neg + eps)
        t = np.clip(t, -60, 60)
        logistic = 1.0 / (1.0 + np.exp(-t))
        w_new[neg] = zeta * kernel[neg] * logistic

    return np.clip(w_new, w_floor, w_ceiling)


def CARDIS(
    y,
    return_debug=False,
    return_history=False,
    max_iter=60,
    sigma_struct=6.0,
    kernel_alpha=0.70,
    kernel_gamma=2.0,
    kernel_ema=0.10,
    kernel_update_every=2,
    omega=0.08,
    zeta=2.0,
    beta=0.25,
    w_floor=1e-6,
    w_ceiling=1.0,
    max_floor_fraction=0.95,
    stop_baseline=1e-4,
    stop_weight=1e-3,
    patience=3,
    eps=1e-6,
):
    """CARDIS V2 with consistent iterative state.

    Conditioning, automatic lambda selection, curvature construction and the
    residual-weight formula are unchanged from V1.  Each returned baseline is
    paired with the exact weights used to solve it.  The changing-weight
    surrogate is reported diagnostically but is never used to select an
    iteration.  Reaching ``max_iter`` is reported honestly rather than being
    presented as convergence.  A conservative near-total-at-floor guard prevents
    an accepted next state from discarding the data-fidelity term numerically.
    """
    y_raw = np.asarray(y, float).ravel()
    n = y_raw.size

    if n < 8:
        raise ValueError("CARDIS requires at least 8 finite samples.")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("CARDIS input contains NaN or infinity.")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be at least 1.")
    if int(kernel_update_every) < 1:
        raise ValueError("kernel_update_every must be at least 1.")
    if int(patience) < 1:
        raise ValueError("patience must be at least 1.")
    if not (0.0 < float(beta) <= 1.0):
        raise ValueError("beta must lie in (0, 1].")
    if not (0.0 <= float(kernel_ema) <= 1.0):
        raise ValueError("kernel_ema must lie in [0, 1].")
    if not (0.0 < float(w_floor) <= float(w_ceiling)):
        raise ValueError("Require 0 < w_floor <= w_ceiling.")
    if not (0.0 < float(max_floor_fraction) <= 1.0):
        raise ValueError("max_floor_fraction must lie in (0, 1].")
    sigma = _conditioning_sigma(y_raw)
    y_work = _smooth(y_raw, sigma)
    lam, lambda_debug = _auto_lambda_from_conditioned_spectrum(y_work)

    kernel = _curvature_kernel(
        y_work,
        sigma_struct=sigma_struct,
        alpha=kernel_alpha,
        gamma=kernel_gamma,
    )
    weights = np.clip(kernel, w_floor, w_ceiling)

    previous_baseline = None
    previous_weights = None
    required_patience = max(
        int(patience),
        int(kernel_update_every) + 1,
    )
    stable = 0
    stop_reason = "max_iter"
    history = [] if return_history else None

    baseline = np.zeros(n, float)
    residual = y_work.copy()
    baseline_change = np.inf
    weight_change = np.inf
    final_surrogate = np.nan
    negative_residual_scale = np.nan

    for it in range(int(max_iter)):
        n_iter = it + 1

        # This is the accepted solved state.  These exact weights are returned
        # if the algorithm stops at this iteration.
        baseline = _solve_weighted_whittaker(y_work, weights, lam)
        residual = y_work - baseline
        final_surrogate = float(
            np.sum(weights * residual**2)
            + lam * _smoothness_penalty(baseline)
        )

        if previous_baseline is not None:
            baseline_change = float(
                np.linalg.norm(baseline - previous_baseline)
                / max(
                    np.linalg.norm(baseline),
                    np.linalg.norm(previous_baseline),
                    eps,
                )
            )
            weight_change = float(
                np.linalg.norm(weights - previous_weights)
                / max(
                    np.linalg.norm(weights),
                    np.linalg.norm(previous_weights),
                    eps,
                )
            )

            if (
                baseline_change < float(stop_baseline)
                and weight_change < float(stop_weight)
            ):
                stable += 1
            else:
                stable = 0

        if return_history:
            history.append(
                {
                    "iteration": n_iter,
                    "baseline_change": baseline_change,
                    "weight_change": weight_change,
                    "surrogate_objective": final_surrogate,
                    "weight_mean": float(np.mean(weights)),
                    "weight_median": float(np.median(weights)),
                    "weight_floor_fraction": float(
                        np.mean(weights <= float(w_floor) * 1.01)
                    ),
                    "weight_effective_sample_size": float(
                        np.sum(weights) ** 2
                        / (np.sum(weights**2) + eps)
                    ),
                    "negative_residual_fraction": float(np.mean(residual < 0)),
                }
            )

        if stable >= required_patience:
            stop_reason = "converged"
            break
        if n_iter >= int(max_iter):
            stop_reason = "max_iter"
            break

        candidate_kernel = kernel
        if (it % int(kernel_update_every)) == 0:
            new_kernel = _curvature_kernel(
                residual,
                sigma_struct=sigma_struct,
                alpha=kernel_alpha,
                gamma=kernel_gamma,
            )
            candidate_kernel = np.clip(
                (1.0 - kernel_ema) * kernel + kernel_ema * new_kernel,
                0.0,
                1.0,
            )

        negative = residual < 0
        if np.any(negative):
            mu_neg = float(np.mean(residual[negative]))
            negative_residual_scale = float(
                np.std(residual[negative]) + eps
            )
        else:
            mu_neg = 0.0
            negative_residual_scale = float(_mad(residual) + eps)

        proposed_weights = _update_weights(
            residual=residual,
            kernel=candidate_kernel,
            omega=omega,
            zeta=zeta,
            mu_neg=mu_neg,
            sig_neg=negative_residual_scale,
            w_floor=w_floor,
            w_ceiling=w_ceiling,
            eps=eps,
        )
        candidate_weights = np.clip(
            (1.0 - beta) * weights + beta * proposed_weights,
            w_floor,
            w_ceiling,
        )

        candidate_floor_fraction = float(
            np.mean(
                candidate_weights
                <= float(w_floor) * 1.01
            )
        )
        if candidate_floor_fraction >= float(max_floor_fraction):
            stop_reason = "weight_collapse_guard"
            break

        previous_baseline = baseline.copy()
        previous_weights = weights.copy()
        kernel = candidate_kernel
        weights = candidate_weights

    if return_debug:
        debug = {
            "baseline": baseline,
            "raw_baseline": baseline,
            "weights": weights,
            "weights_used": weights,
            "kernel": kernel,
            "y_work": y_work,
            "residual": residual,
            "lam_input": None,
            "lam_eff": lam,
            "fast_structure_ratio": _fast_structure_ratio(y_raw),
            "conditioning_sigma": sigma,
            "negative_residual_scale": negative_residual_scale,
            "iterations": n_iter,
            "num_solves": n_iter,
            "selected_iteration": n_iter,
            "required_patience": required_patience,
            "converged": stop_reason == "converged",
            "stop_reason": stop_reason,
            "final_baseline_change": baseline_change,
            "final_weight_change": weight_change,
            "final_weight_mean": float(np.mean(weights)),
            "final_weight_median": float(np.median(weights)),
            "final_weight_near_floor_fraction": float(
                np.mean(weights <= float(w_floor) * 1.01)
            ),
            "final_weight_effective_sample_size": float(
                np.sum(weights) ** 2
                / (np.sum(weights**2) + eps)
            ),
            "final_state_objective": final_surrogate,
            "final_surrogate_obj": final_surrogate,
            # Compatibility alias only.  No cross-iteration optimum is claimed.
            "best_obj": final_surrogate,
            # Zero by construction; external audit code may re-solve to verify.
            "state_consistency_error": 0.0,
            "method": "CARDIS_V2_consistent",
            **lambda_debug,
        }
        if return_history:
            debug["history"] = history
        return debug

    return baseline
''',
    # ---- _native/fast_smash.R ----
    '_native/fast_smash.R': r'''#!/usr/bin/env Rscript
# Fixed SMASH model, private FFT arithmetic and response-independent kernels.
# This script deliberately does not modify an installed package namespace.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("INPUT_CSV NEW_JOB_DIRECTORY")
flag <- grep("^--file=", commandArgs(), value=TRUE)[1L]
here <- dirname(normalizePath(sub("^--file=", "", flag)))
for (package in c("smashr", "ashr", "wavethresh"))
  if (!requireNamespace(package, quietly=TRUE)) stop(paste(package,"unavailable"))
expected <- c(smashr="1.3.12", ashr="2.2.63", wavethresh="4.7.3")
versions <- vapply(names(expected),function(p)as.character(packageVersion(p)),character(1))
if (!identical(unname(versions),unname(expected)))
  stop(paste("Validated R dependency versions required:",paste(names(versions),versions,collapse=", ")))
source(file.path(here,"fft_variance.R"))
Y <- as.matrix(read.csv(args[[1]],header=FALSE,check.names=FALSE))
storage.mode(Y) <- "double"
if (nrow(Y)<1L || ncol(Y)<32L || any(!is.finite(Y))) stop("Invalid observed batch")
output <- normalizePath(args[[2]],mustWork=TRUE)
paths <- file.path(output,c("smash_mean.csv","smash_variance.csv","r_diagnostics.csv"))
if (any(file.exists(paths))) stop("Refusing to overwrite previous outputs")
options(digits=17)
f <- make_fft_smash()
for(i in seq_len(nrow(Y))) {
  warnings <- character()
  started <- proc.time()[["elapsed"]]
  fit <- tryCatch(withCallingHandlers(
    f(Y[i,],sigma=NULL,joint=TRUE,post.var=FALSE,return.loglr=FALSE,
      filter.number=8L,family="DaubLeAsymm",
      ashparam=list(outputlevel="PosteriorMean")),
    warning=function(w){warnings <<- c(warnings,conditionMessage(w));invokeRestart("muffleWarning")}),
    error=function(e)e)
  seconds <- proc.time()[["elapsed"]]-started
  error <- ""
  mean <- variance <- rep(NaN,ncol(Y))
  if (inherits(fit,"error")) error <- conditionMessage(fit) else {
    mean <- as.numeric(fit$mu.res)
    variance <- as.numeric(fit$var.res)
    if(length(mean)!=ncol(Y) || any(!is.finite(mean))) {
      mean <- rep(NaN,ncol(Y));error <- "Invalid reconstructed mean"
    }
    if(length(variance)!=ncol(Y) || any(!is.finite(variance)) || any(variance<0)) {
      variance <- rep(NaN,ncol(Y));error <- paste(error,"Invalid residual variance")
    }
  }
  for(j in 1:2) write.table(matrix(if(j==1)mean else variance,nrow=1),paths[[j]],
    sep=",",row.names=FALSE,col.names=FALSE,quote=FALSE,append=i>1L,na="NaN")
  record <- data.frame(row=i,method="smash_DaubLeAsymm_8",filter=8L,
    family="DaubLeAsymm",seconds=seconds,success=error=="",error=error,
    warnings=paste(unique(warnings),collapse=" | "),
    smashr_version=versions[["smashr"]],ashr_version=versions[["ashr"]],
    wavethresh_version=versions[["wavethresh"]],R_version=R.version.string,
    arithmetic="private FFT squared-wavelet variance propagation",
    cache="response-independent kernels only",joint=TRUE,post_var=FALSE,
    return_loglr=FALSE,posterior_output="PosteriorMean")
  write.table(record,paths[[3]],sep=",",row.names=FALSE,col.names=i==1L,
    quote=TRUE,append=i>1L)
  cat(sprintf("SMASH8 %d/%d %.6fs success=%s\n",i,nrow(Y),seconds,error==""))
  flush.console()
  if(nzchar(error)) stop(paste("Spectrum",i,"failed:",error))
}
''',
    # ---- _native/fft_variance.R ----
    '_native/fft_variance.R': r'''# Isolated execution prototype: circular squared-wavelet variance propagation.
# No installed package namespace, release source or statistical default changes.
# For the periodic stationary NDWT, each scale is a circulant linear map.
# Therefore its squared weights applied to independent input variances form
# a circular convolution. The kernel comes from the author's own wd.D impulse.

make_fft_variance_operator <- function(n, filter.number=8L, family="DaubLeAsymm") {
  if (n < 2L || n != 2^round(log2(n))) stop("Dyadic n >= 2 required")
  ns <- asNamespace("smashr")
  impulse <- c(1,rep(0,n-1L))
  J <- as.integer(log2(n))
  h <- get("wd.D",ns)(impulse,filter.number=filter.number,
                           family=family,type="station",bc="periodic")
  stopifnot(length(h)==n*J,all(is.finite(h)))
  kernels <- matrix(h^2,nrow=n,ncol=J)
  spectra <- apply(kernels,2L,fft)
  support <- apply(kernels!=0,2L,fft)
  propagate <- function(v, check=TRUE) {
    if (length(v)!=n || any(!is.finite(v)) || any(v<0))
      stop("Finite nonnegative independent-channel variances required")
    if (!any(v>0)) return(rep(0,n*J))
    fv <- fft(v)
    answer <- Re(apply(spectra*fv,2L,function(z) fft(z,inverse=TRUE)))/n
    # Exact zero coefficients must stay zero. Round-off cancellation from an
    # FFT can otherwise turn an exact absence of coefficient variance into a
    # tiny signed value. Support counts are positive integers before FFT.
    counts <- Re(apply(support*fft(as.numeric(v>0)),2L,
                       function(z) fft(z,inverse=TRUE)))/n
    answer[counts < 0.5] <- 0
    # FFT absolute accuracy can be insufficient for extreme variance dynamic
    # range. Re-evaluate uncertainty-band entries by the actual impulse kernel
    # and circular indices. This is an arithmetic fallback, not a noise floor.
    if (check) {
      band <- 64*.Machine$double.eps*max(v)*colSums(kernels)
      for (j in seq_len(J)) {
        uncertain <- which(counts[,j]>=0.5 & answer[,j] <= band[j])
        if (length(uncertain)) for (t in uncertain) {
          indices <- ((t-seq_len(n)) %% n)+1L
          answer[t,j] <- sum(kernels[indices,j]*v)
        }
      }
    }
    if (any(!is.finite(answer)) || any(answer<0))
      stop("FFT variance propagation produced invalid variance")
    as.vector(answer)
  }
  list(propagate=propagate,kernels=kernels,spectra=spectra,support=support,
       n=n,J=J,filter.number=filter.number,family=family,
       boundary="periodic stationary transform from author wd.D impulse",
       independent_variances=TRUE)
}

make_fft_smash <- function() {
  ns <- asNamespace("smashr")
  original_mu <- get("mu.smooth",ns)
  original_var <- get("var.smooth",ns)
  original_public <- smashr::smash.gaus
  target <- quote(apply((rep(1,n*J) %o% data.var)*Wl$W2,1,sum))
  replacement <- quote(Wl$propagate(data.var))
  rewrite <- function(body,from,to) {
    count <- 0L
    walk <- function(x) {
      if (identical(x,from)) {count <<- count+1L;return(to)}
      if (is.call(x)||is.expression(x))
        for (i in seq_along(x)) x[i] <- list(walk(x[[i]]))
      x
    }
    list(body=walk(body),count=count)
  }
  cache <- new.env(parent=emptyenv())
  env <- new.env(parent=ns)
  env$ndwt.mat <- function(n,filter.number,family) {
    key <- paste(n,filter.number,family,sep="/")
    if (!exists(key,cache,inherits=FALSE))
      assign(key,make_fft_variance_operator(n,filter.number,family),cache)
    get(key,cache,inherits=FALSE)
  }
  for (name in c("mu.smooth","var.smooth")) {
    original <- get(name,ns)
    changed <- rewrite(body(original),target,replacement)
    stopifnot(changed$count==1L)
    reversed <- rewrite(changed$body,replacement,target)
    stopifnot(reversed$count==1L,identical(reversed$body,body(original)))
    clone <- original
    body(clone) <- changed$body
    environment(clone) <- env
    stopifnot(identical(formals(clone),formals(original)))
    assign(name,clone,env)
  }
  public <- original_public
  environment(public) <- env
  stopifnot(identical(body(public),body(original_public)),
            identical(formals(public),formals(original_public)),
            identical(body(get("mu.smooth",ns)),body(original_mu)),
            identical(body(get("var.smooth",ns)),body(original_var)))
  public
}
''',
    # ---- __init__.py ----
    '__init__.py': r'''"""Canonical CARDIS release exports."""
from .CARDIS import CARDIS, CARDIS_batch, prepare_conditioning, prepare_conditioning_batch
from .CARDIS import PreparedConditioning, PreparedFastDiffusor
from .CARDIS import DEFAULT_ITERATIONS, MIN_ITERATIONS, MAX_ITERATIONS, RELEASE

__all__ = ["CARDIS", "CARDIS_batch", "prepare_conditioning", "prepare_conditioning_batch",
           "PreparedConditioning", "PreparedFastDiffusor", "DEFAULT_ITERATIONS",
           "MIN_ITERATIONS", "MAX_ITERATIONS", "RELEASE"]
''',
    # ---- runtime_config.json ----
    'runtime_config.json': r'''{
  "rscript_path": "../../Runtime/Rscript",
  "runtime_bundled": false
}
''',
    # ---- THIRD_PARTY_NOTICES.md ----
    'THIRD_PARTY_NOTICES.md': r'''# Native conditioning dependencies

CARDIS uses the separately installed SMASH implementation from smashr 1.3.12,
with ashr 2.2.63 and wavethresh 4.7.3. The SMASH method is described by Xing,
Carbonetto and Stephens, Journal of Machine Learning Research 22(93), 2021,
https://jmlr.org/papers/v22/19-042.html.
The native scripts retain the package model, parameter defaults, wavelet family
and filter, while using private function clones for FFT variance arithmetic.
Installed package namespaces are not modified. Dependency versions are checked
before execution.

The inspected smashr DESCRIPTION declares GPL(>=3); its inst/COPYRIGHTS notes
separate GPLv2 terms for src/functions.c. The R dependencies are not bundled by
this release. Their authorship and licence terms remain applicable independently
of the CARDIS project. No claim of broader relicensing is made. The native
scripts invoke and privately clone package functions at runtime, with reversible
source-expression audits of the arithmetic substitution.

NumPy and SciPy are separately installed Python dependencies. The baseline
reference and guarded numerical backend are copied unchanged from the retained
CARDIS release; their byte hashes are part of the source-integrity manifest.
''',
}

_BUNDLED_SHA256 = {'CARDIS.py': '74f92e0ff9ce5f29e87cf1c7d9c685691c6d98ad5df3058b4c9d398e7c3e6ae6', '_support.py': 'c909a52642008ac5a06c46ac56bfcaf0d2e309473e3e0a8b441d22c9db26a9ed', 'diffusion.py': '0d5f0cfbf3d17dca1de285f79243315ef3b798df0f67e3f40f3abeabb59a1ddc', '_engine/baseline_bridge.py': '5a70207878c0bad08d5c532c0f2c5ccd79bdce733a23db6c394910012539e28e', '_engine/baseline_numerics.py': 'a3d711534e3aea58ad1a4df600e99cdec2964d17605b8f0089d0d4b38db60234', '_engine/CARDIS_paper_reference.py': 'c333a9e4830e3f2817403f0fa5002fe03b84a98f02ae5f538eaf6dc5a6d9b0f1', '_native/fast_smash.R': 'df71cb8b69d71b2ece273254350181a2763fee1308170d6c07a85d2cbf4206a5', '_native/fft_variance.R': '3da701dd0823726e71d92659052cc8f2fb0937d4be15d7ec10e1b395875b47d8', '__init__.py': '1cce7ffb6f568132290875c8592b7e447edf14b36cf906c6a5cd4f21ee16d692', 'runtime_config.json': '468212fbb7888af30618f3a088aee6db4991c1187d9cef722a05a41682ffb755', 'THIRD_PARTY_NOTICES.md': '65bb278d2bdc3933dcd5c2aa4da966734400c2061c50cde6cc95655b76dc8d4d'}


def _load_embedded_engine():
    workspace = _tempfile.TemporaryDirectory(prefix="cardis_standalone_")
    root = _Path(workspace.name)
    try:
        for name, text in _BUNDLED_SOURCES.items():
            content = text.encode("utf-8")
            if _hashlib.sha256(content).hexdigest() != _BUNDLED_SHA256[name]:
                raise RuntimeError("Embedded CARDIS source changed: " + name)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        # Explicit arguments and CARDIS_RSCRIPT still take priority in the
        # unchanged engine. Discover an existing nearby R launcher or an Rscript
        # executable on PATH; neither is required to contain CARDIS source.
        candidates = [parent / "Runtime/Rscript" for parent in _Path(__file__).resolve().parents]
        installed = _shutil.which("Rscript")
        if installed:
            candidates.append(_Path(installed))
        runtime = next((str(p.resolve()) for p in candidates if p.is_file()), None)
        (root / "runtime_config.json").write_text(_json.dumps({
            "rscript_path": runtime, "runtime_bundled": False,
        }) + "\n", encoding="utf-8")
        path = root / "CARDIS.py"
        name = "_cardis_standalone_engine_" + _hashlib.sha256(str(root).encode()).hexdigest()
        spec = _importlib_util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the embedded CARDIS implementation")
        module = _importlib_util.module_from_spec(spec)
        _sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            _sys.modules.pop(name, None)
            raise
    except BaseException:
        workspace.cleanup()
        raise
    _atexit.register(workspace.cleanup)
    return module, workspace


_ENGINE, _EMBEDDED_WORKSPACE = _load_embedded_engine()

PreparedConditioning = _ENGINE.PreparedConditioning
PreparedFastDiffusor = _ENGINE.PreparedFastDiffusor
ConditioningRuntimeError = _ENGINE.ConditioningRuntimeError
DEFAULT_ITERATIONS = _ENGINE.DEFAULT_ITERATIONS
MIN_ITERATIONS = _ENGINE.MIN_ITERATIONS
MAX_ITERATIONS = _ENGINE.MAX_ITERATIONS
RELEASE = _ENGINE.RELEASE


def CARDIS(y, max_iter=DEFAULT_ITERATIONS, return_debug=False, *, prepared=None,
           return_history=False, timeout_seconds=120., diagnostics_root=None,
           keep_diagnostics=False, rscript_path=None):
    """Return the baseline; subtract it from the original measured spectrum."""
    return _ENGINE.CARDIS(y, max_iter=max_iter, return_debug=return_debug,
        prepared=prepared, return_history=return_history,
        timeout_seconds=timeout_seconds, diagnostics_root=diagnostics_root,
        keep_diagnostics=keep_diagnostics, rscript_path=rscript_path)


def CARDIS_batch(spectra, max_iter=DEFAULT_ITERATIONS, return_debug=False, *,
                 prepared=None, chunk_size=128, timeout_seconds=120.,
                 diagnostics_root=None, keep_diagnostics=False, rscript_path=None):
    """Independent baselines for a [channels, spectra] array."""
    return _ENGINE.CARDIS_batch(spectra, max_iter=max_iter,
        return_debug=return_debug, prepared=prepared, chunk_size=chunk_size,
        timeout_seconds=timeout_seconds, diagnostics_root=diagnostics_root,
        keep_diagnostics=keep_diagnostics, rscript_path=rscript_path)


def prepare_conditioning(y, *, timeout_seconds=120., diagnostics_root=None,
                         keep_diagnostics=False, rscript_path=None):
    """Prepare one SMASH diffusor reconstruction for reuse within this process."""
    return _ENGINE.prepare_conditioning(y, timeout_seconds=timeout_seconds,
        diagnostics_root=diagnostics_root, keep_diagnostics=keep_diagnostics,
        rscript_path=rscript_path)


def prepare_conditioning_batch(spectra, *, timeout_seconds=120.,
                              diagnostics_root=None, keep_diagnostics=False,
                              rscript_path=None):
    """Prepare independent reconstructions for columns of a spectral matrix."""
    return _ENGINE.prepare_conditioning_batch(spectra,
        timeout_seconds=timeout_seconds, diagnostics_root=diagnostics_root,
        keep_diagnostics=keep_diagnostics, rscript_path=rscript_path)

__all__ = ["CARDIS", "CARDIS_batch", "prepare_conditioning",
           "prepare_conditioning_batch", "PreparedConditioning",
           "PreparedFastDiffusor", "ConditioningRuntimeError",
           "DEFAULT_ITERATIONS", "MIN_ITERATIONS", "MAX_ITERATIONS", "RELEASE"]


if __name__ == "__main__":
    print(__doc__)
