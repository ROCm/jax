# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""pytest configuration"""

import json
import os
import socket
import time
import pytest


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
  import jax
  import numpy

  doctest_namespace["jax"] = jax
  doctest_namespace["lax"] = jax.lax
  doctest_namespace["jnp"] = jax.numpy
  doctest_namespace["np"] = numpy


# A pytest hook that runs immediately before test collection (i.e. when pytest
# loads all the test cases to run). When running parallel tests via xdist on
# GPU or Cloud TPU, we use this hook to set the env vars needed to run multiple
# test processes across different chips.
#
# It's important that the hook runs before test collection, since jax tests end
# up initializing the TPU runtime on import (e.g. to query supported test
# types). It's also important that the hook gets called by each xdist worker
# process. Luckily each worker does its own test collection.
#
# The pytest_collection hook can be used to overwrite the collection logic, but
# we only use it to set the env vars and fall back to the default collection
# logic by always returning None. See
# https://docs.pytest.org/en/latest/how-to/writing_hook_functions.html#firstresult-stop-at-first-non-none-result
# for details.
#
# For TPU, the env var JAX_ENABLE_TPU_XDIST must be set for this hook to have an
# effect. We do this to minimize any effect on non-TPU tests, and as a pointer
# in test code to this "magic" hook. TPU tests should not specify more xdist
# workers than the number of TPU chips.
#
# For GPU, the env var JAX_ENABLE_CUDA_XDIST must be set equal to the number of
# CUDA devices. Test processes will be assigned in round robin fashion across
# the devices.
def pytest_collection() -> None:
  if os.environ.get("JAX_ENABLE_TPU_XDIST", None):
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault("TPU_VISIBLE_CHIPS", str(xdist_worker_number))
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")

  elif num_cuda_devices := os.environ.get("JAX_ENABLE_CUDA_XDIST", None):
    num_cuda_devices = int(num_cuda_devices)
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", str(xdist_worker_number % num_cuda_devices)
    )

  elif num_rocm_devices := os.environ.get("JAX_ENABLE_ROCM_XDIST", None):
    num_rocm_devices = int(num_rocm_devices)
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    allocated = os.environ.get("ROCR_VISIBLE_DEVICES")
    allocated_tokens = (
        [t.strip() for t in allocated.split(",") if t.strip()]
        if allocated
        else []
    )
    if allocated_tokens:
      selected = allocated_tokens[xdist_worker_number % len(allocated_tokens)]
    else:
      selected = str(xdist_worker_number % num_rocm_devices)
    os.environ["ROCR_VISIBLE_DEVICES"] = selected
    # ROCR_VISIBLE_DEVICES filters HSA to a single physical device, which
    # becomes HIP index 0. The container env-file may preset
    # HIP_VISIBLE_DEVICES to all GPUs; override to "0" so HIP doesn't try to
    # enable agents that ROCr just hid.
    os.environ["HIP_VISIBLE_DEVICES"] = "0"


# ==============================================================================
# Flake forensics -- opt-in, entirely inert unless JAX_FLAKE_FORENSICS=1.
#
# Used by the ROCm CI flake hunt (branch magaonka/pytest). The flakes we are
# chasing are rare (~1 test in ~400k executions), wrong-value rather than
# crash, and have never reproduced on an idle machine -- so the single CI
# failure we get has to carry as much structure as possible.
#
# Two things are collected, both of which cost nothing while tests pass:
#   1. On a failed array comparison, the *index structure* of the mismatching
#      elements (contiguous runs, alignment, per-axis spread, shift pairing).
#      A data-movement race should mark whole tiles/wavefront-aligned blocks;
#      an fp-precision problem should not. This runs only after a failure.
#   2. A test timeline held in memory on the xdist controller, used to print
#      which tests were co-resident on the GPU while a failed test ran. No
#      per-test I/O, so the timing race is not perturbed.
# ==============================================================================

_FORENSICS = os.environ.get("JAX_FLAKE_FORENSICS", "") == "1"
# Single greppable marker so every forensics block can be pulled out of a
# multi-megabyte GitHub job log with one rg.
_MARKER = ">>>FLAKE-FORENSICS<<<"
_MAX_SECTION_LINES = 200
_MAX_ANALYZED = 100_000


def _forensics_dir() -> str:
  d = os.environ.get("JAX_FLAKE_FORENSICS_DIR", "logs/forensics")
  os.makedirs(d, exist_ok=True)
  return d


def _recover_compared_arrays(excinfo):
  """Recovers (actual, desired, rtol, atol) from the numpy comparison frames.

  numpy raises the AssertionError from inside assert_array_compare, whose
  locals still hold both arrays, while the assert_allclose frame above it
  holds the tolerances and the arrays with their original shape (the inner
  frame may have masked/flattened them). Both are collected and the
  higher-rank pair wins, since per-axis structure is the whole point here.
  """
  import numpy as np

  candidates = []
  rtol = atol = None
  try:
    entries = list(excinfo.traceback)
  except Exception:
    return None, None, None, None
  for entry in reversed(entries):
    try:
      code_name = entry.frame.code.raw.co_name
      f_locals = entry.frame.f_locals
    except Exception:
      continue
    if rtol is None and "rtol" in f_locals and "atol" in f_locals:
      try:
        rtol = float(f_locals["rtol"])
        atol = float(f_locals["atol"])
      except Exception:
        rtol = atol = None
    if code_name == "assert_array_compare":
      candidates.append((f_locals.get("x"), f_locals.get("y")))
    if "actual" in f_locals and "desired" in f_locals:
      candidates.append((f_locals["actual"], f_locals["desired"]))

  best = (None, None)
  best_rank = -1
  for cand_a, cand_d in candidates:
    # Only already-materialized host arrays: never pull a jax.Array off the
    # device from here, which could block or fault on a sick GPU. numpy's own
    # assert_allclose has already converted its arguments by this point.
    if not (isinstance(cand_a, np.ndarray) and isinstance(cand_d, np.ndarray)):
      continue
    if cand_a.shape != cand_d.shape or cand_a.size == 0:
      continue
    if cand_a.ndim > best_rank:
      best, best_rank = (cand_a, cand_d), cand_a.ndim
  return best[0], best[1], rtol, atol


def _hist_line(label, values, top=5):
  import numpy as np

  vals, counts = np.unique(values, return_counts=True)
  order = np.argsort(-counts)[:top]
  body = ", ".join(f"{vals[i]}x{counts[i]}" for i in order)
  return f"  {label} ({vals.size} distinct, top{top}): {body}"


def _index_structure(idx, shape):
  """Describes the spatial structure of the mismatching flat indices."""
  import numpy as np

  lines = []
  breaks = np.flatnonzero(np.diff(idx) != 1)
  starts = np.concatenate(([0], breaks + 1))
  ends = np.concatenate((breaks, [idx.size - 1]))
  run_len = ends - starts + 1
  lines.append(
      f"  contiguous runs: {run_len.size} (len min/med/max ="
      f" {run_len.min()}/{int(np.median(run_len))}/{run_len.max()})"
  )
  lines.append(_hist_line("run-length hist", run_len))
  if run_len.size > 1:
    gaps = idx[starts[1:]] - idx[ends[:-1]] - 1
    lines.append(_hist_line("gap-between-runs hist", gaps))
  lines.append(f"  first run starts at flat idx {idx[0]}, last ends at {idx[-1]}")
  for m in (32, 64, 128, 256, 1024, 4096):
    res = np.unique(idx % m)
    extra = f" -> {res[:16].tolist()}" if res.size <= 16 else ""
    lines.append(f"  flat_idx % {m:<5d}: {res.size} distinct residues{extra}")
  try:
    multi = np.unravel_index(idx, shape)
  except Exception:
    return lines
  for axis, comp in enumerate(multi):
    uniq = np.unique(comp)
    extra = f", values={uniq[:16].tolist()}" if uniq.size <= 16 else ""
    lines.append(
        f"  axis {axis} (dim={shape[axis]}): {uniq.size} distinct,"
        f" min={comp.min()}, max={comp.max()}{extra}"
    )
  return lines


def _shift_pairing(flat_actual, flat_desired, idx, shape):
  """Tests whether the wrong values are the right values from another offset.

  A permutation-style corruption (misplaced elements) is explained by a
  constant flat-index shift; fp noise is not explained by any shift. Values
  are scored against a random-shift baseline, because in sparse 0/1 data
  (reshape jacobians) almost any shift "explains" a wrong zero by accident.
  """
  import numpy as np

  lines = []
  size = flat_actual.size
  sample = idx[:_MAX_ANALYZED]

  def explained(shift):
    src = sample + shift
    keep = (src >= 0) & (src < size)
    if keep.sum() < max(1, sample.size // 2):
      return None
    hits = np.isclose(flat_actual[sample[keep]], flat_desired[src[keep]],
                      rtol=1e-6, atol=1e-6, equal_nan=True)
    return float(hits.mean())

  rng = np.random.default_rng(0)
  baseline = [explained(int(s)) for s in rng.integers(1, max(2, size), size=8)]
  baseline = [b for b in baseline if b is not None]
  base = max(baseline) if baseline else 0.0

  strides = []
  acc = 1
  for dim in reversed(shape):
    strides.append(acc)
    acc *= int(dim)
  candidates = sorted(
      {s for s in strides if 0 < s < size}
      | {1 << k for k in range(1, 22) if (1 << k) < size}
  )
  scored = []
  for shift in candidates:
    for signed in (shift, -shift):
      frac = explained(signed)
      if frac is not None and frac > max(0.5, base + 0.2):
        scored.append((frac, signed))
  scored.sort(reverse=True)
  lines.append(f"  random-shift baseline (chance level): {base:.1%}")
  if scored:
    body = ", ".join(f"shift={s:+d} explains {f:.1%}" for f, s in scored[:4])
    lines.append(f"  constant-shift pairing: {body}")
  else:
    lines.append("  constant-shift pairing: no stride/power-of-2 shift beats chance")

  # Data-derived offsets: for each sampled bad element, where does its value
  # actually live in `desired`? Only informative (rare) values are used, so a
  # sea of zeros cannot dominate the histogram.
  window = 8192
  offsets = []
  informative = 0
  for i in sample[:: max(1, sample.size // 64)][:64]:
    lo = max(0, int(i) - window)
    hi = min(size, int(i) + window + 1)
    near = np.flatnonzero(np.isclose(flat_desired[lo:hi], flat_actual[i],
                                     rtol=1e-6, atol=1e-6, equal_nan=True))
    if near.size == 0 or near.size > 32:
      continue
    informative += 1
    offsets.extend((near + lo - int(i)).tolist())
  if informative >= 4 and offsets:
    vals, counts = np.unique(np.asarray(offsets), return_counts=True)
    order = np.argsort(-counts)[:5]
    if counts[order[0]] > 1:  # a single hit each way is coincidence, not signal
      body = ", ".join(f"{vals[i]:+d}x{counts[i]}" for i in order)
      lines.append(
          f"  where the wrong value lives in `desired` (offsets from {informative}"
          f" informative elements, top5): {body}")
  return lines


def _describe_mismatch(actual, desired, rtol, atol):
  import numpy as np

  lines = []
  a = np.asarray(actual)
  d = np.asarray(desired)
  lines.append(f"actual: shape={a.shape} dtype={a.dtype}")
  lines.append(f"desired: shape={d.shape} dtype={d.dtype}")
  if a.shape != d.shape:
    lines.append("SHAPE MISMATCH -- no element analysis possible")
    return lines
  if a.dtype == object or a.size == 0:
    lines.append("unsupported dtype / empty array -- skipping")
    return lines
  if rtol is None:
    rtol, atol = 1e-7, 0.0
  lines.append(f"tolerance used: rtol={rtol} atol={atol}")

  cast = np.complex128 if (np.iscomplexobj(a) or np.iscomplexobj(d)) else np.float64
  af = a.ravel().astype(cast)
  df = d.ravel().astype(cast)
  diff = np.abs(af - df)
  bad = diff > (atol + rtol * np.abs(df))
  bad |= np.isnan(af) != np.isnan(df)
  idx = np.flatnonzero(bad)
  if idx.size == 0:
    lines.append("no elements exceed tolerance (failure was not a value mismatch)")
    return lines

  lines.append(
      f"mismatched: {idx.size}/{af.size} ({100.0 * idx.size / af.size:.5f}%)"
  )
  finite = diff[np.isfinite(diff)]
  if finite.size:
    lines.append(f"max abs diff: {finite.max():.6g}")
  lines.append(_hist_line("abs-diff hist", np.round(diff[idx], 9)))
  if not np.iscomplexobj(af):
    lines.append(_hist_line("signed-diff hist", np.round((af - df)[idx].real, 9)))
    lines.append(_hist_line("desired-value hist at bad idx", np.round(df[idx].real, 9)))
    lines.append(_hist_line("actual-value hist at bad idx", np.round(af[idx].real, 9)))

  analyzed = idx[:_MAX_ANALYZED]
  if analyzed.size < idx.size:
    lines.append(f"(index structure computed on first {analyzed.size} indices)")
  lines.append("index structure:")
  lines.extend(_index_structure(analyzed, a.shape))
  lines.extend(_shift_pairing(af, df, analyzed, a.shape))

  lines.append("first mismatches (flat, multi, actual, desired):")
  for flat in idx[:24]:
    try:
      multi = tuple(int(c) for c in np.unravel_index(flat, a.shape))
    except Exception:
      multi = ()
    lines.append(f"  {int(flat):>10d} {multi} actual={af[flat]} desired={df[flat]}")
  return lines


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
  """Attaches mismatch forensics to the failure report.

  This has to be a makereport wrapper rather than pytest_exception_interact:
  under xdist the report is serialized and shipped to the controller inside
  pytest_runtest_logreport, which runs *before* exception_interact, so a
  section added there never reaches the job log.
  """
  outcome = yield
  if not _FORENSICS:
    return
  try:
    report = outcome.get_result()
  except Exception:
    return
  if report.when != "call" or not report.failed or call.excinfo is None:
    return
  try:
    header = [
        f"{_MARKER} array-mismatch forensics",
        f"nodeid: {report.nodeid}",
        f"host={socket.gethostname()} pid={os.getpid()}"
        f" worker={os.environ.get('PYTEST_XDIST_WORKER', 'main')}"
        f" ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', '')}"
        f" HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')}",
        f"wallclock: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"exception: {call.excinfo.typename}",
    ]
    actual, desired, rtol, atol = _recover_compared_arrays(call.excinfo)
    if actual is None or desired is None:
      body = ["could not recover compared arrays from the traceback frames"]
    else:
      body = _describe_mismatch(actual, desired, rtol, atol)
    lines = header + body
  except Exception as exc:  # forensics must never mask the real failure
    lines = [f"{_MARKER} forensics failed: {type(exc).__name__}: {exc}"]

  text = "\n".join(lines)
  try:
    safe = report.nodeid.replace("/", "_").replace(":", "_")[:150]
    path = os.path.join(_forensics_dir(), f"mismatch-{safe}-{os.getpid()}.txt")
    with open(path, "w") as f:
      f.write(text + "\n")
  except Exception:
    pass
  if len(lines) > _MAX_SECTION_LINES:
    lines = lines[:_MAX_SECTION_LINES] + ["... (truncated; full copy in logs/forensics)"]
  report.sections.append(("flake forensics", "\n".join(lines)))


class _TestTimeline:
  """Controller-side record of when each test ran, for co-residency analysis."""

  def __init__(self):
    self.records = []

  def pytest_runtest_logreport(self, report):
    if report.when != "call":
      return
    node = getattr(report, "node", None)
    worker = getattr(getattr(node, "gateway", None), "id", None)
    if not worker:
      worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    self.records.append({
        "nodeid": report.nodeid,
        "worker": worker,
        "outcome": report.outcome,
        "start": getattr(report, "start", None),
        "stop": getattr(report, "stop", None),
    })

  def pytest_terminal_summary(self, terminalreporter):
    if not self.records:
      return
    tag = os.environ.get("JAX_FLAKE_FORENSICS_TAG", "run")
    try:
      path = os.path.join(_forensics_dir(), f"timeline-{tag}.jsonl")
      with open(path, "w") as f:
        for rec in self.records:
          f.write(json.dumps(rec) + "\n")
    except Exception:
      pass

    failed = [r for r in self.records if r["outcome"] == "failed"]
    if not failed:
      return
    write = terminalreporter.write_line
    for rec in failed[:10]:
      start, stop = rec["start"], rec["stop"]
      write(f"{_MARKER} co-residency for FAILED {rec['nodeid']} [{rec['worker']}]")
      if start is None or stop is None:
        write("  no timing information available")
        continue
      write(f"  ran {stop - start:.2f}s, {time.strftime('%H:%M:%S', time.gmtime(start))}"
            f" -> {time.strftime('%H:%M:%S', time.gmtime(stop))} UTC")
      overlap = [
          o for o in self.records
          if o is not rec and o["start"] is not None and o["stop"] is not None
          and o["start"] < stop and o["stop"] > start
      ]
      write(f"  {len(overlap)} test(s) overlapped this window:")
      for o in sorted(overlap, key=lambda o: o["start"])[:40]:
        write(f"    [{o['worker']}] {o['stop'] - o['start']:7.2f}s {o['nodeid']}")
      if len(overlap) > 40:
        write(f"    ... and {len(overlap) - 40} more")


def pytest_configure(config):
  if _FORENSICS:
    config.pluginmanager.register(_TestTimeline(), "jax_flake_timeline")
