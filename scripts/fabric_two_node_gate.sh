#!/usr/bin/env bash
# =============================================================================
# fabric_two_node_gate.sh
#
# the two-node cpu gate for `simbi launch`: two scheduler-started workers on
# two distinct nodes evolve a 2D cartesian newtonian problem over the fabric to
# a fixed end time, and their final checkpoint is compared, interior cell for
# interior cell, against `simbi run` of the same problem to the same end time
# on one node. the script is the batch step of a two-node allocation: it
# writes the layout and the task step once into the shared output directory,
# starts one task per node, runs the reference, and compares. it records each worker's hostname, backend hash, and exit code and
# fails unless the workers occupied two distinct nodes, every exit code is
# zero, every backend hash equals the reference's, and each arm left exactly
# one final checkpoint in a fresh output directory on a shared filesystem.
#
# usage:
#   sbatch --nodes=2 --ntasks-per-node=1 scripts/fabric_two_node_gate.sh /shared/fresh-dir
#   (or the same command line inside `salloc --nodes=2 --ntasks-per-node=1`)
#
# environment: CONFIG (a 2D cartesian newtonian config), RESOLUTION ("nx,ny",
# nx even), END_TIME.
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-simbi_configs/examples/newtonian/kh.py}"
RESOLUTION="${RESOLUTION:-32,32}"
END_TIME="${END_TIME:-0.01}"
PROBLEM_FLAGS=(--resolution "$RESOLUTION" --end-time "$END_TIME" --checkpoint-interval 1e9)

die() { echo "two-node gate: $*" >&2; exit 1; }

# ---- the batch step ----------------------------------------------------------
OUT="${1:?usage: fabric_two_node_gate.sh <fresh output directory on a shared filesystem>}"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside a two-node slurm allocation (sbatch or salloc)"
[[ "${SLURM_JOB_NUM_NODES:-0}" -ge 2 ]] || die "the allocation has ${SLURM_JOB_NUM_NODES:-0} nodes; two are required"
if [[ -e "$OUT" ]]; then
    [[ -d "$OUT" && -z "$(ls -A "$OUT")" ]] || die "$OUT exists and is not empty; give a fresh directory"
fi
mkdir -p "$OUT/records"
OUT="$(cd "$OUT" && pwd)"

# the layout, written once, before any worker exists
cat > "$OUT/two-node.toml" <<TOML
[execution]
mode = "scheduler"
workers = 2
bind = "0.0.0.0"

[partition]
shape = [2, 1]

[checkpoint]
directory = "$OUT/launched"
staging_mb = 4
TOML

# the task step, written to the shared output directory: a batch script lives in the batch
# node's spool directory alone, so the other node cannot execute it by path
CONFIG_ABS="$CONFIG"; [[ -e "$CONFIG" ]] && CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
cat > "$OUT/task.sh" <<TASK
#!/usr/bin/env bash
set -uo pipefail
cd "$PWD"
rank="\${SLURM_PROCID:?the task step runs under srun}"
hostname > "$OUT/records/worker-\$rank.host"
python -m simbi.cli launch "$CONFIG_ABS" --layout "$OUT/two-node.toml" ${PROBLEM_FLAGS[*]} \\
    --data-directory "$OUT/launched" > "$OUT/records/worker-\$rank.out" 2> "$OUT/records/worker-\$rank.err"
code=\$?
echo "\$code" > "$OUT/records/worker-\$rank.exit"
exit "\$code"
TASK
chmod +x "$OUT/task.sh"

set +e
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 "$OUT/task.sh"
srun_code=$?
set -e
echo "$srun_code" > "$OUT/records/srun.exit"

# the reference: the public single-grid command on this node, its streams kept
set +e
python -m simbi.cli run "$CONFIG_ABS" --mode cpu "${PROBLEM_FLAGS[@]}" --data-directory "$OUT/single" \
    > "$OUT/records/reference.out" 2> "$OUT/records/reference.err"
reference_code=$?
set -e
echo "$reference_code" > "$OUT/records/reference.exit"
if [[ "$reference_code" -ne 0 ]]; then
    tail -n 40 "$OUT/records/reference.err" >&2
    die "the reference run exited $reference_code"
fi

python - "$OUT" "$srun_code" <<'PY'
import glob, re, sys
from pathlib import Path

import h5py
import numpy as np

out, srun_code = Path(sys.argv[1]), int(sys.argv[2])
records = out / "records"
failures = []

hosts = {k: (records / f"worker-{k}.host").read_text().strip() for k in (0, 1) if (records / f"worker-{k}.host").exists()}
exits = {k: int((records / f"worker-{k}.exit").read_text()) for k in (0, 1) if (records / f"worker-{k}.exit").exists()}
provenance = re.compile(r"backend=([0-9a-f]{40}(?:-dirty)?) config=")
def backend(path):
    found = set(provenance.findall(path.read_text(errors="replace"))) if path.exists() else set()
    return next(iter(found)) if len(found) == 1 else None
backends = {f"worker-{k}": backend(records / f"worker-{k}.err") for k in (0, 1)}
backends["reference"] = backend(records / "reference.err")

print("two-node gate records")
for k in (0, 1):
    print(f"  worker {k}: host {hosts.get(k)!r} exit {exits.get(k)!r} backend {backends[f'worker-{k}']!r}")
print(f"  reference: backend {backends['reference']!r}; srun exit {srun_code}")

if len(hosts) != 2 or len(set(hosts.values())) != 2:
    failures.append(f"the workers did not occupy two distinct nodes: {hosts}")
if exits != {0: 0, 1: 0} or srun_code != 0:
    failures.append(f"worker exit codes {exits}, srun exit {srun_code}")
if None in backends.values() or len(set(backends.values())) != 1:
    failures.append(f"backend hashes disagree or are missing: {backends}")

finals = {}
for arm in ("launched", "single"):
    found = sorted(glob.glob(str(out / arm / "*final*.h5")))
    if len(found) != 1:
        failures.append(f"{arm}: expected exactly one final checkpoint, found {found}")
    else:
        finals[arm] = found[0]

if not failures:
    with h5py.File(finals["launched"]) as fa, h5py.File(finals["single"]) as fb:
        la, lb = fa["level_0"], fb["level_0"]
        for key in ("time", "iteration"):
            if la.attrs[key] != lb.attrs[key]:
                failures.append(f"level attribute {key}: {la.attrs[key]} against {lb.attrs[key]}")
        ng = int(la["mesh"].attrs["halo_width"])
        names = []
        la.visit(lambda n: names.append(n) if isinstance(la[n], h5py.Dataset) else None)
        checked = 0
        for n in names:
            if n.startswith("mesh/") or n.startswith("partition_0/owned") or "/domain/" in n:
                continue
            x = la[n][()]
            if x.ndim != 2:
                continue
            if n not in lb:
                failures.append(f"{n} is missing from the reference")
                continue
            y = lb[n][()]
            if not np.array_equal(x[ng:-ng, ng:-ng], y[ng:-ng, ng:-ng]):
                failures.append(f"{n} differs by {np.abs(x - y)[ng:-ng, ng:-ng].max():.3e}")
            checked += 1
        if checked == 0:
            failures.append("no field datasets were compared")
        elif not failures:
            print(f"  {checked} datasets equal at t = {la.attrs['time']}, iteration {int(la.attrs['iteration'])}")

if failures:
    print("two-node gate: FAILED", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    for k in (0, 1):
        err = records / f"worker-{k}.err"
        if err.exists() and exits.get(k) != 0:
            print(f"--- worker {k} stderr (tail)\n" + "\n".join(err.read_text(errors='replace').splitlines()[-30:]), file=sys.stderr)
    sys.exit(1)
print("two-node gate: PASSED")
PY
