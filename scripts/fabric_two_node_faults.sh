#!/usr/bin/env bash
# =============================================================================
# fabric_two_node_faults.sh
#
# the cross-node fault and timing gate for `simbi launch`, as the batch step
# of a two-node allocation. four arms, each in its own directory under the
# output directory, each with two scheduler-started workers on two nodes:
#
#   rejection  worker 1 reports a stage rejection at a named step. every worker
#              rolls back to the same halved step; the result equals one worker
#              driven through the same rejection, interior cell for interior
#              cell, and every worker reports exactly one rejection.
#   bad_cfl    worker 1 reports an invalid timestep candidate. every worker
#              exits nonzero within the limit and no final checkpoint appears.
#   lost       worker 1 ends as a lost node would after a named step. worker 0
#              exits nonzero with a disconnect within the limit and no final
#              checkpoint appears.
#   timing     a larger grid for a fixed number of steps on two nodes and on
#              one worker; wall times and each worker's compute, collective,
#              and exchange times are recorded. equal results are required,
#              the times are recorded and not judged.
#
# tasks run with --kill-on-bad-exit=0 so the fabric's own termination ends the
# surviving worker, and slurm's does not. the faults come from
# SIMBI_FABRIC_INJECT, which the worker reads at startup. the records of every
# arm and the source provenance are archived as faults-archive.tar.gz.
#
# usage:
#   sbatch --nodes=2 --ntasks-per-node=1 --cpus-per-task=<cores per node> \
#       scripts/fabric_two_node_faults.sh /shared/fresh-dir
#
# the cpus per task matter: srun gives a task one cpu unless the job names
# more, and a worker whose thread pool shares one core reports compute and
# collective times that measure the starvation. each task records its cpus.
#
# environment: CONFIG (a 2D cartesian newtonian config), RESOLUTION, STEPS,
# FAULT_STEP, TIMING_RESOLUTION, TIMING_STEPS, LIMIT_SECONDS.
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-simbi_configs/examples/newtonian/kh.py}"
RESOLUTION="${RESOLUTION:-32,32}"
STEPS="${STEPS:-8}"
FAULT_STEP="${FAULT_STEP:-3}"
TIMING_RESOLUTION="${TIMING_RESOLUTION:-1024,1024}"
TIMING_STEPS="${TIMING_STEPS:-40}"
LIMIT_SECONDS="${LIMIT_SECONDS:-120}"

die() { echo "two-node faults: $*" >&2; exit 1; }

OUT="${1:?usage: fabric_two_node_faults.sh <fresh output directory on a shared filesystem>}"
[[ -n "${SLURM_JOB_ID:-}" ]] || die "run inside a two-node slurm allocation (sbatch or salloc)"
[[ "${SLURM_JOB_NUM_NODES:-0}" -ge 2 ]] || die "the allocation has ${SLURM_JOB_NUM_NODES:-0} nodes; two are required"
if [[ -e "$OUT" ]]; then
    [[ -d "$OUT" && -z "$(ls -A "$OUT")" ]] || die "$OUT exists and is not empty; give a fresh directory"
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CONFIG_ABS="$CONFIG"; [[ -e "$CONFIG" ]] && CONFIG_ABS="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

archive() { tar czf "$OUT/faults-archive.tar.gz" -C "$OUT" --exclude='*.h5' . 2>/dev/null || true; }
trap archive EXIT

{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "slurm job: ${SLURM_JOB_ID} nodes: ${SLURM_JOB_NODELIST:-unknown}"
    echo "config: $CONFIG resolution: $RESOLUTION steps: $STEPS fault step: $FAULT_STEP"
    echo "timing resolution: $TIMING_RESOLUTION timing steps: $TIMING_STEPS limit: ${LIMIT_SECONDS}s"
    echo "--- git rev-parse HEAD"; git rev-parse HEAD
    echo "--- git status --short"; git status --short
    echo "--- git show --no-patch --format=fuller HEAD"; git show --no-patch --format=fuller HEAD
} > "$OUT/provenance.txt" 2>&1 || true

# two scheduler-started workers on two nodes: <arm> <fault or none> <resolution> <steps>
two_nodes() {
    local arm="$1" fault="$2" resolution="$3" steps="$4" dir="$OUT/$1"
    mkdir -p "$dir/records"
    cat > "$dir/layout.toml" <<TOML
[execution]
mode = "scheduler"
workers = 2
bind = "0.0.0.0"

[partition]
shape = [2, 1]

[checkpoint]
directory = "$dir/launched"
staging_mb = 4
TOML
    cat > "$dir/task.sh" <<TASK
#!/usr/bin/env bash
set -uo pipefail
cd "$PWD"
rank="\${SLURM_PROCID:?the task step runs under srun}"
hostname > "$dir/records/worker-\$rank.host"
# the cpus this task may run on: a worker starved to one core inflates every time it reports
{ echo "nproc: \$(nproc)"; grep -i cpus_allowed_list /proc/self/status 2>/dev/null; echo "SLURM_CPUS_PER_TASK: \${SLURM_CPUS_PER_TASK:-unset}"; } > "$dir/records/worker-\$rank.cpus"
[[ "$fault" != none ]] && export SIMBI_FABRIC_INJECT="$fault"
start=\$(python -c "import time; print(time.time())")
python -m simbi.cli launch "$CONFIG_ABS" --layout "$dir/layout.toml" --resolution "$resolution" \\
    --max-steps "$steps" --checkpoint-interval 1e9 --data-directory "$dir/launched" \\
    > "$dir/records/worker-\$rank.out" 2> "$dir/records/worker-\$rank.err"
code=\$?
end=\$(python -c "import time; print(time.time())")
echo "\$code" > "$dir/records/worker-\$rank.exit"
echo "\$start \$end" > "$dir/records/worker-\$rank.span"
exit "\$code"
TASK
    chmod +x "$dir/task.sh"
    set +e
    # every cpu the job holds per task, which srun's default of one cpu per task would withhold
    srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        --kill-on-bad-exit=0 "$dir/task.sh"
    echo "$?" > "$dir/records/srun.exit"
    set -e
}

# one worker on this node through the same command: <arm> <fault or none> <resolution> <steps>
one_worker() {
    local arm="$1" fault="$2" resolution="$3" steps="$4" dir="$OUT/$1"
    mkdir -p "$dir/records"
    cat > "$dir/one.toml" <<TOML
[execution]
workers = 1

[partition]
shape = [1, 1]

[checkpoint]
staging_mb = 4
TOML
    local start end code
    { echo "nproc: $(nproc)"; grep -i cpus_allowed_list /proc/self/status 2>/dev/null; echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"; } > "$dir/records/reference.cpus"
    start=$(python -c "import time; print(time.time())")
    set +e
    if [[ "$fault" != none ]]; then
        SIMBI_FABRIC_INJECT="$fault" python -m simbi.cli launch "$CONFIG_ABS" --layout "$dir/one.toml" \
            --resolution "$resolution" --max-steps "$steps" --checkpoint-interval 1e9 \
            --data-directory "$dir/single" > "$dir/records/reference.out" 2> "$dir/records/reference.err"
    else
        python -m simbi.cli launch "$CONFIG_ABS" --layout "$dir/one.toml" \
            --resolution "$resolution" --max-steps "$steps" --checkpoint-interval 1e9 \
            --data-directory "$dir/single" > "$dir/records/reference.out" 2> "$dir/records/reference.err"
    fi
    code=$?
    set -e
    end=$(python -c "import time; print(time.time())")
    echo "$code" > "$dir/records/reference.exit"
    echo "$start $end" > "$dir/records/reference.span"
}

two_nodes rejection "reject:1:$FAULT_STEP:1" "$RESOLUTION" "$STEPS"
one_worker rejection "reject:0:$FAULT_STEP:1" "$RESOLUTION" "$STEPS"
two_nodes bad_cfl "bad_cfl:1:$FAULT_STEP" "$RESOLUTION" "$STEPS"
two_nodes lost "exit:1:$FAULT_STEP" "$RESOLUTION" "$STEPS"
two_nodes timing none "$TIMING_RESOLUTION" "$TIMING_STEPS"
one_worker timing none "$TIMING_RESOLUTION" "$TIMING_STEPS"

python - "$OUT" "$LIMIT_SECONDS" <<'PY'
import glob, re, sys
from pathlib import Path

import h5py
import numpy as np

out, limit = Path(sys.argv[1]), float(sys.argv[2])
failures: list[str] = []
summary = re.compile(r"SIMBI launch: worker=(\d+)/(\d+) steps=(\d+) rejections=(\d+) compute=([\d.]+)s collectives=([\d.]+)s exchange=([\d.]+)s final_checkpoint=([\d.]+)s")
provenance = re.compile(r"backend=([0-9a-f]{40}(?:-dirty)?) config=")


def text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def records(arm: str) -> dict:
    rec = out / arm / "records"
    info = {"hosts": {}, "exits": {}, "spans": {}, "err": {}, "summaries": {}, "backends": {}}
    for k in (0, 1):
        if (rec / f"worker-{k}.host").exists():
            info["hosts"][k] = text(rec / f"worker-{k}.host").strip()
        if (rec / f"worker-{k}.exit").exists():
            info["exits"][k] = int(text(rec / f"worker-{k}.exit"))
        if (rec / f"worker-{k}.span").exists():
            a, b = text(rec / f"worker-{k}.span").split()
            info["spans"][k] = float(b) - float(a)
        info["err"][k] = text(rec / f"worker-{k}.err")
        found = summary.search(info["err"][k])
        if found:
            info["summaries"][k] = found.groups()
        hashes = set(provenance.findall(info["err"][k]))
        info["backends"][k] = next(iter(hashes)) if len(hashes) == 1 else None
    info["reference_err"] = text(rec / "reference.err")
    info["reference_exit"] = int(text(rec / "reference.exit")) if (rec / "reference.exit").exists() else None
    if (rec / "reference.span").exists():
        a, b = text(rec / "reference.span").split()
        info["reference_span"] = float(b) - float(a)
    info["finals"] = sorted(glob.glob(str(out / arm / "launched" / "*final*.h5")))
    info["reference_finals"] = sorted(glob.glob(str(out / arm / "single" / "*final*.h5")))
    return info


def distinct_nodes(arm: str, info: dict) -> None:
    if len(info["hosts"]) != 2 or len(set(info["hosts"].values())) != 2:
        failures.append(f"{arm}: the workers did not occupy two distinct nodes: {info['hosts']}")


def same_state(arm: str, a: str, b: str) -> int:
    checked = 0
    with h5py.File(a) as fa, h5py.File(b) as fb:
        la, lb = fa["level_0"], fb["level_0"]
        for key in ("time", "iteration"):
            if la.attrs[key] != lb.attrs[key]:
                failures.append(f"{arm}: level attribute {key}: {la.attrs[key]} against {lb.attrs[key]}")
        ng = int(la["mesh"].attrs["halo_width"])
        names: list[str] = []
        la.visit(lambda n: names.append(n) if isinstance(la[n], h5py.Dataset) else None)
        for n in names:
            if n.startswith("mesh/") or n.startswith("partition_0/owned") or "/domain/" in n:
                continue
            x = la[n][()]
            if x.ndim != 2:
                continue
            y = lb[n][()]
            if not np.array_equal(x[ng:-ng, ng:-ng], y[ng:-ng, ng:-ng]):
                failures.append(f"{arm}: {n} differs by {np.abs(x - y)[ng:-ng, ng:-ng].max():.3e}")
            checked += 1
    if checked == 0:
        failures.append(f"{arm}: no field datasets were compared")
    return checked


def equal_arm(arm: str, expect_rejections: int | None) -> dict:
    info = records(arm)
    distinct_nodes(arm, info)
    if info["exits"] != {0: 0, 1: 0} or info["reference_exit"] != 0:
        failures.append(f"{arm}: exits {info['exits']}, reference {info['reference_exit']}")
    if len(info["finals"]) != 1 or len(info["reference_finals"]) != 1:
        failures.append(f"{arm}: finals {info['finals']} against {info['reference_finals']}")
    else:
        info["checked"] = same_state(arm, info["finals"][0], info["reference_finals"][0])
    reference = summary.search(info["reference_err"])
    info["reference_summary"] = reference.groups() if reference else None
    if expect_rejections is not None:
        counts = {k: int(v[3]) for k, v in info["summaries"].items()}
        ref = int(reference.group(4)) if reference else None
        if counts != {0: expect_rejections, 1: expect_rejections} or ref != expect_rejections:
            failures.append(f"{arm}: rejections {counts} against reference {ref}; expected {expect_rejections} on every worker")
    return info


def failed_arm(arm: str, worker_1: str, worker_0: tuple[str, ...], exit_1: int | None) -> dict:
    info = records(arm)
    distinct_nodes(arm, info)
    if len(info["exits"]) != 2 or any(code == 0 for code in info["exits"].values()):
        failures.append(f"{arm}: every worker must exit nonzero: {info['exits']}")
    if exit_1 is not None and info["exits"].get(1) != exit_1:
        failures.append(f"{arm}: worker 1 exit {info['exits'].get(1)}, expected {exit_1}")
    if worker_1 and worker_1 not in info["err"][1]:
        failures.append(f"{arm}: worker 1 stderr lacks {worker_1!r}")
    if not any(word in info["err"][0] for word in worker_0):
        failures.append(f"{arm}: worker 0 stderr names none of {worker_0}")
    if info["finals"]:
        failures.append(f"{arm}: a final checkpoint was published: {info['finals']}")
    slow = {k: round(v, 1) for k, v in info["spans"].items() if v > limit}
    if slow:
        failures.append(f"{arm}: termination took {slow} seconds against a limit of {limit}")
    return info


rejection = equal_arm("rejection", expect_rejections=1)
bad_cfl = failed_arm("bad_cfl", "invalid CFL candidate", ("aborted", "disconnected"), None)
lost = failed_arm("lost", "", ("disconnected",), 86)
timing = equal_arm("timing", expect_rejections=0)

backends = {b for arm in (rejection, bad_cfl, lost, timing) for b in arm["backends"].values() if b}
if len(backends) != 1:
    failures.append(f"the workers report backends {backends}")

print("two-node faults records")
for name, info in (("rejection", rejection), ("bad_cfl", bad_cfl), ("lost", lost), ("timing", timing)):
    spans = {k: round(v, 2) for k, v in info["spans"].items()}
    print(f"  {name}: hosts {info['hosts']} exits {info['exits']} seconds {spans}")
print(f"  backend: {sorted(backends)}")
def cpus(path: Path) -> str:
    return " ".join(text(path).split()) or "unrecorded"

print("cpus")
for k in (0, 1):
    print(f"  two nodes, worker {k}: {cpus(out / 'timing' / 'records' / f'worker-{k}.cpus')}")
print(f"  one worker: {cpus(out / 'timing' / 'records' / 'reference.cpus')}")
print("timing (seconds)")
for k, s in sorted(timing["summaries"].items()):
    print(f"  two nodes, worker {k}: steps {s[2]} compute {s[4]} collectives {s[5]} exchange {s[6]} final checkpoint {s[7]}; process wall {timing['spans'].get(k, float('nan')):.2f}")
if timing.get("reference_summary"):
    s = timing["reference_summary"]
    print(f"  one worker: steps {s[2]} compute {s[4]} collectives {s[5]} exchange {s[6]} final checkpoint {s[7]}; process wall {timing.get('reference_span', float('nan')):.2f}")

if failures:
    print("two-node faults: FAILED", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    sys.exit(1)
print("two-node faults: PASSED")
PY
