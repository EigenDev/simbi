#!/usr/bin/env bash
# =============================================================================
# ks_excision_assessment.sh
#
# the bounded before/after measurement of an excision or correction change on
# the python test_cartesian_ks_mhd_3d fixture. the baseline revision is built in
# an isolated git worktree with its own cargo target directory; the candidate is
# the active tree, which this script only reads. the active tree is never
# stashed, checked out or written. each arm (revision x edge EMF) runs as a
# fresh process, so the process-global fofc horizon radius cannot leak between
# simulations, under the harness's step cap and wall-clock watchdog and an
# outer kill of the whole process group. the output directory receives the
# per-step logs, the states at t = 1, provenance.txt (revisions, dirty state,
# commands, configuration) and comparison.txt.
#
# usage:
#  scripts/ks_excision_assessment.sh <baseline-rev> <out-dir>
# environment: KS_MAX_STEPS (400), KS_WALL_SECONDS (1200), KS_KILL_SECONDS (1500),
# PYTHON (python3; the comparison needs numpy)
# the baseline revision must contain crates/symbi/tests/ks_excision_assessment.rs's
# dependencies; the harness file itself is copied into the baseline worktree.
# the worktree and its target directory stay under <out-dir> for a rerun; remove them with
#  git worktree remove --force <out-dir>/baseline-src && rm -rf <out-dir>/target-baseline
# =============================================================================
set -euo pipefail

base_rev=${1:?usage: ks_excision_assessment.sh <baseline-rev> <out-dir>}
out=${2:?usage: ks_excision_assessment.sh <baseline-rev> <out-dir>}
repo=$(git rev-parse --show-toplevel)
mkdir -p "$out"
out=$(cd "$out" && pwd)
max_steps=${KS_MAX_STEPS:-400}
wall=${KS_WALL_SECONDS:-1200}
kill_after=${KS_KILL_SECONDS:-1500}
harness=src/crates/symbi/tests/ks_excision_assessment.rs

# the baseline in its own worktree and target directory
base_tree="$out/baseline-src"
if [ ! -d "$base_tree" ]; then
    git -C "$repo" worktree add --detach "$base_tree" "$base_rev" >/dev/null
fi
cp "$repo/$harness" "$base_tree/$harness"

build() { # <source-root> <target-dir> -> path of the harness executable
    (cd "$1/src" && CARGO_INCREMENTAL=0 CARGO_TARGET_DIR="$2" cargo test -p symbi --test ks_excision_assessment --no-run 2>&1 |
        grep -o "deps/ks_excision_assessment-[0-9a-f]*" | tail -1 | sed "s|^|$2/debug/|")
}
base_exe=$(build "$base_tree" "$out/target-baseline")
cand_exe=$(build "$repo" "$repo/src/target")
cp "$base_exe" "$out/harness-baseline"
cp "$cand_exe" "$out/harness-candidate"

{
    echo "baseline: $(git -C "$base_tree" rev-parse HEAD) ($base_rev)"
    echo "candidate: $(git -C "$repo" rev-parse HEAD) dirty=$(git -C "$repo" status --porcelain --untracked-files=no | wc -l | tr -d ' ') files"
    echo "harness: $harness (sha1 $(shasum "$repo/$harness" | cut -d' ' -f1))"
    echo "limits: max_steps=$max_steps wall=${wall}s kill=${kill_after}s"
    echo "command: $0 $*"
} >"$out/provenance.txt"

for tag in baseline candidate; do
    for ct in contact uct; do
        python3 - "$out/harness-$tag" "$out/${tag}_$ct" "$ct" "$max_steps" "$wall" "$kill_after" <<'EOF'
import os, signal, subprocess, sys, time
exe, stem, ct, steps, wall, kill_after = sys.argv[1:]
env = dict(os.environ, KS_CT=ct, KS_MAX_STEPS=steps, KS_WALL_SECONDS=wall, KS_DUMP=stem + ".bin")
start = time.time()
with open(stem + ".log", "w") as log:
    proc = subprocess.Popen([exe, "--ignored", "--nocapture", "--test-threads=1"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=float(kill_after))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        code = f"killed after {kill_after} s"
print(f"{os.path.basename(stem)}: exit {code} in {time.time() - start:.0f} s")
EOF
        grep -ho "CONFIG.*" "$out/${tag}_$ct.log" | head -1 | sed "s/^/${tag}_$ct /" >>"$out/provenance.txt" || true
    done
done
"${PYTHON:-python3}" "$repo/scripts/ks_excision_compare.py" "$out" baseline candidate | tee "$out/comparison.txt"
