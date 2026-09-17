#!/usr/bin/env bash
# =============================================================================
# fabric_two_node_gate.sh
#
# the two-node cpu gate for `simbi launch`: two scheduler-started workers on
# two nodes evolve the kelvin-helmholtz example over the fabric and their
# final checkpoint is compared, interior cell for interior cell, against a
# single-grid run of the same problem on one node. the layout binds every
# worker on all interfaces and advertises the node's hostname; the
# coordinator's advertised address and the session credential travel through
# the restricted rendezvous file on the shared output directory.
#
# usage (inside an allocation of two nodes, one task each):
#   sbatch --nodes=2 --ntasks-per-node=1 scripts/fabric_two_node_gate.sh /shared/out
#   or: OUT=/shared/out srun --nodes=2 --ntasks=2 scripts/fabric_two_node_gate.sh
# =============================================================================
set -euo pipefail
OUT="${1:-${OUT:-$PWD/fabric-two-node}}"
STEPS="${STEPS:-8}"
CONFIG="${CONFIG:-simbi_configs/examples/newtonian/kh.py}"
mkdir -p "$OUT"
LAYOUT="$OUT/two-node.toml"
cat > "$LAYOUT" <<TOML
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

if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_PROCID:-}" ]]; then
    # the batch step: fan out one task per node, then compare on this node
    srun --nodes=2 --ntasks=2 "$0" "$OUT"
    python -m simbi.cli run "$CONFIG" --mode cpu --resolution 32,32 --checkpoint-interval 1e9 \
        --data-directory "$OUT/single" --end-time 1e9 --max-steps "$STEPS" 2>/dev/null || \
    python - "$CONFIG" "$OUT/single" "$STEPS" <<'PY'
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz
runner.run(KelvinHelmholtz(resolution=(32, 32), data_directory=sys.argv[2], checkpoint_interval=1e9), compute_mode="cpu", max_steps=int(sys.argv[3]))
PY
    python - "$OUT" <<'PY'
import sys, glob
import h5py, numpy as np
out = sys.argv[1]
a = glob.glob(f"{out}/launched/*final*.h5")[0]
b = glob.glob(f"{out}/single/*final*.h5")[0]
with h5py.File(a) as fa, h5py.File(b) as fb:
    la, lb = fa["level_0"], fb["level_0"]
    assert la.attrs["time"] == lb.attrs["time"] and la.attrs["iteration"] == lb.attrs["iteration"]
    ng = int(la["mesh"].attrs["halo_width"])
    names = []
    la.visit(lambda n: names.append(n) if isinstance(la[n], h5py.Dataset) else None)
    checked = 0
    for n in names:
        if n.startswith("mesh/") or n.startswith("partition_0/owned") or "/domain/" in n:
            continue
        x, y = la[n][()], lb[n][()]
        if x.ndim == 2:
            np.testing.assert_array_equal(x[ng:-ng, ng:-ng], y[ng:-ng, ng:-ng], err_msg=n)
            checked += 1
    assert checked > 0
    print(f"two-node gate: {checked} datasets equal at t = {la.attrs['time']}")
PY
    exit 0
fi

# the task step: one worker per node, identity from SLURM_PROCID / SLURM_NTASKS
exec python -m simbi.cli launch "$CONFIG" --layout "$LAYOUT" --max-steps "$STEPS" \
    --resolution 32,32 --checkpoint-interval 1e9 --data-directory "$OUT/launched"
