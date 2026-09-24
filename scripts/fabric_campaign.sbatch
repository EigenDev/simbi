#!/usr/bin/env bash
# =============================================================================
# fabric_campaign.sbatch
#
# the batch step of the launch validation campaign on a slurm cluster: one
# allocation, one installation, the smoke tier first and the full tier after
# it, each into its own fresh directory under the shared output root. every
# arm of the campaign is one srun inside this allocation, so the allocation
# must hold at least as many tasks as the widest layout in the case table
# (four) and the cpus per task the workers are meant to run on.
#
# usage:
#   sbatch --nodes=2 --ntasks-per-node=2 --cpus-per-task=16 --time=02:00:00 \
#       scripts/fabric_campaign.sbatch /shared/fresh-root
#
# environment: SIMBI_ADVERTISE when a node's hostname does not resolve from
# its peers (the campaign exports every SIMBI_* variable into each task).
# =============================================================================
set -euo pipefail
root="${1:?usage: fabric_campaign.sbatch <fresh root directory on a shared filesystem>}"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "run inside a slurm allocation" >&2; exit 1; }
mkdir -p "$root"
python scripts/fabric_campaign.py "$root/smoke" --mode scheduler --tier smoke
python scripts/fabric_campaign.py "$root/full" --mode scheduler --tier full
