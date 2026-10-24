// =============================================================================
// device_identity.rs
//
// the physical gpu of every launched worker, agreed by the whole session before
// any state moves.
// - a worker's binding is its host name and its gpu's pci address, a host build's
//   is the absence of a device, and a worker whose gpu runtime refused the binding
//   reports that refusal
// - one assembled collective gives every worker every binding
// - two workers on one host at one pci address, or any refused binding, end the
//   session on every worker at that collective; one pci address on two hosts names
//   two gpus
// usage:
//   agree_device_bindings(&mut fabric, me, workers, &local, deadline)?;
// =============================================================================

use crate::worker::{WorkerError, fail};
use std::time::Duration;
use symbi_fabric::{Fabric, Link};

/// one worker's device binding as it enters the session
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceBinding {
    /// a host build: the worker computes on no device
    Host,
    /// the physical gpu: the host's name and the gpu's pci bus address
    Gpu { host: String, pci: String },
    /// the gpu runtime refused the binding; the reason is in the worker's own record
    Refused,
}

const WORDS: u64 = 3;
const HOST: u64 = 0;
const GPU: u64 = 1;
const REFUSED: u64 = 2;

/// 64-bit FNV-1a: equal text gives equal words in every process, so a collision can only
/// merge two distinct identities into one, which refuses a valid session and never admits
/// a shared gpu
fn fnv1a(text: &str) -> u64 {
    text.bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
}

/// the three words a worker contributes: the binding kind, the host hash and the pci hash
pub fn binding_words(binding: &DeviceBinding) -> [u64; 3] {
    match binding {
        DeviceBinding::Host => [HOST, 0, 0],
        DeviceBinding::Gpu { host, pci } => [GPU, fnv1a(host), fnv1a(pci)],
        DeviceBinding::Refused => [REFUSED, 0, 0],
    }
}

/// why a session's assembled bindings end it
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingRefusal {
    /// these workers' gpu runtimes refused their bindings
    Refused(Vec<usize>),
    /// the first two workers, in worker order, bound one gpu of one host
    Shared(usize, usize),
}

/// the session's decision over every worker's three words, in worker order
pub fn session_refusal(words: &[u64]) -> Option<BindingRefusal> {
    let bindings: Vec<&[u64]> = words.chunks_exact(WORDS as usize).collect();
    let refused: Vec<usize> = (0..bindings.len()).filter(|&ww| bindings[ww][0] == REFUSED).collect();
    if !refused.is_empty() {
        return Some(BindingRefusal::Refused(refused));
    }
    for ii in 0..bindings.len() {
        for jj in ii + 1..bindings.len() {
            if bindings[ii][0] == GPU && bindings[ii] == bindings[jj] {
                return Some(BindingRefusal::Shared(ii, jj));
            }
        }
    }
    None
}

/// every worker contributes its binding and every worker receives all of them: a refusal is
/// decided identically on every worker at this collective and returned after the peers are
/// told, so the session ends on all of them within the collective's deadline
pub fn agree_device_bindings<L: Link>(
    fabric: &mut Fabric<L>,
    me: usize,
    workers: usize,
    local: &DeviceBinding,
    deadline: Duration,
) -> Result<(), WorkerError> {
    let base = me as u64 * WORDS;
    let entries: Vec<(u64, u64)> = binding_words(local).iter().enumerate().map(|(kk, &w)| (base + kk as u64, w)).collect();
    let words = fabric.assemble_windowed(workers as u64 * WORDS, &entries, deadline).map_err(|e| fail(fabric, e.into()))?;
    let detail = match session_refusal(&words) {
        None => return Ok(()),
        Some(BindingRefusal::Refused(workers)) => format!(
            "the gpu binding of worker(s) {workers:?} was refused by its runtime; each refusal is in that worker's record"
        ),
        Some(BindingRefusal::Shared(ii, jj)) => {
            let named = match local {
                DeviceBinding::Gpu { host, pci } if me == ii || me == jj => format!(" (host {host}, pci {pci})"),
                _ => String::new(),
            };
            format!(
                "workers {ii} and {jj} bound one physical gpu{named}; each launched worker computes on a gpu of its \
                 own. under a scheduler bind one gpu per task (on Frontier: srun -c7 --gpus-per-task=1 \
                 --gpu-bind=closest)"
            )
        }
    };
    Err(fail(fabric, WorkerError::Refused(detail)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(host: &str, pci: &str) -> DeviceBinding {
        DeviceBinding::Gpu { host: host.into(), pci: pci.into() }
    }

    fn decide(bindings: &[DeviceBinding]) -> Option<BindingRefusal> {
        session_refusal(&bindings.iter().flat_map(binding_words).collect::<Vec<_>>())
    }

    #[test]
    fn distinct_gpus_of_one_host_are_admitted() {
        assert_eq!(decide(&[gpu("n1", "0000:c1:00.0"), gpu("n1", "0000:c6:00.0")]), None);
    }

    #[test]
    fn one_pci_address_on_two_hosts_is_two_gpus() {
        assert_eq!(decide(&[gpu("n1", "0000:c1:00.0"), gpu("n2", "0000:c1:00.0")]), None);
    }

    #[test]
    fn one_gpu_of_one_host_bound_twice_is_refused() {
        let bindings = [gpu("n1", "0000:c1:00.0"), gpu("n2", "0000:c1:00.0"), gpu("n1", "0000:c1:00.0")];
        assert_eq!(decide(&bindings), Some(BindingRefusal::Shared(0, 2)));
    }

    #[test]
    fn a_refused_binding_ends_the_session_before_any_comparison() {
        let bindings = [gpu("n1", "a"), DeviceBinding::Refused, gpu("n1", "a"), DeviceBinding::Refused];
        assert_eq!(decide(&bindings), Some(BindingRefusal::Refused(vec![1, 3])));
    }

    #[test]
    fn host_workers_share_no_device() {
        assert_eq!(decide(&[DeviceBinding::Host, DeviceBinding::Host]), None);
    }

    #[test]
    fn the_hash_separates_the_frontier_gcd_addresses() {
        let addresses = ["0000:c1:00.0", "0000:c6:00.0", "0000:c9:00.0", "0000:ce:00.0", "0000:d1:00.0", "0000:d6:00.0", "0000:d9:00.0", "0000:de:00.0"];
        let words: std::collections::BTreeSet<u64> = addresses.iter().map(|a| fnv1a(a)).collect();
        assert_eq!(words.len(), addresses.len());
    }
}
