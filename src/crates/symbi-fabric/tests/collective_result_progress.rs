// =============================================================================
// collective_result_progress.rs
//
// the coordinator's collective result reaches a peer over tcp while the
// coordinator makes no further fabric call: once the last contribution is tabled
// the results are written to the sockets before the coordinator's collective
// returns. two fabrics in one process share a connected socket pair; the peer
// contributes and drives once, the coordinator completes its collective, and the
// peer alone then waits for the result.
//
// usage:
//   cargo test -p symbi-fabric --test collective_result_progress
// =============================================================================

use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};
use symbi_fabric::tcp::TcpLink;
use symbi_fabric::{Fabric, OpKind, SessionId, WorkerId};

const COORDINATOR: WorkerId = WorkerId(0);
const PEER: WorkerId = WorkerId(1);

fn connected() -> (Fabric<TcpLink>, Fabric<TcpLink>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let dialed = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (accepted, _) = listener.accept().unwrap();
    let coordinator_link = TcpLink::new(COORDINATOR, 2, 1 << 16);
    coordinator_link.attach(PEER, accepted).unwrap();
    let peer_link = TcpLink::new(PEER, 2, 1 << 16);
    peer_link.attach(COORDINATOR, dialed).unwrap();
    let fabric = |link, me| Fabric::new(link, me, SessionId(7), 2, 1, &[], vec![vec![0], vec![0]]);
    (fabric(coordinator_link, COORDINATOR), fabric(peer_link, PEER))
}

#[test]
fn a_peer_receives_the_result_while_the_coordinator_makes_no_further_fabric_call() {
    let (mut coordinator, mut peer) = connected();
    peer.begin_collective(OpKind::Min, 2.0f64.to_bits()).unwrap();
    assert_eq!(peer.collective_progress().unwrap(), None, "the result exists before the coordinator contributed");
    let reduced = coordinator.collective(OpKind::Min, 3.0f64.to_bits(), Duration::from_secs(5)).unwrap();
    assert_eq!(f64::from_bits(reduced), 2.0, "the coordinator reduced another minimum");
    // the coordinator is idle from here on; only the peer services its own end
    let start = Instant::now();
    let received = loop {
        if let Some(bits) = peer.collective_progress().unwrap() {
            break bits;
        }
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "the result never reached the peer while the coordinator made no fabric call: it was queued, not written"
        );
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(f64::from_bits(received), 2.0);
    drop(coordinator);
}
