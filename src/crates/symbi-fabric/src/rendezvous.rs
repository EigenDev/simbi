// =============================================================================
// rendezvous.rs
//
// the session handshake. every worker binds a listener, connects to the
// coordinator, and sends a Hello carrying its identity: protocol version,
// credential, build and configuration digests, plan digest, placement digest,
// halo width, worker count, block credit, listen port, byte order. the
// coordinator admits the session only when every Hello equals its own, then
// sends each worker a Welcome with the session id and the listen table; the
// workers complete a full mesh and the streams become the TCP link. no
// simulation frame travels before the Welcome.
//
// a mismatch is refused by category: build or configuration, numerical plan,
// or placement (a launcher error). every blocking step honors the startup
// deadline.
//
// usage:
//  let (link, session) = rendezvous::connect(&Rendezvous { .. }, max_payload)?;
// =============================================================================

use crate::error::{Deadline, FabricError};
use crate::frame::{HEADER_LEN, Header, Kind};
use crate::ident::{Digest, Epoch, SessionId, WorkerId};
use crate::tcp::TcpLink;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

pub const PROTOCOL: u16 = 1;
const LITTLE_ENDIAN: u8 = 1;

/// what a worker must agree on with every other worker before the session commits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub credential: u64,
    pub build_id: String,
    pub config_digest: Digest,
    pub plan_digest: Digest,
    pub placement_digest: Digest,
    pub ng: u32,
    pub workers: u32,
    pub block_credit: u32,
}

#[derive(Debug, Clone)]
pub struct Rendezvous {
    pub me: WorkerId,
    /// where the coordinator listens. with `announce` set the coordinator binds this
    /// address with port zero and publishes the bound address; the workers read it.
    pub coordinator: SocketAddr,
    /// the rendezvous file: written by the coordinator once bound, polled by the workers.
    pub announce: Option<std::path::PathBuf>,
    pub identity: Identity,
    pub startup: Duration,
}

/// the coordinator's bound address, from the rendezvous file once it appears.
fn announced(
    path: &std::path::Path,
    start: Instant,
    budget: Duration,
) -> Result<SocketAddr, FabricError> {
    loop {
        remaining(start, budget)?;
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(addr) = text.trim().parse::<SocketAddr>() {
                return Ok(addr);
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn none_epoch() -> Epoch {
    Epoch {
        step: 0,
        attempt: 0,
        point: Epoch::NO_POINT,
        axis: Epoch::NO_AXIS,
    }
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

struct Cursor<'a>(&'a [u8], usize);

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], FabricError> {
        if self.1 + n > self.0.len() {
            return Err(FabricError::Rendezvous {
                detail: "truncated handshake payload".into(),
            });
        }
        let s = &self.0[self.1..self.1 + n];
        self.1 += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, FabricError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, FabricError> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }
    fn u32(&mut self) -> Result<u32, FabricError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64, FabricError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
}

/// the Hello payload.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Hello {
    protocol: u16,
    identity: Identity,
    port: u16,
    byte_order: u8,
}

impl Hello {
    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_u16(&mut out, self.protocol);
        put_u64(&mut out, self.identity.credential);
        put_u32(&mut out, self.identity.build_id.len() as u32);
        out.extend_from_slice(self.identity.build_id.as_bytes());
        put_u64(&mut out, self.identity.config_digest.0);
        put_u64(&mut out, self.identity.plan_digest.0);
        put_u64(&mut out, self.identity.placement_digest.0);
        put_u32(&mut out, self.identity.ng);
        put_u32(&mut out, self.identity.workers);
        put_u32(&mut out, self.identity.block_credit);
        put_u16(&mut out, self.port);
        out.push(self.byte_order);
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self, FabricError> {
        let mut c = Cursor(bytes, 0);
        let protocol = c.u16()?;
        let credential = c.u64()?;
        let n = c.u32()? as usize;
        if n > 4096 {
            return Err(FabricError::Rendezvous {
                detail: "build id longer than 4096 bytes".into(),
            });
        }
        let build_id = String::from_utf8_lossy(c.take(n)?).into_owned();
        let config_digest = Digest(c.u64()?);
        let plan_digest = Digest(c.u64()?);
        let placement_digest = Digest(c.u64()?);
        let ng = c.u32()?;
        let workers = c.u32()?;
        let block_credit = c.u32()?;
        let port = c.u16()?;
        let byte_order = c.u8()?;
        Ok(Self {
            protocol,
            identity: Identity {
                credential,
                build_id,
                config_digest,
                plan_digest,
                placement_digest,
                ng,
                workers,
                block_credit,
            },
            port,
            byte_order,
        })
    }
}

/// the category a disagreement is reported under.
fn disagreement(mine: &Hello, theirs: &Hello, worker: WorkerId) -> Option<String> {
    let m = &mine.identity;
    let t = &theirs.identity;
    if theirs.protocol != mine.protocol {
        return Some(format!(
            "{worker:?} speaks protocol {} against {}",
            theirs.protocol, mine.protocol
        ));
    }
    if theirs.byte_order != LITTLE_ENDIAN {
        return Some(format!("{worker:?} is not little-endian"));
    }
    if t.credential != m.credential {
        return Some(format!(
            "{worker:?} presented a different session credential"
        ));
    }
    if t.build_id != m.build_id
        || t.config_digest != m.config_digest
        || t.ng != m.ng
        || t.workers != m.workers
        || t.block_credit != m.block_credit
    {
        return Some(format!(
            "build or configuration disagreement with {worker:?}"
        ));
    }
    if t.plan_digest != m.plan_digest {
        return Some(format!("numerical plan disagreement with {worker:?}"));
    }
    if t.placement_digest != m.placement_digest {
        return Some(format!(
            "placement disagreement with {worker:?} (launcher error)"
        ));
    }
    None
}

fn remaining(start: Instant, budget: Duration) -> Result<Duration, FabricError> {
    let elapsed = start.elapsed();
    if elapsed >= budget {
        return Err(FabricError::Deadline {
            phase: Deadline::Startup,
            peer: None,
            epoch: None,
            pending: 0,
        });
    }
    Ok(budget - elapsed)
}

/// a short socket timeout: every partial read or write returns to the caller to recheck the
/// absolute deadline, so trickled progress cannot extend the handshake past it.
const SLICE: Duration = Duration::from_millis(50);

fn write_frame(
    stream: &mut TcpStream,
    header: Header,
    payload: &[u8],
    start: Instant,
    budget: Duration,
) -> Result<(), FabricError> {
    let mut head = [0u8; HEADER_LEN];
    header.encode(&mut head);
    for bytes in [&head[..], payload] {
        let mut sent = 0;
        while sent < bytes.len() {
            stream.set_write_timeout(Some(remaining(start, budget)?.min(SLICE)))?;
            match stream.write(&bytes[sent..]) {
                Ok(0) => {
                    return Err(FabricError::Rendezvous {
                        detail: "peer closed during the handshake".into(),
                    });
                }
                Ok(n) => sent += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) => {}
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(())
}

fn read_exact_by(
    stream: &mut TcpStream,
    buf: &mut [u8],
    start: Instant,
    budget: Duration,
) -> Result<(), FabricError> {
    let mut have = 0;
    while have < buf.len() {
        stream.set_read_timeout(Some(remaining(start, budget)?.min(SLICE)))?;
        match stream.read(&mut buf[have..]) {
            Ok(0) => {
                return Err(FabricError::Rendezvous {
                    detail: "peer closed during the handshake".into(),
                });
            }
            Ok(n) => have += n,
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn read_frame(
    stream: &mut TcpStream,
    start: Instant,
    budget: Duration,
    max_payload: usize,
) -> Result<(Header, Vec<u8>), FabricError> {
    let mut head = [0u8; HEADER_LEN];
    read_exact_by(stream, &mut head, start, budget)?;
    let header = Header::decode(&head).map_err(|e| FabricError::Rendezvous {
        detail: format!("handshake frame: {e}"),
    })?;
    if header.payload_len as usize > max_payload {
        return Err(FabricError::Rendezvous {
            detail: format!("handshake payload of {} bytes", header.payload_len),
        });
    }
    let mut payload = vec![0u8; header.payload_len as usize];
    read_exact_by(stream, &mut payload, start, budget)?;
    Ok((header, payload))
}

fn abort_frame(detail: &str) -> Header {
    Header {
        kind: Kind::Abort,
        session: SessionId(0),
        epoch: none_epoch(),
        id: 0,
        payload_len: detail.len() as u32,
    }
}

fn fresh_session() -> SessionId {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    SessionId(nanos ^ (u64::from(std::process::id()) << 40) ^ 0x9e37_79b9_7f4a_7c15)
}

const HANDSHAKE_PAYLOAD: usize = 8192;

/// establish the session and return the connected link. the coordinator is worker 0 and
/// listens at `r.coordinator`; every other worker connects there. streams are blocking
/// with the startup deadline through the handshake and nonblocking once attached.
pub fn connect(r: &Rendezvous, max_payload: usize) -> Result<(TcpLink, SessionId), FabricError> {
    let start = Instant::now();
    let budget = r.startup;
    let workers = r.identity.workers as usize;
    let listener = if r.me == WorkerId(0) {
        TcpListener::bind(r.coordinator)?
    } else {
        TcpListener::bind((r.coordinator.ip(), 0))?
    };
    let port = listener.local_addr()?.port();
    let coordinator = match (&r.announce, r.me) {
        (Some(path), WorkerId(0)) => {
            let bound = listener.local_addr()?;
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, bound.to_string())?;
            std::fs::rename(&tmp, path)?;
            bound
        }
        (Some(path), _) => announced(path, start, budget)?,
        (None, _) => r.coordinator,
    };
    let mine = Hello {
        protocol: PROTOCOL,
        identity: r.identity.clone(),
        port,
        byte_order: LITTLE_ENDIAN,
    };
    let link = TcpLink::new(r.me, workers, max_payload);
    let mut table: Vec<u16> = vec![0; workers];
    let session;

    if r.me == WorkerId(0) {
        listener.set_nonblocking(true)?;
        let mut streams: Vec<Option<TcpStream>> = (0..workers).map(|_| None).collect();
        let mut hellos: Vec<Option<Hello>> = (0..workers).map(|_| None).collect();
        table[0] = port;
        let mut admitted = 1;
        while admitted < workers {
            remaining(start, budget)?;
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false)?;
                    let (header, payload) =
                        read_frame(&mut stream, start, budget, HANDSHAKE_PAYLOAD)?;
                    if header.kind != Kind::Hello {
                        return Err(FabricError::Rendezvous {
                            detail: format!("expected Hello, got {:?}", header.kind),
                        });
                    }
                    let worker = WorkerId(header.id);
                    if worker.0 as usize >= workers
                        || worker == WorkerId(0)
                        || streams[worker.0 as usize].is_some()
                    {
                        return Err(FabricError::Rendezvous {
                            detail: format!("unexpected or duplicate worker id {}", header.id),
                        });
                    }
                    let hello = Hello::decode(&payload)?;
                    if let Some(detail) = disagreement(&mine, &hello, worker) {
                        let _ = write_frame(
                            &mut stream,
                            abort_frame(&detail),
                            detail.as_bytes(),
                            start,
                            budget,
                        );
                        for s in streams.iter_mut().flatten() {
                            let _ = write_frame(
                                s,
                                abort_frame(&detail),
                                detail.as_bytes(),
                                start,
                                budget,
                            );
                        }
                        return Err(FabricError::Rendezvous { detail });
                    }
                    table[worker.0 as usize] = hello.port;
                    hellos[worker.0 as usize] = Some(hello);
                    streams[worker.0 as usize] = Some(stream);
                    admitted += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => return Err(e.into()),
            }
        }
        session = fresh_session();
        let mut welcome = Vec::new();
        put_u64(&mut welcome, session.0);
        for &p in &table {
            put_u16(&mut welcome, p);
        }
        let header = Header {
            kind: Kind::Welcome,
            session,
            epoch: none_epoch(),
            id: 0,
            payload_len: welcome.len() as u32,
        };
        for (idx, stream) in streams.iter_mut().enumerate() {
            if let Some(stream) = stream {
                write_frame(stream, header, &welcome, start, budget)?;
                link.attach(WorkerId(idx as u32), stream.try_clone()?)?;
            }
        }
    } else {
        let mut stream = loop {
            let left = remaining(start, budget)?;
            match TcpStream::connect_timeout(&coordinator, left.min(Duration::from_millis(200))) {
                Ok(s) => break s,
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        };
        let payload = mine.encode();
        let header = Header {
            kind: Kind::Hello,
            session: SessionId(0),
            epoch: none_epoch(),
            id: r.me.0,
            payload_len: payload.len() as u32,
        };
        write_frame(&mut stream, header, &payload, start, budget)?;
        let (header, payload) = read_frame(&mut stream, start, budget, HANDSHAKE_PAYLOAD)?;
        match header.kind {
            Kind::Welcome => {}
            Kind::Abort => {
                return Err(FabricError::Rendezvous {
                    detail: String::from_utf8_lossy(&payload).into_owned(),
                });
            }
            other => {
                return Err(FabricError::Rendezvous {
                    detail: format!("expected Welcome, got {other:?}"),
                });
            }
        }
        let mut c = Cursor(&payload, 0);
        session = SessionId(c.u64()?);
        for slot in table.iter_mut() {
            *slot = c.u16()?;
        }
        link.attach(WorkerId(0), stream)?;
        // the mesh among workers 1..n: connect to every lower id, accept every higher id.
        let me = r.me.0 as usize;
        for j in 1..me {
            let addr = SocketAddr::new(coordinator.ip(), table[j]);
            let mut s = loop {
                let left = remaining(start, budget)?;
                match TcpStream::connect_timeout(&addr, left.min(Duration::from_millis(200))) {
                    Ok(s) => break s,
                    Err(_) => std::thread::sleep(Duration::from_millis(5)),
                }
            };
            let header = Header {
                kind: Kind::Hello,
                session,
                epoch: none_epoch(),
                id: r.me.0,
                payload_len: 0,
            };
            write_frame(&mut s, header, &[], start, budget)?;
            link.attach(WorkerId(j as u32), s)?;
        }
        listener.set_nonblocking(true)?;
        let mut pending = workers - me - 1;
        while pending > 0 {
            remaining(start, budget)?;
            match listener.accept() {
                Ok((mut s, _)) => {
                    s.set_nonblocking(false)?;
                    let (header, _) = read_frame(&mut s, start, budget, HANDSHAKE_PAYLOAD)?;
                    if header.kind != Kind::Hello || header.session != session {
                        return Err(FabricError::Rendezvous {
                            detail: "unexpected peer preamble".into(),
                        });
                    }
                    let j = header.id as usize;
                    if j <= me || j >= workers {
                        return Err(FabricError::Rendezvous {
                            detail: format!("peer {j} connected out of order"),
                        });
                    }
                    link.attach(WorkerId(j as u32), s)?;
                    pending -= 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
    if link.attached() != workers - 1 {
        return Err(FabricError::Rendezvous {
            detail: format!("{} of {} peers attached", link.attached(), workers - 1),
        });
    }
    Ok((link, session))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> Identity {
        Identity {
            credential: 7,
            build_id: "test".into(),
            config_digest: Digest(1),
            plan_digest: Digest(2),
            placement_digest: Digest(3),
            ng: 2,
            workers: 2,
            block_credit: 4096,
        }
    }

    #[test]
    fn hello_round_trips() {
        let h = Hello {
            protocol: PROTOCOL,
            identity: identity(),
            port: 4242,
            byte_order: LITTLE_ENDIAN,
        };
        assert_eq!(Hello::decode(&h.encode()).unwrap(), h);
    }

    #[test]
    fn disagreements_are_categorized() {
        let mine = Hello {
            protocol: PROTOCOL,
            identity: identity(),
            port: 1,
            byte_order: LITTLE_ENDIAN,
        };
        let mut plan = mine.clone();
        plan.identity.plan_digest = Digest(9);
        assert!(
            disagreement(&mine, &plan, WorkerId(1))
                .unwrap()
                .contains("numerical plan")
        );
        let mut place = mine.clone();
        place.identity.placement_digest = Digest(9);
        assert!(
            disagreement(&mine, &place, WorkerId(1))
                .unwrap()
                .contains("launcher")
        );
        let mut build = mine.clone();
        build.identity.build_id = "other".into();
        assert!(
            disagreement(&mine, &build, WorkerId(1))
                .unwrap()
                .contains("build or configuration")
        );
        assert!(disagreement(&mine, &mine, WorkerId(1)).is_none());
    }
}

#[cfg(test)]
mod trickle_tests {
    use super::*;

    /// a client that writes a valid Hello one byte every 100 ms holds the coordinator's
    /// read open far past a one-second startup deadline; the coordinator must give up at
    /// the deadline, not when the trickle ends.
    #[test]
    fn a_trickled_hello_cannot_extend_the_startup_deadline() {
        let file = std::env::temp_dir().join(format!("symbi_rv_trickle_{}", std::process::id()));
        let _ = std::fs::remove_file(&file);
        let r = Rendezvous {
            me: WorkerId(0),
            coordinator: "127.0.0.1:0".parse().unwrap(),
            announce: Some(file.clone()),
            identity: Identity {
                credential: 1,
                build_id: "trickle".into(),
                config_digest: Digest(1),
                plan_digest: Digest(2),
                placement_digest: Digest(3),
                ng: 2,
                workers: 2,
                block_credit: 1,
            },
            startup: Duration::from_secs(1),
        };
        let coordinator = std::thread::spawn(move || {
            let started = Instant::now();
            let result = connect(&r, 1024);
            (result.err(), started.elapsed())
        });
        let addr = announced(&file, Instant::now(), Duration::from_secs(5)).unwrap();
        let mut stream = TcpStream::connect(addr).unwrap();
        let hello = Hello {
            protocol: PROTOCOL,
            identity: Identity {
                credential: 1,
                build_id: "trickle".into(),
                config_digest: Digest(1),
                plan_digest: Digest(2),
                placement_digest: Digest(3),
                ng: 2,
                workers: 2,
                block_credit: 1,
            },
            port: 1,
            byte_order: LITTLE_ENDIAN,
        };
        let payload = hello.encode();
        let header = Header {
            kind: Kind::Hello,
            session: SessionId(0),
            epoch: none_epoch(),
            id: 1,
            payload_len: payload.len() as u32,
        };
        let mut head = [0u8; HEADER_LEN];
        header.encode(&mut head);
        let mut bytes = head.to_vec();
        bytes.extend(payload);
        // trickle until the coordinator gives up; a broken pipe ends the trickle
        for b in bytes {
            if stream.write_all(&[b]).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
            if coordinator.is_finished() {
                break;
            }
        }
        let (err, elapsed) = coordinator.join().unwrap();
        let _ = std::fs::remove_file(&file);
        assert!(
            matches!(
                err,
                Some(FabricError::Deadline {
                    phase: Deadline::Startup,
                    ..
                })
            ),
            "{err:?}"
        );
        assert!(
            elapsed < Duration::from_millis(2500),
            "gave up after {elapsed:?}"
        );
    }
}
