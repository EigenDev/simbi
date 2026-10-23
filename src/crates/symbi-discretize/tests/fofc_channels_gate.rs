// =============================================================================
// fofc_channels_gate.rs
//
// structural gate for the fallback ladder's channel law: classification may
// trigger an action, but only the component performing the action reports it.
// the scan covers the production sources of the discretize, substrate, and sim
// crates plus the bake script, and pins:
// - the recomputing probe family stays deleted (`fofc_probe` / `fofc_freeze`
//   have no lowercase spelling; the `FOFC_FREEZE_*` census counters and the
//   `freeze_streak`/`FREEZE_HALT` names are their own words);
// - each channel buffer's field access appears only inside its sanctioned
//   producer/reader functions (brace-tracked extents, so an unrelated
//   function cannot borrow a sanctioned spelling; a spelling matches as a
//   whole identifier, so `.fofc_flags_in(` is a call, distinct from the
//   `.fofc_flag` field):
//     C2pStatus (`.c2p_error`)      — the regime `c2p` producers, the status
//                                     decodes (plain and excision-masked), and
//                                     the diagnostic scanners;
//     TroubledCell (`.fofc_flag`)   — the orchestrator, the GRMHD ladder and the
//                                     mark, which bind it and pass the local; the
//                                     troubled count; the cut exchange that
//                                     carries the flags between tiles; and the
//                                     one injector of a physical troubled cell;
//     FreezeApplied (`.freeze_applied`) — the orchestrator, which hands it to
//                                     the correcting selects;
// - the three channels are three distinct field allocations.
// two narrow access categories, each a list of (file, function) pairs and a law the gate checks
// on every line of the pair's body:
//   checkpoint inspection — the checkpoint writer's read of the recovery status to decide which
//                           covered cells it publishes without primitives; reads only (`.view()`);
//   byte-preserving transfer — the device-local mirror and state transfers, which copy whole
//                           channel buffers between memory spaces; no element access at all.
// one pending finding stays reported by name until its contract is decided: the dual-energy
// synchronization clears the recovery status of a cell it recovers from the entropy, a second
// producer of `C2pStatus`.
// =============================================================================

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("scanned source dir must exist") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// the file's production lines: every line outside `#[cfg(test)]` items,
/// brace-skipped like the CT vocabulary gate.
fn production_lines(text: &str) -> Vec<(usize, &str)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = Vec::new();
    let mut ii = 0;
    while ii < lines.len() {
        if lines[ii].trim_start().starts_with("#[cfg(test)]") {
            let mut depth: i64 = 0;
            let mut entered = false;
            while ii < lines.len() {
                let line = lines[ii];
                depth += line.matches('{').count() as i64;
                depth -= line.matches('}').count() as i64;
                entered |= line.contains('{');
                let terse_end = !entered && line.trim_end().ends_with(';');
                ii += 1;
                if (entered && depth <= 0) || terse_end {
                    break;
                }
            }
        } else {
            kept.push((ii + 1, lines[ii]));
            ii += 1;
        }
    }
    kept
}

/// a channel's boundary: the field-access spelling and the functions allowed
/// to spell it.
struct ChannelRule {
    spelling: &'static str,
    allowed_fns: &'static [&'static str],
}

/// the channel table — the one statement of who may touch which buffer.
const CHANNELS: &[ChannelRule] = &[
    ChannelRule {
        // C2pStatus: produced by the regime c2p dispatches, decoded by
        // `troubled_from_status`, read by the diagnostic scanners.
        spelling: ".c2p_error",
        allowed_fns: &[
            "c2p",
            "troubled_from_status",
            "dispatch_fofc_flag_unexcised",
            "scan_c2p_errors",
            "first_c2p_error",
            "first_c2p_failure_state",
        ],
    },
    ChannelRule {
        // TroubledCell: bound by the orchestrator, the GRMHD ladder body and the
        // mark, then passed as a local to the ghost fill, splices, and counts;
        // reduced by the troubled count; moved across cuts by the exchange plan
        // and the in-process trouble exchange; raised at one physical cell by
        // the single troubled-cell injector.
        spelling: ".fofc_flag",
        allowed_fns: &[
            "fofc_orchestrate",
            "fofc_impl",
            "fofc_mark",
            "fofc_flags_in",
            "plan_fields",
            "exchange_trouble",
            "raise_troubled_cell",
        ],
    },
    ChannelRule {
        // FreezeApplied: bound once by the orchestrator, which hands it to the
        // correcting selects — the only writers — and the count reduction.
        spelling: ".freeze_applied",
        allowed_fns: &["fofc_orchestrate"],
    },
];

/// a channel access granted to one function of one file, under a law its lines must obey
struct ScopedAccess {
    category: &'static str,
    file: &'static str,
    function: &'static str,
    law: fn(&str) -> bool,
}

/// every channel spelling on the line is followed by a read-only view
fn reads_only(code: &str) -> bool {
    CHANNELS.iter().all(|rule| {
        code.match_indices(rule.spelling)
            .all(|(at, _)| code[at + rule.spelling.len()..].starts_with(".view()"))
    })
}

/// the line moves or measures whole buffers: no element read or write on any field
fn whole_buffers_only(code: &str) -> bool {
    [".set(", ".view(", ".view_mut(", ".at(", ".add_assign_checked("].iter().all(|op| !code.contains(op))
}

/// the scoped grants, by function within one file
const SCOPED: &[ScopedAccess] = &[
    ScopedAccess { category: "checkpoint inspection", file: "symbi-sim/src/state.rs", function: "withhold_unrecovered_primitives", law: reads_only },
    ScopedAccess { category: "checkpoint inspection", file: "symbi-sim/src/state.rs", function: "has_unrecovered_excluded", law: reads_only },
    ScopedAccess { category: "byte-preserving transfer", file: "symbi-sim/src/device_local.rs", function: "mirror_store", law: whole_buffers_only },
    ScopedAccess { category: "byte-preserving transfer", file: "symbi-sim/src/device_local.rs", function: "state_pairs", law: whole_buffers_only },
    ScopedAccess { category: "byte-preserving transfer", file: "symbi-sim/src/device_local.rs", function: "resident_bytes", law: whole_buffers_only },
];

/// an access outside the channel contract whose decision is pending: one statement of one
/// function, matched by its code, reported by name. any other access in the same function is
/// a violation
struct Pending {
    file: &'static str,
    function: &'static str,
    code: &'static str,
    finding: &'static str,
}

const PENDING: &[Pending] = &[Pending {
    file: "symbi-substrate/src/regimes/dual_energy.rs",
    function: "synchronize",
    code: "sim.fields.c2p_error.set(c, Sc::from_f64(0.0));",
    finding: "the entropy recovery clears the c2p status, a second producer of C2pStatus; pending the \
              recovery-contract decision for the opt-in dual-energy prototype",
}];

/// whether the access `code` in `function` of `file` is the pending one
fn is_pending(file: &str, function: &str, code: &str) -> bool {
    PENDING.iter().any(|p| file.ends_with(p.file) && p.function == function && p.code == code)
}

/// true when `code` spells `spelling` as a whole identifier: the character after it
/// continues no identifier.
fn spells(code: &str, spelling: &str) -> bool {
    code.match_indices(spelling).any(|(at, _)| {
        code[at + spelling.len()..]
            .chars()
            .next()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
    })
}

/// channel-boundary violations in one source text: comment text is stripped,
/// function extents are brace-tracked (a `fn` line opens a body, brace balance
/// closes it), and a line spelling a channel's field access outside that
/// channel's sanctioned functions is flagged.
fn channel_violations(text: &str) -> Vec<(usize, String, String)> {
    let mut fn_name = String::new();
    let mut in_fn = false;
    let mut entered = false;
    let mut depth: i64 = 0;
    let mut hits = Vec::new();
    for (number, raw) in production_lines(text) {
        let code = raw.split("//").next().unwrap_or("");
        if !in_fn {
            if let Some(pos) = code.find("fn ") {
                fn_name = code[pos + 3..]
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("")
                    .to_string();
                in_fn = true;
                entered = false;
                depth = 0;
            }
        }
        if in_fn {
            depth += code.matches('{').count() as i64;
            depth -= code.matches('}').count() as i64;
            entered |= code.contains('{');
        }
        for rule in CHANNELS {
            if spells(code, rule.spelling)
                && !(in_fn && rule.allowed_fns.contains(&fn_name.as_str()))
            {
                hits.push((number, if in_fn { fn_name.clone() } else { String::new() }, code.trim().to_string()));
                break;
            }
        }
        if in_fn && entered && depth <= 0 {
            in_fn = false;
            fn_name.clear();
        }
    }
    hits
}

fn scanned_files() -> Vec<PathBuf> {
    let here = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources(&here.join("src"), &mut files);
    for sibling in ["symbi-substrate", "symbi-sim"] {
        rust_sources(
            &here.parent().unwrap().join(sibling).join("src"),
            &mut files,
        );
    }
    files.push(here.parent().unwrap().join("symbi-aot").join("build.rs"));
    files
}

#[test]
fn the_recomputing_probe_family_stays_deleted() {
    let files = scanned_files();
    assert!(
        files.len() >= 60,
        "the scan found only {} files; the gate is not seeing the crates",
        files.len()
    );
    let mut hits = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("source file must be readable");
        for (number, line) in production_lines(&text) {
            let code = line.split("//").next().unwrap_or("");
            if code.contains("fofc_probe") || code.contains("fofc_freeze") {
                hits.push(format!("{}:{number}: {}", path.display(), line.trim()));
            }
        }
    }
    assert!(
        hits.is_empty(),
        "a recomputing fallback classifier returned (the TroubledCell decode and \
         the FreezeApplied act write are the channels' only producers):\n{}",
        hits.join("\n")
    );
}

#[test]
fn each_channel_stays_inside_its_boundary() {
    let files = scanned_files();
    let mut seen = 0usize;
    let mut hits = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path).expect("source file must be readable");
        for (_, line) in production_lines(&text) {
            let code = line.split("//").next().unwrap_or("");
            if CHANNELS.iter().any(|r| spells(code, r.spelling)) {
                seen += 1;
            }
        }
        let file = path.display().to_string();
        for (number, function, code) in channel_violations(&text) {
            let scoped = SCOPED.iter().find(|grant| file.ends_with(grant.file) && grant.function == function);
            let pending = is_pending(&file, &function, &code);
            match scoped {
                Some(grant) if (grant.law)(&code) => {}
                Some(grant) => hits.push(format!("{file}:{number}: {code}  (breaks the {} law of {})", grant.category, grant.function)),
                None if pending => {}
                None => hits.push(format!("{file}:{number}: {code}")),
            }
        }
    }
    assert!(
        seen >= 10,
        "the scan saw only {seen} channel accesses; the gate is not seeing the producers"
    );
    assert!(
        hits.is_empty(),
        "a channel buffer crossed outside its sanctioned functions:\n{}",
        hits.join("\n")
    );
}

/// every out-of-contract access the scan finds, as (file, function, code)
fn out_of_contract() -> Vec<(String, String, String)> {
    let mut found = Vec::new();
    for path in scanned_files() {
        let text = std::fs::read_to_string(&path).expect("source file must be readable");
        for (_, function, code) in channel_violations(&text) {
            found.push((path.display().to_string(), function, code));
        }
    }
    found
}

#[test]
fn every_scoped_grant_names_a_live_access() {
    let found = out_of_contract();
    for grant in SCOPED {
        assert!(
            found.iter().any(|(file, function, _)| file.ends_with(grant.file) && *function == grant.function),
            "the {} grant to {}::{} matches no channel access; remove it",
            grant.category,
            grant.file,
            grant.function
        );
    }
}

#[test]
fn the_pending_channel_findings_remain_reported() {
    let found = out_of_contract();
    for p in PENDING {
        let matching = found.iter().filter(|(f, func, code)| is_pending(f, func, code)).count();
        assert_eq!(
            matching, 1,
            "{}::{} holds {matching} copies of its pending access `{}`; resolve or restate the entry ({})",
            p.file, p.function, p.code, p.finding
        );
        eprintln!("pending channel finding: {}::{}: `{}`: {}", p.file, p.function, p.code, p.finding);
    }
}

#[test]
fn the_pending_exception_admits_its_one_statement_alone() {
    let file = "crates/symbi-substrate/src/regimes/dual_energy.rs";
    assert!(is_pending(file, "synchronize", "sim.fields.c2p_error.set(c, Sc::from_f64(0.0));"));
    // a second access in the same function is a violation, as is the same statement elsewhere
    assert!(!is_pending(file, "synchronize", "sim.fields.c2p_error.set(c, Sc::from_f64(64.0));"));
    assert!(!is_pending(file, "synchronize", "let flag = &sim.workspace.fofc_flag;"));
    assert!(!is_pending(file, "drain_entropy", "sim.fields.c2p_error.set(c, Sc::from_f64(0.0));"));
    assert!(!is_pending("crates/symbi-substrate/src/regimes/fofc.rs", "synchronize", "sim.fields.c2p_error.set(c, Sc::from_f64(0.0));"));
}

#[test]
fn the_three_channels_are_three_distinct_allocations() {
    let state = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("symbi-sim")
        .join("src")
        .join("state.rs");
    let text = std::fs::read_to_string(&state).expect("state.rs");
    for alloc in [
        "c2p_error: Field::zeros(&allocated)?",
        "fofc_flag: Field::zeros(&allocated)?",
        "freeze_applied: Field::zeros(&allocated)?",
    ] {
        assert_eq!(
            text.matches(alloc).count(),
            1,
            "channel allocation drifted: {alloc}"
        );
    }
}

#[test]
fn channel_gate_flags_cross_channel_borrowing() {
    // the freeze output bound to the TroubledCell buffer is caught: the
    // `.fofc_flag` spelling sits outside its sanctioned functions.
    let freeze_to_flag = "fn rogue_select() {\n    fofc_select(sim, &ws.fofc_flag);\n}\n";
    assert_eq!(channel_violations(freeze_to_flag).len(), 1);

    // the status buffer read from an unrelated helper is caught.
    let stray_read = "fn stray_helper() {\n    let x = sim.fields.c2p_error.view();\n}\n";
    assert_eq!(channel_violations(stray_read).len(), 1);

    // the freeze mask written by a non-select dispatch is caught.
    let rogue_write =
        "fn other_kernel() {\n    dispatch(sim, Some(&sim.fields.freeze_applied));\n}\n";
    assert_eq!(channel_violations(rogue_write).len(), 1);

    // lawful producer and reader spellings pass for each channel.
    let lawful = "fn c2p() {\n    dispatch(sim, Some(&sim.fields.c2p_error));\n}\n\n\
                  fn troubled_from_status() {\n    go(&[&sim.fields.c2p_error]);\n}\n\n\
                  fn scan_c2p_errors() {\n    let v = sim.fields.c2p_error.view();\n}\n\n\
                  fn fofc_orchestrate() {\n    let flag = &ws.fofc_flag;\n    \
                  let freeze = &ws.freeze_applied;\n}\n";
    assert!(channel_violations(lawful).is_empty());

    // a call whose name extends a channel spelling is a different identifier.
    let count_call = "fn stray_count() {\n    let n = kernels.fofc_flags_in(sim, region);\n}\n";
    assert!(channel_violations(count_call).is_empty());
    // the field spelled at the end of a line, or before punctuation, is still caught.
    let field_at_end = "fn stray_view() {\n    let f = &ws.fofc_flag;\n    let g = ws.fofc_flag\n}\n";
    assert_eq!(channel_violations(field_at_end).len(), 2);
}

#[test]
fn the_inspection_law_admits_reads_alone() {
    assert!(reads_only("let status = self.fields.c2p_error.view();"));
    assert!(!reads_only("let status = self.fields.c2p_error.view_mut();"), "a mutable view passed as a read");
    assert!(!reads_only("self.fields.c2p_error.set(c, Sc::from_f64(0.0));"), "a write passed as a read");
    assert!(!reads_only("let flag = &self.workspace.fofc_flag;"), "a bare borrow passed as a read");
}

#[test]
fn the_transfer_law_admits_whole_buffers_alone() {
    assert!(whole_buffers_only("c2p_error: mirror_field(&f.c2p_error)?,"));
    assert!(whole_buffers_only("+ f.c2p_error.domain().volume()"));
    assert!(!whole_buffers_only("w.fofc_flag.set(c, Sc::from_f64(1.0));"), "an element write passed as a transfer");
    assert!(!whole_buffers_only("let v = f.c2p_error.view();"), "an element read passed as a transfer");
    assert!(!whole_buffers_only("*w.freeze_applied.view_mut().at_mut(c) = x;"), "an element write passed as a transfer");
}

#[test]
fn a_granted_function_that_breaks_its_law_is_flagged() {
    // the scanner reports the granted function's access; the law decides it
    let text = "fn mirror_store() {\n    w.fofc_flag.set(c, 1.0);\n}\n";
    let hits = channel_violations(text);
    assert_eq!(hits.len(), 1, "the scanner missed the access");
    let (_, function, code) = &hits[0];
    assert_eq!(function, "mirror_store");
    assert!(!whole_buffers_only(code), "an element write inside a transfer grant passed its law");
}
