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
//   function cannot borrow a sanctioned spelling):
//     C2pStatus (`.c2p_error`)      — the regime `c2p` producers, the status
//                                     decode, and the diagnostic scanners;
//     TroubledCell (`.fofc_flag`)   — the orchestrator and the GRMHD ladder,
//                                     which bind it once and pass the local;
//     FreezeApplied (`.freeze_applied`) — the orchestrator, which hands it to
//                                     the correcting selects;
// - the three channels are three distinct field allocations.
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
            "scan_c2p_errors",
            "first_c2p_error",
            "first_c2p_failure_state",
        ],
    },
    ChannelRule {
        // TroubledCell: bound once by the orchestrator (and the GRMHD ladder
        // body), then passed as a local to the ghost fill, splices, and counts.
        spelling: ".fofc_flag",
        allowed_fns: &["fofc_orchestrate", "fofc_impl"],
    },
    ChannelRule {
        // FreezeApplied: bound once by the orchestrator, which hands it to the
        // correcting selects — the only writers — and the count reduction.
        spelling: ".freeze_applied",
        allowed_fns: &["fofc_orchestrate"],
    },
];

/// channel-boundary violations in one source text: comment text is stripped,
/// function extents are brace-tracked (a `fn` line opens a body, brace balance
/// closes it), and a line spelling a channel's field access outside that
/// channel's sanctioned functions is flagged.
fn channel_violations(text: &str) -> Vec<String> {
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
            if code.contains(rule.spelling)
                && !(in_fn && rule.allowed_fns.contains(&fn_name.as_str()))
            {
                hits.push(format!("{number}: {}", raw.trim()));
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
            if CHANNELS.iter().any(|r| code.contains(r.spelling)) {
                seen += 1;
            }
        }
        for hit in channel_violations(&text) {
            hits.push(format!("{}:{hit}", path.display()));
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
}
