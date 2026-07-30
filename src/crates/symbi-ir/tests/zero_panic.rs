// =============================================================================
// zero_panic.rs
//
// the algebraic / primitives core files MUST contain NO `panic!`, `.unwrap()`, or
// `.expect(` in production code. admitted panics live only at I/O / driver boundaries
// (config parse, NVRTC compile, HDF5 write) — those happen in other files. these are the
// substrate constitution and they are zero-panic by invariant.
//
// the files themselves carry `#![deny(clippy::panic, clippy::unwrap_used,
// clippy::expect_used)]`, so `cargo clippy` will catch violations during
// development. this integration test is the CI-side complement: it runs under
// plain `cargo test` (no clippy required) and fails the build on any new
// panic-style escape that slips into the algebraic core.
// =============================================================================

#[test]
fn zero_panic_in_constitution_files() {
    let files = [
        ("src/algebra.rs", "algebra"),
        ("src/primitives.rs", "primitives"),
    ];

    let forbidden_patterns = ["panic!", ".unwrap()", ".expect("];
    let mut violations: Vec<String> = Vec::new();

    for (path, name) in files {
        // integration tests run from the crate root (where Cargo.toml lives),
        // so paths are relative to that.
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(err) => {
                violations.push(format!("{name}: could not read {path}: {err}"));
                continue;
            }
        };

        // production code is everything BEFORE the first `#[cfg(test)]` guard.
        // the in-file test module is allowed to use panic-style macros (it's
        // test code; asserts are normal). this is a coarse heuristic; if a
        // production block ever lands AFTER a test module, this test misses
        // it — but the file-level deny lints catch that.
        let prod_code = content.split("#[cfg(test)]").next().unwrap_or("");

        for (line_idx, line) in prod_code.lines().enumerate() {
            // skip comments — doc comments and `//` lines.
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for forbidden in &forbidden_patterns {
                if line.contains(forbidden) {
                    violations.push(format!(
                        "{name}:{}: forbidden '{forbidden}' in algebraic core\n    {}",
                        line_idx + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        // assert! itself uses panic — this is in a test file, panic-style
        // failures are how a test fails.
        let count = violations.len();
        let report = violations.join("\n");
        assert!(
            false,
            "zero-panic gate failed — {count} violation(s):\n\n{report}",
        );
    }
}
