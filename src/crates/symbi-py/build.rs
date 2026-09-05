// =============================================================================
// build.rs
//
// embeds the source identity of the rust backend in the extension, then applies
// the macos pyo3 linker policy. the identity names the code that was COMPILED,
// not whichever checkout happens to be visible when a job later runs.
//
// on macos the pyo3 `extension-module` feature leaves libpython unlinked —
// the python symbols are resolved at import time by the host
// interpreter. the mach-o linker rejects the resulting undefined symbols
// unless told to defer them, so pass `-undefined dynamic_lookup` scoped
// narrowly to this crate's cdylib artifact.
// =============================================================================

use std::path::{Path, PathBuf};
use std::process::Command;

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn source_identity() -> (String, bool) {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let repo = git(&manifest, &["rev-parse", "--show-toplevel"])
        .map(PathBuf::from)
        .unwrap_or(manifest);

    // a commit changes HEAD; staging changes the index; edits anywhere in the rust
    // workspace change `src/crates`. together these make cargo rerun this probe whenever
    // the compiled backend's source identity can change, without rebuilding rust merely
    // because a python configuration changed (that has its own runtime content hash).
    println!(
        "cargo:rerun-if-changed={}",
        repo.join(".git/HEAD").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo.join(".git/index").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo.join("src/crates").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo.join("src/Cargo.toml").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo.join("src/Cargo.lock").display()
    );

    let sha = std::env::var("SIMBI_BUILD_GIT_SHA")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| git(&repo, &["rev-parse", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = std::env::var("SIMBI_BUILD_GIT_DIRTY")
        .ok()
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or_else(|| {
            git(
                &repo,
                &[
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=no",
                    "--",
                    "src/crates",
                    "src/Cargo.toml",
                    "src/Cargo.lock",
                ],
            )
            .is_some_and(|status| !status.is_empty())
        });
    (sha, dirty)
}

fn main() {
    println!("cargo:rerun-if-env-changed=SIMBI_BUILD_GIT_SHA");
    println!("cargo:rerun-if-env-changed=SIMBI_BUILD_GIT_DIRTY");
    let (sha, dirty) = source_identity();
    println!("cargo:rustc-env=SIMBI_BUILD_GIT_SHA={sha}");
    println!(
        "cargo:rustc-env=SIMBI_BUILD_GIT_DIRTY={}",
        if dirty { "1" } else { "0" }
    );

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}
