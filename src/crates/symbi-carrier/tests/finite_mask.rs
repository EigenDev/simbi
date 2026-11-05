// =============================================================================
// finite_mask.rs
//
// the shared finiteness operation: true for every finite value of each precision and false for
// NaN and both infinities, on the evaluating carriers and on the dual carrier, whose tangent has
// no say. the sources route every carrier-generic finiteness test through it: `x - x` of a
// product is that product's rounding error under a compiler that fuses `a * b - c`, so the
// cancellation form rejects finite values on such a target.
//
// run: cargo test -p symbi-carrier --test finite_mask
// =============================================================================
use std::path::Path;

use symbi_carrier::{Dual, Scalar};

#[test]
fn every_finite_f64_is_finite_and_nothing_else_is() {
    let finite = [0.0, -0.0, 1.0, -1.0, f64::MAX, -f64::MAX, f64::MIN_POSITIVE, -f64::MIN_POSITIVE, f64::from_bits(1), -f64::from_bits(1), 7.325450581720952e-2];
    for x in finite {
        assert!(x.is_finite_mask(), "{x:e}");
        assert_eq!(x.is_finite_mask(), x - x == 0.0, "{x:e} left the cancellation form's answer on a non-fusing target");
    }
    for x in [f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!x.is_finite_mask(), "{x:e}");
        assert_eq!(x.is_finite_mask(), x - x == 0.0, "{x:e} left the cancellation form's answer on a non-fusing target");
    }
    // the largest finite value stays inside the bound
    assert!(f64::MAX.is_finite_mask());
    assert!(!(f64::MAX * 2.0).is_finite_mask());
}

#[test]
fn the_bound_follows_the_precision() {
    for x in [0.0f32, -0.0, f32::MAX, -f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1)] {
        assert!(x.is_finite_mask(), "{x:e}");
    }
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!x.is_finite_mask(), "{x:e}");
    }
    assert!(!(f32::MAX * 2.0).is_finite_mask());
}

#[test]
fn the_dual_carrier_reads_its_value() {
    let dual = |value: f64, tangent: f64| Dual { value, tangent };
    assert!(dual(1.0, f64::INFINITY).is_finite_mask(), "a finite value with an infinite tangent");
    assert!(dual(-f64::MAX, 1.0).is_finite_mask());
    assert!(!dual(f64::NAN, 0.0).is_finite_mask());
    assert!(!dual(f64::INFINITY, 0.0).is_finite_mask());
    assert!(!dual(f64::NEG_INFINITY, 0.0).is_finite_mask());
}

/// the cancellation test in carrier-generic code: `(name - name).cmp_eq(`
fn cancellation_sites(text: &str) -> Vec<String> {
    let mut sites = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        let mut rest = line;
        while let Some(at) = rest.find(".cmp_eq(") {
            let before = rest[..at].trim_end();
            if let Some(inner) = before.strip_suffix(')').and_then(|b| b.rsplit_once('(')).map(|(_, inner)| inner) {
                if let Some((a, b)) = inner.split_once(" - ") {
                    if a.trim() == b.trim() {
                        sites.push(format!("{}: {}", number + 1, line.trim()));
                    }
                }
            }
            rest = &rest[at + 8..];
        }
    }
    sites
}

#[test]
fn the_scan_finds_the_cancellation_form() {
    assert_eq!(cancellation_sites("let f = (v - v).cmp_eq(S::ZERO);").len(), 1);
    assert_eq!(cancellation_sites("let g = (a - b).cmp_eq(S::ZERO);").len(), 0);
    assert_eq!(cancellation_sites("// (v - v).cmp_eq(S::ZERO) in prose").len(), 0);
}

#[test]
fn no_carrier_generic_source_tests_finiteness_by_cancellation() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut found = Vec::new();
    let mut files = 0;
    let mut stack: Vec<_> = ["symbi-hydro", "symbi-discretize", "symbi-carrier", "symbi-ir", "symbi-substrate", "symbi-sim", "symbi-refinement"].iter().map(|name| crates.join(name).join("src")).collect();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{dir:?}: {e}")) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files += 1;
                for site in cancellation_sites(&std::fs::read_to_string(&path).unwrap()) {
                    found.push(format!("{}:{site}", path.display()));
                }
            }
        }
    }
    assert!(files > 100, "the scan read {files} files; it checks too little");
    assert!(found.is_empty(), "{found:#?}");
}
