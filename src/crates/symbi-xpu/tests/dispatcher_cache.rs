// =============================================================================
// dispatcher_cache.rs
//
// **the dispatcher discipline canary**: the JIT/binary kernel cache MUST
// dedup by SOURCE CONTENT, not just by user-supplied name. without this:
//
//   1. two callers passing the same `kernel_name` (or `cache_key`) but
//      distinct source strings would silently share a cache slot;
//   2. whoever compiled first wins; the other's launch attempts the
//      first's cached PTX with the SECOND's argument layout;
//   3. CUDA returns `CUDA_ERROR_INVALID_VALUE` from `cuLaunchKernel`,
//      or worse, a wrong-result kernel that silently corrupts data;
//   4. test-ORDERING becomes correctness-critical — a classic distributed-
//      state footgun.
//
// the fix is content-addressed cache keys: `compute_internal_cache_key(name,
// content) = "{name}#{hash(content):016x}"`. distinct contents always
// produce distinct keys regardless of `name`. this test asserts the
// discipline structurally — no GPU required, no `--features cuda` needed.
//
// run: cargo test -p symbi-xpu --test dispatcher_cache
// =============================================================================

use symbi_xpu::runtime::compute_internal_cache_key;

#[test]
fn same_name_distinct_content_produces_distinct_keys() {
    // **the load-bearing canary**: two different kernel sources sharing
    // the same `name` (e.g., both call themselves "mom_source") must NOT
    // share a cache slot.
    let name = "mom_source";
    let source_a = "extern \"C\" __global__ void mom_source(const double* p0, double* o0, unsigned n) { /* 1 input */ }";
    let source_b = "extern \"C\" __global__ void mom_source(const double* p0, const double* p1, double* o0, unsigned n) { /* 2 inputs */ }";

    let key_a = compute_internal_cache_key(name, source_a.as_bytes());
    let key_b = compute_internal_cache_key(name, source_b.as_bytes());

    assert_ne!(
        key_a, key_b,
        "distinct sources under the same name MUST produce distinct cache keys; \
         got `{key_a}` for both — the content-vs-name footgun is back",
    );
}

#[test]
fn identical_content_produces_identical_keys() {
    // dedup correctness: two callers passing the SAME source must hit the
    // same cache slot. without this, kernel compilation is duplicated
    // wastefully every time the same source is requested.
    let name = "shared_kernel";
    let source = "extern \"C\" __global__ void shared_kernel() { return; }";

    let k1 = compute_internal_cache_key(name, source.as_bytes());
    let k2 = compute_internal_cache_key(name, source.as_bytes());

    assert_eq!(
        k1, k2,
        "identical content MUST produce identical cache keys"
    );
}

#[test]
fn name_is_prefixed_for_diagnostics() {
    // the user's `name` is preserved as a prefix — useful for grep, logs,
    // and debugger inspection. the hash suffix enforces correctness.
    let name = "my_diagnostic_label";
    let key = compute_internal_cache_key(name, b"any content");
    assert!(
        key.starts_with(&format!("{name}#")),
        "internal cache key must start with the diagnostic name; got `{key}`",
    );
}

#[test]
fn distinct_names_for_same_content_produce_distinct_keys() {
    // the user might want to RECOMPILE the same kernel under different
    // names (e.g., for separate metric tracking). respect that — distinct
    // names with the same content must still produce distinct cache slots.
    // (the user can opt-in to dedup by sharing the name; the framework
    // doesn't force collapse.)
    let source = b"same content";
    let k_a = compute_internal_cache_key("name_a", source);
    let k_b = compute_internal_cache_key("name_b", source);
    assert_ne!(k_a, k_b, "distinct names must produce distinct cache keys");
}

#[test]
fn very_small_content_changes_produce_different_keys() {
    // single-byte differences in source must produce different cache
    // entries. catches near-misses (e.g., a debug `printf` vs no printf,
    // or one numeric literal differing by an ULP — the latter is a real
    // physics-altering change).
    let s1 = "auto _v_0 = 1.0;";
    let s2 = "auto _v_0 = 1.1;";
    let k1 = compute_internal_cache_key("kernel", s1.as_bytes());
    let k2 = compute_internal_cache_key("kernel", s2.as_bytes());
    assert_ne!(
        k1, k2,
        "ULP-level numeric differences must produce distinct keys"
    );
}

#[test]
fn binary_content_is_hashed_same_as_source() {
    // the dispatcher uses the same content-hash helper for pre-compiled PTX
    // as for NVRTC source (`jit_kernel_keyed`). asserting the helper works on
    // arbitrary byte slices, not just text.
    let bin_a: &[u8] = &[0x7f, 0x45, 0x4c, 0x46, 0x02, 0x01]; // ELF magic-ish
    let bin_b: &[u8] = &[0x7f, 0x45, 0x4c, 0x46, 0x02, 0x02]; // one byte different
    let k_a = compute_internal_cache_key("ptx_blob", bin_a);
    let k_b = compute_internal_cache_key("ptx_blob", bin_b);
    assert_ne!(
        k_a, k_b,
        "byte-level differences in binary must produce distinct keys"
    );
}
