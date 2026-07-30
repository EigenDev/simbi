# git hooks

## pre-push

a fast build check before allowing push to remote: does the whole workspace
still compile? it does NOT run the test suite -- that belongs in CI.

### installation

copy `pre-push` into `.git/hooks/` (or point `core.hooksPath` at this dir):

```bash
cp .githooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

### bypass (emergencies only)

```bash
git push --no-verify
```

### how it works

1. checks that `cargo` and the `src/` workspace are present (skips if not)
2. runs `cargo check --workspace --all-targets --manifest-path src/Cargo.toml`
   (type-checks lib + bins + tests + examples, no codegen, no test run)
3. allows push only if the workspace compiles

this catches the common breakage -- a signature change that misses a call site,
a stale example -- in under a minute. run the actual test suite yourself (or in
CI) when relevant:

```bash
cargo test --release --manifest-path src/Cargo.toml --workspace          # cpu
cargo test --features cuda --manifest-path src/Cargo.toml                 # gpu host
```
