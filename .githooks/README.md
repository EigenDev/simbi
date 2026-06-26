# git hooks

## pre-push

runs the rust test suite before allowing push to remote.

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

<<<<<<< HEAD
1. checks that `cargo` and the `src/` workspace are present (skips if not)
2. runs `cargo test --release --manifest-path src/Cargo.toml --workspace` (cpu)
3. allows push only if all tests pass

gpu-gated tests require a cuda host and are run separately with
`cargo test --features cuda`; the hook stays cpu-only so it is portable.
=======
1. runs the rust backend test suite
2. allows push only if all tests pass

### running the tests

```bash
cargo test --manifest-path src/Cargo.toml
```
>>>>>>> a0f1bee5d4b79e13848645fed2d9774461d3cba5
