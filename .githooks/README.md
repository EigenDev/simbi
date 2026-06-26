# git hooks

## pre-push

runs test suite before allowing push to remote.

### installation

hooks are automatically installed in `.git/hooks/` when you run any command that touches the repo.

### bypass (emergencies only)

```bash
git push --no-verify
```

### how it works

1. runs the rust backend test suite
2. allows push only if all tests pass

### running the tests

```bash
cargo test --manifest-path src/Cargo.toml
```
