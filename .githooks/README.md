# git hooks

## pre-push

two gates before allowing push to remote: an authorship guard, then a fast
build check. it does NOT run the test suite -- that belongs in CI.

the authorship guard rejects any push whose tip can reach a commit carrying a
machine co-author trailer. authorship denotes accountability for the code,
which an automated tool cannot hold; tooling credit lives in README.md.

it checks every reachable commit, not just new ones, because a merge that pulls
a pre-rewrite lineage back alongside a rewritten one restores such a trailer
without introducing any new commit that carries it. after history is rewritten,
sync other clones with `git reset --hard origin/<branch>` rather than
`git pull`, which re-merges the superseded lineage.

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

1. scans each pushed tip for a reachable machine co-author trailer and rejects
   the push, naming the offending commit, if one is found (ref deletions, which
   arrive as an all-zero sha, are skipped)
2. checks that `cargo` and the `src/` workspace are present (skips if not)
3. runs `cargo check --workspace --all-targets --manifest-path src/Cargo.toml`
   (type-checks lib + bins + tests + examples, no codegen, no test run)
4. allows push only if the workspace compiles

this catches the common breakage -- a signature change that misses a call site,
a stale example -- in under a minute. run the actual test suite yourself (or in
CI) when relevant:

```bash
cargo test --release --manifest-path src/Cargo.toml --workspace          # cpu
cargo test --features cuda --manifest-path src/Cargo.toml                 # gpu host
```
