# git hooks

## Pre-commit hook

The hook rejects symlinks anywhere in the staged tree, including renamed links,
regular files changed into links, and links already present in the index. Local,
untracked symlinks are allowed. `simbi_configs/science` stays local and is ignored.

Git ignore patterns match paths rather than file types, so the hook enforces the
type-based policy. Hooks apply only where installed and can be bypassed explicitly.

Enable all repository hooks with:

```bash
git config core.hooksPath .githooks
```

Run the isolated symlink-policy tests with:

```bash
python -m unittest discover -s .githooks/tests -v
```

## Pre-push hook

The hook runs two checks before a push: an authorship guard and a fast build
check. It does not run the test suite; the full suite runs in CI.

The authorship guard rejects any push whose tip can reach a commit carrying a
machine co-author trailer. Authorship denotes accountability for the code,
which an automated tool cannot hold; tooling credit lives in README.md.

It checks every reachable commit, not just new ones, because a merge that pulls
a pre-rewrite lineage back alongside a rewritten one restores such a trailer
without introducing any new commit that carries it. After history is rewritten,
sync other clones with `git reset --hard origin/<branch>` rather than
`git pull`, which re-merges the superseded lineage.

### Installation

Copy `pre-push` into `.git/hooks/`, or point `core.hooksPath` at this directory:

```bash
cp .githooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

### Bypass

```bash
git push --no-verify
```

Use `--no-verify` only when you have a specific reason to bypass the checks.

### How it works

1. scans each pushed tip for a reachable machine co-author trailer and rejects
   the push, naming the offending commit, if one is found (ref deletions, which
   arrive as an all-zero sha, are skipped)
2. checks that `cargo` and the `src/` workspace are present (skips if not)
3. runs `cargo check --workspace --all-targets --manifest-path src/Cargo.toml`
   (type-checks lib + bins + tests + examples, no codegen, no test run)
4. allows push only if the workspace compiles

This catches common build errors, such as a signature change that misses a call
site or a stale example. Run the relevant tests locally as well; the complete
suite also runs in CI:

```bash
cargo test --release --manifest-path src/Cargo.toml --workspace          # cpu
cargo test --features cuda --manifest-path src/Cargo.toml                 # gpu host
```
