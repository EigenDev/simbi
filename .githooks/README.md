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

1. checks if build directory exists
2. checks if tests are enabled in build
3. runs `meson test -C build --print-errorlogs`
4. allows push only if all tests pass

### enabling tests in build

```bash
./dev.py build --tests
```

or with meson directly:

```bash
meson setup build -Dbuild_tests=true
meson compile -C build
```
