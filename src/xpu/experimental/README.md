# xpu multi-gpu experimental tests

isolated test environment for multi-gpu functionality.

## requirements

- 2+ nvidia gpus with peer access support
- cuda 11.0 or later
- meson build system

## building

```bash
cd src/xpu/experimental
meson setup build
meson compile -C build
```

## running

```bash
./build/test_multigpu
```

## what gets tested

1. **device affinity tracking** - memory blocks remember which gpu they belong to
2. **device context guards** - raii ensures device context is restored after operations
3. **multi-device executors** - independent work on different gpus without interference
4. **peer-to-peer transfers** - direct gpu-to-gpu memory copies

## expected output

```
=============================================================================
multi-gpu test suite
=============================================================================
detected 2 cuda devices

device 0: nvidia geforce rtx 3090 (8.6)
device 1: nvidia geforce rtx 3090 (8.6)

test 1: device affinity tracking
  pass: device affinity tracked correctly
test 2: device context guard
  pass: device guard restores context
test 3: multi-device executors
  pass: independent device operations
test 4: peer-to-peer transfer
  pass: peer-to-peer transfer correct

=============================================================================
results: 4 passed, 0 failed
=============================================================================
```

## notes

- test will fail/skip on single-gpu machines
- peer access test skips if gpus don't support p2p
- adjust `arch=sm_75` in meson.build for your gpu architecture
