# =============================================================================
# meson.build
#
# build configuration for simbi
# =============================================================================

project(
  'simbi',
  'c',
  'cpp',
  version: run_command('git', 'describe', '--tag', '--abbrev=0', check: false).stdout().strip(),
  license: 'MIT',
  meson_version: '>=1.4.0',
  default_options: {
    'cpp_std': 'c++20',
    'warning_level': '3',
    'buildtype': 'release',
    'optimization': '3',
    'b_lto': true,
  },
)

# python setup (needed for both backends)
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() / 'simbi' / 'libs'

backend = get_option('hydro_backend')

if backend == 'cpp'
  # ---- legacy C++ hydro backend — DISABLED by default. the C++ source in src/
  #      is KEPT but compiled ONLY with -Dhydro_backend=cpp. all the C++ deps,
  #      the build_options.hpp config header, and the C++ rad_hydro module live
  #      here, so a default (rust) build pulls in no C++ at all. ----
  pybind11_dep = dependency('pybind11')
  cc = meson.get_compiler('cpp')
  openmp_dep = dependency('openmp', language: 'cpp', required: true)
  hdf5_dep = dependency('hdf5', language: 'cpp', required: true)
  all_inc = include_directories('src')

  conf_data = configuration_data()
  foreach opt : ['four_velocity', 'column_major', 'unified_memory']
    conf_data.set10(opt.to_upper(), get_option(opt))
  endforeach
  conf_data.set10('FLOAT_PRECISION', get_option('precision') == 'single')
  conf_data.set(
    'BUILD_DATE',
    run_command('date', '+%Y-%m-%d %H:%M:%S', check: true).stdout().strip(),
  )
  configure_file(
    input: 'build_options.hpp.in',
    output: 'build_options.hpp',
    configuration: conf_data,
  )

  general_dependencies = [openmp_dep, hdf5_dep, py3.dependency(), pybind11_dep]

  # radiation post-processing (rad_hydro) — C++; replaced by symbi-afterglow on
  # the rust path, so it builds only with the C++ backend.
  subdir('simbi/afterglow')

  # include src subdirectory first to define sources and build gpu library
  subdir('src')

  # compile cpu core library after sources are defined
  simbi_core = static_library(
    'simbi_core',
    sources,
    include_directories: all_inc,
    dependencies: general_dependencies,
  )

  # cpu extension module
  py3.extension_module(
    'cpu_ext',
    'src/bindings/cpu_binding.cpp',
    include_directories: all_inc,
    dependencies: general_dependencies,
    link_with: simbi_core,
    install: true,
    install_dir: py_install_dir,
  )

  # gpu compilation
  if get_option('gpu_compilation').enabled()
    subdir('gpu')
  endif

  # tests
  if get_option('build_tests')
    subdir('tests')
  endif
else
  # ---- rust hydro backend: build the symbi-py cdylib, install as cpu_ext ----
  # the unchanged `simbi.libs.cpu_ext` import then loads rust with no other change.
  cargo = find_program('cargo')
  ext_suffix = run_command(
    py3, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))',
    check: true,
  ).stdout().strip()
  rust_src_dir = meson.current_source_dir() / 'rust_src'
  rust_build = files('scripts/build_rust_backend.py')

  # cargo feature set for the hydro cdylib. user-supplied (`-Drust_features=`)
  # plus the `cuda` feature when gpu compilation is enabled — the rust backend's
  # gpu path is a cargo feature, not the c++ `gpu/` subdir.
  rust_feats = get_option('cargo_features')
  if get_option('gpu_compilation').enabled()
    rust_feats = rust_feats == '' ? 'cuda' : rust_feats + ',cuda'
  endif

  # cargo is the incremental authority: `build_always_stale` makes ninja delegate
  # to cargo on every `meson compile`, so a `.rs` edit is always picked up (cargo
  # no-ops in well under a second when nothing changed). without it the no-input
  # custom_target is cached on its existing output and `.rs` edits are missed.

  # the hydro backend -> simbi.libs.cpu_ext
  custom_target(
    'cpu_ext_rust',
    output: 'cpu_ext' + ext_suffix,
    command: [py3, rust_build, '@OUTPUT@', 'cpu_ext', rust_src_dir, 'symbi-py', rust_feats],
    build_by_default: true,
    build_always_stale: true,
    install: true,
    install_dir: py_install_dir,
  )

  # the radiation post-processing module (rust symbi-afterglow) -> simbi.libs.rad_hydro.
  # symbi-rad-py exposes no cargo features, so the gpu/user feature set is NOT
  # forwarded here.
  custom_target(
    'rad_hydro_rust',
    output: 'rad_hydro' + ext_suffix,
    command: [py3, rust_build, '@OUTPUT@', 'rad_hydro', rust_src_dir, 'symbi-rad-py'],
    build_by_default: true,
    build_always_stale: true,
    install: true,
    install_dir: py_install_dir,
  )
endif

# =============================================================================
# build summary
# =============================================================================

summary(
  {
    'Hydro Backend': backend,
    'Buildtype': get_option('buildtype'),
    'Optimization': get_option('optimization'),
    'Precision': get_option('precision'),
    'GPU Compilation': get_option('gpu_compilation'),
  },
  section: 'Configuration',
  bool_yn: true,
)

# C++-specific summary only when the (disabled-by-default) cpp backend is built.
if backend == 'cpp'
  summary(
    {
      'C++ Standard': get_option('cpp_std'),
      'Column Major': get_option('column_major'),
      'Four Velocity Prim': get_option('four_velocity'),
      'Build Tests': get_option('build_tests'),
      'Compiler': cc.get_id() + ' ' + cc.version(),
      'OpenMP': openmp_dep.found(),
      'HDF5': hdf5_dep.found(),
    },
    section: 'C++ backend',
    bool_yn: true,
  )
endif
