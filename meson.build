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

# python and pybind11 setup
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() / 'simbi' / 'libs'
pybind11_dep = dependency('pybind11')

# compiler
cc = meson.get_compiler('cpp')

# build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major', 'unified_memory']
  conf_data.set10(opt.to_upper(), get_option(opt))
endforeach
conf_data.set10('STATE_LAYOUT_SOA', get_option('state_layout') == 'soa')
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

# dependencies
openmp_dep = dependency('openmp', language: 'cpp', required: true)
hdf5_dep = dependency('hdf5', language: 'cpp', required: true)

all_inc = include_directories('src')

general_dependencies = [
  openmp_dep,
  hdf5_dep,
  py3.dependency(),
  pybind11_dep,
]

# include src subdirectory first to define sources and build gpu library
subdir('src')

# compile cpu core library after sources are defined
simbi_core = static_library(
  'simbi_core',
  sources,
  include_directories: all_inc,
  dependencies: general_dependencies,
)

# include other subdirectories
subdir('simbi/afterglow')

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

# =============================================================================
# build summary
# =============================================================================

summary(
  {
    'Buildtype': get_option('buildtype'),
    'Optimization': get_option('optimization'),
    'C++ Standard': get_option('cpp_std'),
    'Precision': get_option('precision'),
    'Column Major': get_option('column_major'),
    'Four Velocity Prim': get_option('four_velocity'),
    'State Layout': get_option('state_layout').to_upper(),
    'GPU Compilation': get_option('gpu_compilation'),
    'Build Tests': get_option('build_tests'),
    'Compiler': cc.get_id() + ' ' + cc.version(),
    'OpenMP': openmp_dep.found(),
    'HDF5': hdf5_dep.found(),
  },
  section: 'Configuration',
  bool_yn: true,
)
