# =============================================================================
# meson.build
#
# main build configuration for simbi
# =============================================================================

project(
  'simbi',
  'c',
  'cpp',
  version: run_command('git', 'describe', '--tag', '--abbrev=0', check: false).stdout().strip(),
  license: 'MIT',
  meson_version: '>=1.2.0',
  default_options: {
    'cpp_std': 'c++20',
    'warning_level': '3',
    'buildtype': 'release',
    'optimization': '3',
    'b_lto': true,
  },
)

# detect and use fastest available linker
linker_preference = ['mold', 'lld', 'gold', 'bfd']
selected_linker = 'bfd'  # fallback

foreach linker : linker_preference
  if find_program(linker, required: false).found()
    selected_linker = linker
    break
  endif
endforeach

if selected_linker != 'bfd'
  add_project_link_arguments('-fuse-ld=' + selected_linker, language: ['c', 'cpp'])
endif

# common project arguments
add_project_arguments(
  [
    '-Wno-nested-anon-types',
    '-Wno-gnu-anonymous-struct',
  ],
  language: 'cpp',
)

# python and pybind11 setup
py3 = import('python').find_installation('python3', required: true)
py_install_dir = meson.current_source_dir() / 'simbi' / 'libs'
pybind11_dep = dependency('pybind11')

# limit error output during compilation
cc = meson.get_compiler('cpp')
if cc.get_id() == 'clang'
  add_project_arguments('-ferror-limit=1', language: 'cpp')
elif cc.get_id() == 'gcc'
  add_project_arguments('-fmax-errors=1', language: 'cpp')
endif

# build configuration
conf_data = configuration_data()
foreach opt : ['four_velocity', 'column_major']
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

# dependencies
openmp_dep = dependency('openmp', language: 'cpp', required: true)
hdf5_dep = dependency('hdf5', language: 'cpp', required: true)

all_inc = include_directories('src')

# include subdirectories
subdir('src')
subdir('simbi/afterglow')

general_dependencies = [
  openmp_dep,
  hdf5_dep,
  py3.dependency(),
  pybind11_dep,
]

# compile once, link many
simbi_core = static_library(
  'simbi_core',
  sources + cpu_sources,
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

# =============================================================================
# build configuration summary
# =============================================================================

summary({
  'Buildtype': get_option('buildtype'),
  'Optimization': get_option('optimization'),
  'C++ Standard': get_option('cpp_std'),
  'Precision': get_option('precision'),
  'Column Major': get_option('column_major'),
  'Four Velocity': get_option('four_velocity'),
  'GPU Compilation': get_option('gpu_compilation'),
  'Build Tests': get_option('build_tests'),
  'Compiler': cc.get_id() + ' ' + cc.version(),
  'Linker': selected_linker,
  'LTO': get_option('b_lto'),
  'OpenMP': openmp_dep.found(),
  'HDF5': hdf5_dep.found(),
}, section: 'Configuration', bool_yn: true)
