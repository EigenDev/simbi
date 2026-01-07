# =============================================================================
# src/meson.build
#
# source compilation with gpu backend support
# =============================================================================

# core source files that need compilation
sources = files(
  'bindings/config_converter.cpp',
  'bindings/driver.cpp',
  'bindings/evolver.cpp',
  'context/progress.cpp',
  'dag/evaluator.cpp',
  'dag/exp_load.cpp',
  'dag/linearize.cpp',
  'io/console/statistics.cpp',
  'io/exceptions.cpp',
  'io/tabulate/table.cpp',
  'physics/em/contact.cpp',
  'physics/em/zero.cpp',
  'physics/ib/factory.cpp',
  'utility/helpers.cpp',
)

# =============================================================================
# gpu compilation system
# =============================================================================

if get_option('gpu_compilation').enabled()
  # detect gpu backend and validate environment
  cuda_dep = dependency('CUDA', required: false)
  hip_dep = dependency('HIP', required: false)

  gpu_backend = 'none'
  gpu_compiler = 'none'
  gpu_runtime_deps = []

  if cuda_dep.found() and find_program('nvcc', required: false).found()
    gpu_backend = 'cuda'
    gpu_compiler = 'nvcc'
    gpu_runtime_deps = [cuda_dep]
  elif hip_dep.found() and find_program('hipcc', required: false).found()
    gpu_backend = 'hip'
    gpu_compiler = 'hipcc'
    gpu_runtime_deps = [hip_dep]
  else
    error('gpu compilation enabled but no supported backend found (cuda/hip)')
  endif

  message('gpu backend: ' + gpu_backend)

  # build include paths for gpu compilation
  gpu_includes = [
    meson.current_source_dir(),
    meson.project_build_root(),
  ]

  # add python includes
  pybind11_inc = run_command(py3, ['-c', 'import pybind11; print(pybind11.get_include())'], check: true).stdout().strip()
  python_inc = run_command(
    py3,
    ['-c', 'import sysconfig; print(sysconfig.get_path("include"))'],
    check: true,
  ).stdout().strip()

  gpu_includes += [pybind11_inc, python_inc]

  # add hdf5 includes
  hdf5_result = run_command('pkg-config', '--cflags-only-I', 'hdf5', check: false)
  if hdf5_result.returncode() == 0
    hdf5_inc = hdf5_result.stdout().strip()
    if hdf5_inc != ''
      gpu_includes += [hdf5_inc]
    endif
  endif

  # build compiler flags based on backend
  gpu_flags = []
  if gpu_backend == 'cuda'
    gpu_flags = [
      '-x', 'cu',
      '--extended-lambda',
      '--expt-relaxed-constexpr',
      '-std=c++20',
      '--compiler-options',
      '-fPIC',
      '--compiler-options',
      '-fopenmp',
      '--compiler-options',
      '-O3',
      '--compiler-options',
      '-DNDEBUG',
      '-O3',
    ]

    device_arch = get_option('device_arch')
    if device_arch != ''
      gpu_flags += ['-arch', device_arch]
    endif

  elif gpu_backend == 'hip'
    gpu_flags = [
      '-x', 'hip',
      '-std=c++20',
      '-fPIC',
      '-fopenmp',
      '-O3',
      '-DNDEBUG',
    ]

    device_arch = get_option('device_arch')
    if device_arch != ''
      gpu_flags += ['--offload-arch', device_arch]
    endif
  endif

  # add includes to flags
  foreach inc : gpu_includes
    gpu_flags += ['-I', inc]
  endforeach

  # add gpu defines
  gpu_defines = [
    'GPU_ENABLED=1',
    gpu_backend.to_upper() + '_ENABLED=1',
    'XPU_' + gpu_backend.to_upper() + '_AVAILABLE',
    'MANAGED_MEMORY',
  ]

  foreach def : gpu_defines
    gpu_flags += ['-D', def]
  endforeach

  # compile gpu objects individually for reliability
  gpu_objects = []
  foreach src : sources
    src_name = src.full_path().split('/')[-1]
    obj_name = src_name.replace('.cpp', '_gpu.o')

    gpu_obj = custom_target(
      obj_name,
      input: src,
      output: obj_name,
      command: [gpu_compiler] + gpu_flags + ['-c', '@INPUT@', '-o', '@OUTPUT@'],
      install: false,
    )

    gpu_objects += gpu_obj
  endforeach

  # create gpu library
  simbi_gpu_core = static_library(
    'simbi_gpu_core',
    objects: gpu_objects,
    install: false,
  )

  # gpu build summary
  summary(
    {
      'Backend': gpu_backend,
      'Compiler': gpu_compiler,
      'Sources': sources.length().to_string(),
    },
    section: 'GPU Compilation',
    bool_yn: false,
  )

endif
