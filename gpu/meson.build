# =============================================================================
# gpu/meson.build
#
# gpu extension module using precompiled gpu core library
# =============================================================================

# verify gpu compilation is enabled
if not get_option('gpu_compilation').enabled()
  error('gpu subdir should only be processed when gpu_compilation is enabled')
endif

# gpu backend detection and configuration
cuda = dependency('CUDA', required: false)
hip = dependency('HIP', required: false)

if not cuda.found() and not hip.found()
  error('gpu compilation enabled but no gpu runtime found')
endif

# select backend configuration
if cuda.found()
  gpu_dep = cuda
  link_args = [
    '-L/opt/cuda/lib64',
    '-lcudart',
    '-lcudadevrt',
    '-lcuda',
  ]
elif hip.found()
  gpu_dep = hip
  link_args = [
    '-lamdhip64',
  ]
endif

# =============================================================================
# gpu python extension
# =============================================================================

py3.extension_module(
  'gpu_ext',
  '../src/bindings/gpu_binding.cpp',
  include_directories: all_inc,
  dependencies: general_dependencies + [gpu_dep],
  link_with: simbi_gpu_core,
  link_args: link_args,
  link_language: 'cpp',
  install: true,
  install_dir: py_install_dir,

)
