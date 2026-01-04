# =============================================================================
# gpu/meson.build
#
# extensible gpu backend compilation supporting cuda, hip, sycl, metal
# uses custom targets to handle .cpp files with gpu code
# batched linking to prevent oom
# =============================================================================

fs = import('fs')

# gpu backend detection
cuda = dependency('CUDA', required: false)
hip = dependency('HIP', required: false)
# sycl = dependency('sycl', required: false)  # future
# metal = dependency('metal', required: false)  # future

if not cuda.found() and not hip.found()
  error('gpu compilation enabled but no gpu runtime found')
endif

# gpu-specific sources (cuda implementation + backend dispatch)
gpu_sources = files(
  '../src/hesi/cuda/event.cpp',
  '../src/hesi/cuda/stream.cpp',
  '../src/hesi/cuda/memory.cpp',
  '../src/hesi/cuda/transfer.cpp',
  '../src/hesi/backend/event.cpp',
  '../src/hesi/backend/stream.cpp',
  '../src/hesi/backend/memory.cpp',
  '../src/hesi/backend/transfer.cpp',
)

# common gpu flags
gpu_common_flags = [
  '-DGPU_ENABLED=1',
  '-DMANAGED_MEMORY',
]

# build include flags manually since not all dependency types support get_compile_args()
# just use the essential paths that we know we need
include_flags = [
  '-I' + meson.current_source_dir() / '..' / 'src',
  '-I' + meson.current_build_dir() / '..',
]

# add pybind11 include (we know this works)
pybind11_inc = run_command(
  py3,
  ['-c', 'import pybind11; print(pybind11.get_include())'],
  check: true
).stdout().strip()
include_flags += ['-I' + pybind11_inc]

# add python include
python_inc = run_command(
  py3,
  ['-c', 'import sysconfig; print(sysconfig.get_path("include"))'],
  check: true
).stdout().strip()
include_flags += ['-I' + python_inc]

# hdf5 include - try pkg-config
hdf5_inc = run_command(
  'pkg-config',
  '--cflags-only-I',
  'hdf5',
  check: false
).stdout().strip()
if hdf5_inc != ''
  include_flags += [hdf5_inc]
endif

# =============================================================================
# backend configurations
# each backend specifies: compiler, compile args, link args, device link args
# =============================================================================

cuda_config = {
  'name': 'cuda',
  'compiler': 'nvcc',
  'defines': ['-DCUDA_ENABLED'],
  'compile_args': [
    '-x', 'cu',
    '-dc',
    '-rdc=true',
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    '-std=c++20',
    '--compiler-options', '-fPIC',
    '--compiler-options', '-fdiagnostics-color=always',
  ],
  'link_args': ['-L/opt/cuda/lib64', '-lcudart', '-lcudadevrt'],
  'device_link_args': ['-dlink', '-rdc=true', '-Xcompiler', '-fPIC'],
  'device_link_libs': ['-lcudadevrt'],
}

hip_config = {
  'name': 'hip',
  'compiler': 'hipcc',
  'defines': ['-DHIP_ENABLED'],
  'compile_args': [
    '-fPIC',
    '-fdiagnostics-color=always',
    '-std=c++20',
  ],
  'link_args': ['-lamdhip64'],
  'device_link_args': [],  # hip doesn't need separate device link
  'device_link_libs': [],
}

# future backends
# sycl_config = { ... }
# metal_config = { ... }

# select active backend
if cuda.found()
  backend = cuda_config
  gpu_dep = cuda
elif hip.found()
  backend = hip_config
  gpu_dep = hip
endif

compiler = find_program(backend['compiler'])

# =============================================================================
# compile gpu sources to object files
# =============================================================================

gpu_objects = []

foreach src : gpu_sources
  obj_name = fs.name(fs.parent(src)) + '_' + fs.stem(src) + '.o'
  
  gpu_objects += custom_target(
    obj_name,
    input: src,
    output: obj_name,
    command: [
      compiler,
      include_flags,
      backend['compile_args'],
      backend['defines'],
      gpu_common_flags,
      '-c', '@INPUT@',
      '-o', '@OUTPUT@',
    ],
  )
endforeach

# =============================================================================
# device linking (if needed)
# batch objects to prevent oom - link in chunks, then final link
# =============================================================================

device_link_objects = []

if backend['device_link_args'].length() > 0
  # cuda requires device linking
  # batch size: adjust based on available memory (10-15 objects per batch is safe)
  batch_size = 10
  batch_count = (gpu_objects.length() + batch_size - 1) / batch_size
  
  # create batched device link objects
  batch_idx = 0
  foreach i : range(0, gpu_objects.length(), batch_size)
    batch_end = i + batch_size
    if batch_end > gpu_objects.length()
      batch_end = gpu_objects.length()
    endif
    
    batch_objs = []
    foreach j : range(i, batch_end)
      batch_objs += gpu_objects[j]
    endforeach
    
    device_link_objects += custom_target(
      'gpu_dlink_batch_' + batch_idx.to_string(),
      input: batch_objs,
      output: 'gpu_dlink_batch_' + batch_idx.to_string() + '.o',
      command: [
        compiler,
        backend['device_link_args'],
        '@INPUT@',
        '-o', '@OUTPUT@',
      ] + backend['device_link_libs'],
    )
    
    batch_idx += 1
  endforeach
  
  # final device link combining all batches
  if device_link_objects.length() > 1
    final_dlink = custom_target(
      'gpu_dlink_final',
      input: device_link_objects,
      output: 'gpu_dlink_final.o',
      command: [
        compiler,
        backend['device_link_args'],
        '@INPUT@',
        '-o', '@OUTPUT@',
      ] + backend['device_link_libs'],
    )
    all_gpu_objects = gpu_objects + [final_dlink]
  else
    all_gpu_objects = gpu_objects + device_link_objects
  endif
else
  # hip doesn't need device linking
  all_gpu_objects = gpu_objects
endif

# =============================================================================
# build python extension module
# =============================================================================

py3.extension_module(
  'gpu_ext',
  '../src/bindings/gpu_binding.cpp',
  objects: all_gpu_objects,
  include_directories: all_inc,
  dependencies: general_dependencies + [gpu_dep],
  link_with: simbi_core,
  link_whole: simbi_core,
  cpp_args: backend['defines'] + gpu_common_flags,
  link_args: backend['link_args'],
  install: true,
  install_dir: py_install_dir,
)
