# tests/meson.build

test_inc = [all_inc, include_directories('..')]

cpu_test_list = [
  ['boundary_test', 'test_boundary.cpp', 'Boundary Logic Check'],
  [
    'dynamic_boundary_test',
    'test_vm.cpp',
    'Dynamic Boundary Management Check',
  ],
  ['field_test', 'test_field.cpp', 'Field Operations Check'],
  ['smr_fuller_test', 'test_smr.cpp', 'SMR Fuller Functionality Check'],
  ['reduction_test', 'test_reduction.cpp', 'Reduction Operations Check'],
  ['ecs_amr_build_test', 'test_ecs_amr.cpp', 'ECS AMR Build Check'],
  # [
  #   'volume_scaling_test',
  #   'test_volume_scaling.cpp',
  #   'Volume Scaling System Check',
  # ],
]

gpu_test_list = []

# cpu tests
foreach test_pair : cpu_test_list + gpu_test_list
  exe = executable(
    test_pair[0],
    files(test_pair[1]),
    include_directories: test_inc,
    dependencies: general_dependencies,
    link_with: simbi_core,
  )
  test(test_pair[2], exe, verbose: true)
endforeach

# gpu-specific tests when gpu compilation enabled
if get_option('gpu_compilation').enabled()
  foreach test_pair : gpu_test_list
    gpu_test_obj = custom_target(
      test_pair[0] + '_gpu.o',
      input: files(test_pair[1]),
      output: test_pair[0] + '_gpu.o',
      command: [gpu_compiler] + gpu_flags + ['-c', '@INPUT@', '-o', '@OUTPUT@'],
      install: false,
    )

    device_link_flags = ['-dlink', '-Xcompiler=-fPIC']
    if gpu_backend == 'cuda'
      device_arch = get_option('device_arch')
      if device_arch != ''
        device_link_flags += ['-arch', device_arch]
      endif
    elif gpu_backend == 'hip'
      device_arch = get_option('device_arch')
      if device_arch != ''
        device_link_flags += ['--offload-arch', device_arch]
      endif
    endif

    test_device_link_obj = custom_target(
      test_pair[0] + '_device_link.o',
      output: test_pair[0] + '_device_link.o',
      command: [gpu_compiler] + device_link_flags + [gpu_test_obj] + ['-o', '@OUTPUT@'],
      depends: [gpu_test_obj],
      install: false,
    )

    gpu_exe = executable(
      test_pair[0] + '_gpu',
      objects: [gpu_test_obj, test_device_link_obj],
      include_directories: test_inc,
      dependencies: general_dependencies + gpu_runtime_deps,
      link_with: simbi_gpu_core,
      link_language: 'cpp',
    )
    test(test_pair[2] + ' (GPU)', gpu_exe, verbose: true)
  endforeach
endif
