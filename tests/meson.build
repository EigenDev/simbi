# tests/meson.build

# the executable
boundary_test_exe = executable(
  'boundary_test',
  files('test_boundary.cpp') + sources,
  include_directories: [
    all_inc, # Points to src/
    include_directories('..'),
  ],
  dependencies: [openmp_dep, hdf5_dep],
)

dynamic_boundary_test_exe = executable(
  'dynamic_boundary_test',
  files('test_vm.cpp') + sources,
  include_directories: [
    all_inc, # Points to src/
    include_directories('..'),
  ],
  dependencies: [openmp_dep, hdf5_dep],
)

field_test_exe = executable(
  'field_test',
  files('test_field.cpp') + sources,
  include_directories: [
    all_inc, # Points to src/
    include_directories('..'),
  ],
  dependencies: [openmp_dep, hdf5_dep],
)
smr_fuller_test_exe = executable(
  'smr_fuller_test',
  files('test_smr.cpp') + sources,
  include_directories: [
    all_inc, # Points to src/
    include_directories('..'),
  ],
  dependencies: [openmp_dep, hdf5_dep],
)

reduction_test_exe = executable(
  'reduction_test',
  files('test_reduction.cpp') + sources,
  include_directories: [
    all_inc, # Points to src/
    include_directories('..'),
  ],
  dependencies: [openmp_dep, hdf5_dep],
)

ecs_amr_build_exe = executable(
  'ecs_amr_build_test',
  files('test_ecs_amr.cpp') + sources,
  include_directories: [
    all_inc, # Points to src/
    include_directories('..'),
  ],
  dependencies: [openmp_dep, hdf5_dep],
)

# reg it with the test runner
test('Boundary Logic Check', boundary_test_exe, verbose: true)
test(
  'Dynamic Boundary Management Check',
  dynamic_boundary_test_exe,
  verbose: true,
)
test('Field Operations Check', field_test_exe, verbose: true)
test('SMR Fuller Functionality Check', smr_fuller_test_exe, verbose: true)
test('Reduction Operations Check', reduction_test_exe, verbose: true)
test('ECS AMR Build Check', ecs_amr_build_exe, verbose: true)
