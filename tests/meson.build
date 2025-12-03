# tests/meson.build

test_inc = [all_inc, include_directories('..')]

# compile each test once, link against shared core library
foreach test_pair : [
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
]
  exe = executable(
    test_pair[0],
    files(test_pair[1]),
    include_directories: test_inc,
    dependencies: general_dependencies,
    link_with: simbi_core,
  )
  test(test_pair[2], exe, verbose: true)
endforeach
