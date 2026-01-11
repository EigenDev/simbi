# =============================================================================
# experimental/meson.build
#
# isolated test environment for multi-gpu logic
# =============================================================================

project(
    'xpu_multigpu_test',
    'cpp', 'cuda',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++20',
        'warning_level=2',
        'buildtype=debug',
    ]
)

# cuda dependency
cuda_dep = dependency('cuda', version: '>=11.0', required: true)

# include paths for xpu headers
xpu_includes = include_directories(
    '../..',           # src/
    '../../..',        # root
)

# compiler args
cpp_args = [
    '-DXPU_CUDA_AVAILABLE',
    '-DXPU_USE_CUDA',
]

cuda_args = [
    '-std=c++20',
    '--expt-relaxed-constexpr',
    '--extended-lambda',
    '-arch=sm_75',  # adjust for target GPU
]

# build test executable
test_multigpu = executable(
    'test_multigpu',
    'test_multigpu.cu',
    include_directories: xpu_includes,
    cpp_args: cpp_args,
    cuda_args: cuda_args,
    dependencies: [cuda_dep],
)

# register test (won't run on single-GPU machines)
test('multi-gpu basic', test_multigpu)
