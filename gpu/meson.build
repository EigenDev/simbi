# Check for GPU dependencies
hip = dependency('HIP', cmake_module_path: '/opt/rocm', modules: ['hip::device', 'hip::host'], required: false)
cuda = dependency('CUDA', required: false)

gpu_comp_args = []
gpu_link_args = []

gpu_settings = {
    'nvidia': {
        'compiler': 'nvcc',
        'arch': f'sm_@get_option("gpu_arch", "86")@',
        'args': ['-fPIC', f'-arch=sm_@get_option("gpu_arch", "86")@', '-dc', '-dlto', '--expt-relaxed-constexpr', '-O3'],
        'linker': ['-lcudart'],
        'gpu_dependency': cuda
    },
    'amd': {
        'compiler': 'hipcc',
        'arch': f'gfx@get_option("gpu_arch", "906")@',
        'args': ['-fPIC', f'--offload-arch=gfx@get_option("gpu_arch", "70")@', '-O3'],
        'linker': ['-lamdhip64'],
        'gpu_dependency': hip
    }
}

settings = {}
if hip.found() or cuda.found()
    if hip.found()
        settings = gpu_settings['amd']
    elif cuda.found()
        settings = gpu_settings['nvidia']
    endif
    gpu_compiler = find_program(settings['compiler'])
    gpu_args = settings['args']
    gpu_linkers = settings['linker']
endif

if hip.found() or cuda.found()
    if hip.found()
        settings = gpu_settings['amd']
    elif cuda.found()
        settings = gpu_settings['nvidia']
    endif
    gpu_compiler = find_program(settings['compiler'])

    # Compile GPU sources
    gpu_objs = []
    gpu_objs = custom_target(
        'gpu_objects',
        input: sources,
        output: '@BASENAME@.o',
        command: [gpu_compiler, '@INPUT@', '-o', '@OUTPUT@'] + gpu_args,
        build_by_default: true
    )

    # Link GPU objects into a static library
    static_library(
        'gpu_lib',
        gpu_objs,
        link_args: settings['linker'],
        install: false
    )
endif