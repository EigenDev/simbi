# Check for GPU dependencies
hip  = dependency('HIP', cmake_module_path: '/opt/rocm', modules: ['hip::device', 'hip::host'], required: false)
cuda = dependency('CUDA', required: false)
fs = import('fs')
include_flags = []
foreach inc : include_dirs
    include_flags += ['-I' + inc]
endforeach

gpu_flags = [
    '-DGPU_CODE=1',
    '-DMANAGED_MEMORY',
]

# Get CUDA paths
cuda_path = run_command('which', 'nvcc', check: false).stdout().strip()
cuda_root = ''
if cuda_path != ''
    cuda_root = fs.parent(fs.parent(cuda_path))
endif

# Get ROCm paths
rocm_path = run_command('hipconfig', '--path', check: false).stdout().strip()

gpu_arch = get_option('gpu_arch')
gpu_settings = {
    'nvidia': {
        'compiler': 'nvcc',
        'arch': f'-arch=sm_@gpu_arch@',
        'args': [
            f'-arch=sm_@gpu_arch@',
            '-x',
            'cu',
            '-dc',
            '--extended-lambda',
            '-w',
            '--expt-relaxed-constexpr',
            '-std=c++20',
            '--compiler-options',
            '-fPIC',
        ],
        'link_args': [f'-L@cuda_root@/lib64','-lcudart', '-lnvrtc', '-lcuda'],
        'inter_link_args': ['-dlink', '-Xcompiler', '-fPIC'],
        'extra_link_args': ['-lcudadevrt'],
        'gpu_dependency': cuda
    },
    'amd': {
        'compiler': 'hipcc',
        'arch': f'gfx@gpu_arch@',
        'args': ['-fPIC', f'--offload-arch=gfx@gpu_arch@'],
        'link_args': [f'-L@rocm_path@/lib','-lamdhip64', '-lhiprtc'],
        'inter_link_args': [],
        'extra_link_args': [],
        'gpu_dependency': hip
    }
}



settings = {}
if hip.found() or cuda.found()
    if hip.found()
        if run_command('hipconfig', '--platform', check: true).stdout().strip() == 'nvidia'
            settings = gpu_settings['nvidia']
        else
            settings = gpu_settings['amd']
        endif
    elif cuda.found()
        settings = gpu_settings['nvidia']
    endif
    gpu_compiler = find_program(settings['compiler'])

    # Compile GPU sources
    gpu_objs = []
    foreach src : sources
        gpu_objs += [custom_target(
            fs.name(src) + '_gpu_obj',
            input: src,
            output: '@BASENAME@.o',
            command: [gpu_compiler, include_flags, settings['args'] , '-c', '@INPUT@', '-o', '@OUTPUT@', gpu_flags],
            depend_files: [headers, templates],
            build_by_default: true
        )]
    endforeach

    # perform intermediate link step
    gpu_objs += [custom_target(
        'gpu_link',
        input: gpu_objs,
        output: 'gpu_link.o',
        command: [gpu_compiler, settings['link_args'], settings['inter_link_args'], '@INPUT@', '-o', '@OUTPUT@'] + settings['extra_link_args'] + settings['arch'],
        build_by_default: true
    )]

    py3.extension_module(
        'gpu_ext',
        ['../src/gpu_ext.pyx'] + gpu_objs,
        link_args: settings['link_args'],
        include_directories: include_directories('..', '../src', numpy_include),
        dependencies: general_dependencies + settings['gpu_dependency'],
        cpp_args: ['-DMANAGED_MEMORY'],
        override_options: ['cython_language=cpp'],
        install: true,
        install_dir: py_install_dir,
    )
endif
