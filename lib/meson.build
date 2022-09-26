project('simbi', 'cpp', 'cython',
    version : '0.8.0', 
    license : 'MIT',
    default_options : [ 'cpp_std=c++17' ],
)

# BUILD CONFIG
conf_data = configuration_data()
conf_data.set('version', meson.project_version())
conf_data.set10('FLOAT_PRECISION', get_option('float_precision'))
configure_file(
    input:  'build_options.hpp.in',
    output: 'build_options.hpp',
    configuration: conf_data
)

# DEPENDENCIES
fs      = import('fs')
pymod   = import('python')

py3     = pymod.find_installation('python3', required: true)
py3_dep = py3.dependency()
omp     = dependency('openmp')
hdf5    = dependency('hdf5', language: 'cpp', required: true)
message('python path -- ' + py3.path())
message('python install_dir -- ' + py3.get_install_dir())

#################################################################
# SOME BOOK KEEPING FOR ALL INSTALLS
##################################################################
incdir_numpy = run_command(py3,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()
all_inc       = include_directories(incdir_numpy, 'src')
linker_args   = []
cpp_comp_args = ['-std=c++17', '-march=native', '-O3', '-fopenmp', '-flto=auto']
depends       = [py3_dep, hdf5, omp]
gpu_depends   = []
gpu_linkers   = []
gpu_comp_args = []
##################################################################
# GPU CHECK
##################################################################
if get_option('gpu_compilation').enabled()
    hip  = dependency('HIP', cmake_module_path : '/opt/rocm', required: false)
    cuda = dependency('CUDA', required: false)
    host_compiler    = meson.get_compiler('cpp').get_id()
    if hip.found()
        env = environment()
        check_hip_config = run_command('/opt/rocm/hip/bin/hipconfig','--platform', check: true)
        hip_platform     = check_hip_config.stdout().strip()
        hip_platform_err = check_hip_config.stderr().strip()
        gpu_arch = 70
        if hip_platform == 'nvidia'
            hip_runtime  = 'cuda'
            hip_compiler = 'nvcc'
            gpu_depends += [cuda]
            gpu_linkers += ['-lcudart']
            gpu_comp_args = [
                '-std=c++17',
                f'-arch=sm_@gpu_arch@', 
                f'-ccbin=@host_compiler@',
                '--ptxas-options=-v', 
                '--extended-lambda',
                '--compiler-options', 
                '-fPIC', '-O3', '-x=cu'
            ]
        elif hip_platform == 'amd' 
            hip_runtime = 'rocm'
            hip_compiler = 'hipcc'
            gpu_depends += [hip]
            gpu_linkers += ['-lamdhip64']
        endif
        env.set('HIP_PLATFORM', hip_platform)
        env.set('HIP_RUNTIME',  hip_runtime)
        env.set('HIP_COMPILER', hip_compiler)
        message(f'GPU Platform -- @hip_platform@')
        message(f'GPU Runtime  -- @hip_runtime@')
        message(f'GPU Compiler -- @hip_compiler@')
    elif cuda.found()
        gpu_depends += [cuda]
        gpu_linkers += ['-lcudart']
        gpu_comp_args = [
                '-std=c++17',
                # '-arch=sm_${CUDA_SM}', 
                f'-ccbin=@host_compiler@',
                '--ptxas-options=-v', 
                '--extended-lambda',
                '--compiler-options', 
                '-fPIC', '-O3', '-x=cu'
        ]
        message(f'GPU Platform -- nvidia')
        message(f'GPU Runtime  -- cuda')
        message(f'GPU Compiler -- nvcc')
    endif
endif

sources = [
    #######################
    ### Common IMPLEMENTATIONS
    #######################
    'src/common/clattice1D.cpp',
    'src/common/clattice2D.cpp',
    'src/common/clattice3D.cpp',
    'src/common/helpers.cpp',
    'src/common/hydro_structs.cpp',
    # 'src/common/viscous_diff.cpp'
    # 'src/common/viscous_diff.hpp'

    #######################
    #### GPU / CPU IMPLEMENTATIONS
    #######################
    'src/hydro/euler1D.cpp',
    'src/hydro/euler2D.cpp',
    'src/hydro/srhydro1D.hip.cpp',
    'src/hydro/srhydro2D.hip.cpp',
    'src/hydro/srhydro3D.hip.cpp',
    'src/hydro/helpers.hip.cpp',

    #######################
    #### UTIL IMPLEMENTATIONS
    #######################
    'src/util/printb.cpp',
    'src/util/device_api.cpp',
]

src_hdrs = [
    #######################
    ### Common DEPENDENCIES
    #######################
    'src/common/enums.hpp',
    'src/common/clattice1D.hpp',
    'src/common/clattice2D.hpp',
    'src/common/clattice3D.hpp',
    'src/common/helpers.hpp',
    'src/common/helpers.tpp',
    'src/common/hydro_structs.hpp',
    'src/common/traits.hpp',
    #######################
    # GPU / CPU DEPENDCIES
    #######################
    'src/hydro/euler1D.hpp',
    'src/hydro/euler2D.hpp',
    'src/hydro/srhydro1D.hip.hpp',
    'src/hydro/srhydro1D.hip.hpp',
    'src/hydro/srhydro2D.hip.hpp',
    'src/hydro/srhydro3D.hip.hpp',
    'src/hydro/helpers.hip.hpp',
    'src/hydro/srhydro3D.hip.hpp',
    'src/hydro/helpers.hip.hpp',
    'src/hydro/helpers.hip.tpp',
    'src/hydro/srhydro2D.hip.hpp',
    'src/hydro/euler2D.hpp',
    'src/hydro/euler1D.hpp',
    
    #######################
    # UTIL DEPENDENCIES
    #######################
    'src/util/exec_policy.hpp',
    'src/util/kernel.hpp',
    'src/util/launch.hpp',
    'src/util/launch.tpp',
    'src/util/printb.hpp',
    'src/util/dual.hpp',
    'src/util/dual.tpp',
    'src/util/printb.tpp',
    'src/util/parallel_for.hpp',
    'src/util/device_api.hpp',
]

run_command('python', '-m', 'cython', '--cplus', 'src/call_obj.pyx', check: true)
cpu_ext = py3.extension_module(
    'cpu_ext',
    ['src/cpu_ext.pyx', 'src/call_obj.pyx'] + sources,
    include_directories: all_inc,
    cpp_args: cpp_comp_args,
    dependencies: depends,
    override_options: ['cython_language=cpp'],
    install: true,
)

if get_option('gpu_compilation').enabled()
    if hip.found() or cuda.found()
        gpu_cc =  hip.found() ? 'hipcc' : 'nvcc'
        gpu_compiler = find_program(f'@gpu_cc@')
        gpu_objs = []
        gpu_libs = []
        gpu_includes = ['-I../src', '-I/', '-I'+meson.current_build_dir()]
        gpu_libs = []
        gpu_includes += ['-I/usr/include/hdf5/serial', '-I/usr/lib/cuda-11.2/include/']
        # hdf5_lib = hdf5.get_variable(cmake: 'HDF5_CXX_LIBRARIES')
        # This won't work
        # foreach dep : depends
        #     gpu_includes += ['-I' + dep.include_dir()]
        #     gpu_libs += ['-L' + dep.library_dir()]
        # endforeach
        foreach source : sources
            gpu_trg = custom_target(
                fs.name(source) + '_target',
                command : [
                    gpu_compiler,
                    gpu_includes,
                    gpu_libs,
                    gpu_comp_args,
                    '-c',
                    '@INPUT@',
                    '-o',
                    '@OUTPUT@',
                ],
                input : source,
                output : '@BASENAME@.o',
                build_by_default: true,
            )
            gpu_objs += [gpu_trg]
        endforeach

        gpu_lib = custom_target(
            'gpu_library',
            command: [
                'ar',
                'rcs',
                '@OUTPUT@',
                '@INPUT@',
            ],
            input: gpu_objs,
            output: 'libsimbi_gpu.a',
            install: false, 
            build_by_default: true,
        )
        # gpu_cc =  hip.found() ? 'hipcc' : 'nvcc'
        # gpu_compiler = find_program(f'@gpu_cc@')
        # gpu_lib = custom_target(
        #     'libsimbi_gpu',
        #     output: 'libsimbi_gpu.a',
        #     input: sources,
        #     command: [gpu_compiler, '@INPUT@', '@OUTPUT@'],
        #     depend_files: sources,
        #     install: false,
        #     build_by_default: true,
        # )
        # gen = generator(gpu_compiler,
        #         output  : '@BASENAME@.o',
        #         arguments : ['-c', '@INPUT@', '-o', '@OUTPUT@', '--out_dir=@BUILD_DIR@'])
        # gen_objs = gen.process(sources)
        # gpu_lib = static_library(
        #     'simbi_gpu',
        #     # objects: gen_objs,
        #     sources,
        #     include_directories: all_inc,
        #     native: false,
        #     link_args: linker_args + gpu_linkers,
        #     dependencies: depends + gpu_depends,
        # )
        gpu_ext = py3.extension_module(
            'gpu_ext',
            ['src/gpu_ext.pyx', 'src/call_obj.pyx'],
            link_with: [gpu_lib],
            include_directories: all_inc,
            link_args: linker_args + gpu_linkers,
            dependencies: depends + gpu_depends,
            override_options: ['cython_language=cpp'],
            install: true,
        )
    endif
endif

install_subdir(
    'pysimbi',
    install_dir: py3.get_install_dir()
)