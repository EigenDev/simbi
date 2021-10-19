cmake_minimum_required(VERSION 3.1)
project(pysimbi VERSION 0.2.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GPU_CODE "Turns gpu compilation on/off" ON)
option(FLOAT_PRECISION "Turns float precision on/off" OFF)
set(BUILD_OPTS_IN  "${CMAKE_CURRENT_SOURCE_DIR}/build_options.hpp.in")
set(BUILD_OPTS     "${CMAKE_CURRENT_SOURCE_DIR}/src/build_options.hpp")
configure_file(${BUILD_OPTS_IN} ${BUILD_OPTS})
# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(PYTHON3 "python3")
find_package(HDF5 REQUIRED COMPONENTS CXX HL)
find_package(HIP REQUIRED)

# SET (HDF5_LIBS ${HDF5_LIBRARIES} "${HDF5_CPP_LIBRARIES}/../")
# message(STATUS "HDF5 Lib Path: ${HDF5_CXX_LIBRARIES}")
# message(STATUS "HDF5 Lib Path: ${HDF5_LIBRARIES}")

# Determine HIP_PLATFORM
if(NOT DEFINED HIP_PLATFORM)
    if(NOT DEFINED ENV{HIP_PLATFORM})
        execute_process(COMMAND /opt/rocm/hip/bin/hipconfig --platform
            OUTPUT_VARIABLE HIP_PLATFORM
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
        set(HIP_PLATFORM $ENV{HIP_PLATFORM} CACHE STRING "HIP Platform")
    endif()
endif()
message(STATUS "HIP Platform: " ${HIP_PLATFORM})

# Detect the hipcc compiler path
# execute_process(COMMAND which hipcc OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE HIPCC)
set(HIPCC "/opt/rocm/bin/hipcc")
if(HIP_PLATFORM STREQUAL "nvidia")
    set(HIP_RUNTIME "cuda"  CACHE STRING "HIP Runtime")
    set(HIP_COMPILER "nvcc" CACHE STRING "HIP Compiler")
    set(HIP_RUNTIME_LIBRARY_DIRS "/opt/rocm/hip/lib")
    find_package(CUDA)
    enable_language(CUDA)
    set(CUDART_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    list(APPEND  HIP_RUNTIME_LIBRARY_DIRS ${CUDART_LIB_PATH})
    list(APPEND HIP_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    message(STATUS "CUDA_TOOLKIT_PATH: ${CUDA_TOOLKIT_ROOT_DIR}")
elseif(HIP_PLATFORM STREQUAL "amd")
    set(HIP_RUNTIME "rocclr" CACHE STRING "HIP Runtime")
    set(HIP_COMPILER "clang" CACHE STRING "HIP Compiler")
    set(HIP_RUNTIME_LIBRARY_DIRS "/opt/rocm/hip/lib")
else()
    message(FATAL_ERROR "Unexpected HIP_PLATFORM: " ${HIP_PLATFORM})
endif()

message(STATUS "HIP Runtime: "  ${HIP_RUNTIME})
message(STATUS "HIP Compiler: " ${HIP_COMPILER})
message(STATUS "HIP LIBS: " ${HIP_LIBRARIES})

# COPY_IF_DIFFERENT("${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
    set(_pip_args)
else()
    set(_pip_args "--user")
endif()

set(SRCS 
    ${SRCS}
    #######################
    ### Common dependencies
    #######################
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/clattice1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/clattice1D.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/clattice2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/clattice2D.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/clattice3D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/clattice3D.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/helpers.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/hydro_structs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/hydro_structs.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/common/traits.hpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/common/viscous_diff.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/src/common/viscous_diff.hpp

    #######################
    #### GPU Dependencies
    #######################
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/euler1D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/euler1D.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/euler2D.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/euler2D.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/srhydro1D.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/srhydro1D.hip.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/srhydro2D.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/srhydro2D.hip.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/srhydro3D.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/srhydro3D.hip.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/helpers.hip.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/helpers.hip.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/helpers.hip.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/gpu_state.pyx

    #######################
    #### UTIL Dependencies
    #######################
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/exec_policy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/kernel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/launch.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/launch.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/printb.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/printb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/printb.tpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/parallel_for.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/device_api.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/device_api.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/dual.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util/dual.tpp
)

if(PYTHON3)
    set(SETUP_PY_IN  "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
    set(SETUP_PY     "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
    set(DEPS         "${CMAKE_CURRENT_SOURCE_DIR}/pysimbi/__init__.py")
    set(OUTPUT       "${CMAKE_CURRENT_BINARY_DIR}/build")

    list(APPEND DEPS ${SRCS})
    configure_file(${SETUP_PY_IN} ${SETUP_PY})
    # Custom make command for building default cpu instance in place
    add_custom_command(
        OUTPUT  ${OUTPUT}/pytimestamp_gpu
        COMMAND ${PYTHON3} setup.py build_ext --inplace && rm -f 
                                    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/gpu_state.cpp && rm -Rf ${CMAKE_CURRENT_SOURCE_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/pytimestamp_gpu
        DEPENDS ${DEPS}
    )
    add_custom_target(gpu ALL DEPENDS ${OUTPUT}/pytimestamp_gpu)

    add_custom_command(
        OUTPUT  ${OUTPUT}/pytimestamp_cpu
        COMMAND ${PYTHON3} setup.py build_ext --inplace --cpu && rm -f 
                                    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro/gpu_state.cpp && rm -Rf ${CMAKE_CURRENT_SOURCE_DIR}/build
        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/pytimestamp_cpu
        DEPENDS ${DEPS}
    )
    add_custom_target(cpu DEPENDS ${OUTPUT}/pytimestamp_cpu)
    install(CODE "execute_process(COMMAND pip install ${_pip_args} lib/)")
endif(PYTHON3)
