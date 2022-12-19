#!/usr/bin/env bash
YELLOW='\033[0;33m'
RST='\033[0m' # No Color
EXECUTED=false
ERROR_CODE=0
if [ "${BASH_SOURCE[0]}" -ef "$0" ]
then
    printf "${YELLOW}WRN${RST}: Running script as executable. Be aware that none of your build configurations will be cached!\n"
    printf "If you want to cached the build configurations, please use the 'source' or '.' commands.\n"
    read -p "press [Enter] to continue..."
    EXECUTED=true
fi

function usage {
    echo "usage: CXX=<cpp_compiler> $0 [options]"
    echo ""
    echo "   -h | --help                   print help message"
    echo "  --oned                         block size for 1D simulations (only set for gpu compilation)"
    echo "  --twod                         block size for 2D simulations (only set for gpu compilation)"
    echo "  --threed                       block size for 3D simulations (only set for gpu compilation)"
    echo "  --dev-arch                     SM architecture specification for gpu compilation"
    echo "  -v | --verbose                 flag for verbose compilation output"
    echo "  --configure                    flag to only configure the meson build directory without installing"
    echo "  --[float | double]-precision   floating point precision"
    echo "  --[column | row]-major         memory layout for multi-dimensional arrays"
    echo "  --[gpu | cpu]-compilation      compilation mode"
    echo "  --[default | develop]          install mode (normal or editable)"
    ERROR_CODE=1
}

params="$(getopt -o hv -l oned:,twod:,threed:,dev-arch:,help,verbose,\
float-precision,double-precision,column-major,row-major,gpu-compilation,\
cpu-compilation,develop,default,configure --name "$(basename "$0")" -- "$@")"
if [ $? -ne 0 ]
then
    usage
fi
eval set -- "$params"
unset params

SIMBI_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if command -v nvcc &> /dev/null; then 
GPU_RUNTIME_DIR="$(echo $PATH | sed "s/:/\n/g" | grep "cuda/bin" | sed "s/\/bin//g" |  head -n 1)"
elif command -v hipcc &> /dev/null; then
GPU_RUNTIME_DIR=("$(hipconfig --rocmpath)")
fi 
GPU_INCLUDE="${GPU_RUNTIME_DIR}/include"
HDF5_PATH="$(echo $(command -v h5cc) | sed "s/:/\n/g" | grep "bin/h5cc" | sed "s/\/bin//g;s/\/h5cc//g" |  head -n 1)"
HDF5_INCLUDE="$( dirname $(find ${HDF5_PATH} -iname "H5Cpp.h" -type f 2>/dev/null -print -quit ) )"

if test -z "${SIMBI_GPU_COMPILATION}"; then 
export SIMBI_GPU_COMPILATION="disabled"
fi
if test -z "${SIMBI_ONE_BLOCK_SIZE}"; then
export SIMBI_ONE_BLOCK_SIZE=128
fi 
if test -z "${SIMBI_TWOD_BLOCK_SIZE}"; then 
export SIMBI_TWOD_BLOCK_SIZE=16
fi 
if test -z "${SIMBI_THREED_BLOCK_SIZE}"; then 
export SIMBI_THREED_BLOCK_SIZE=4
fi 
if test -z "${SIMBI_COLUMN_MAJOR}"; then 
export SIMBI_COLUMN_MAJOR=false 
fi 
if test -z "${SIMBI_FLOAT_PRECISION}"; then 
export SIMBI_FLOAT_PRECISION=false 
fi 
if test -z "${SIMBI_PROFILE}"; then 
export SIMBI_PROFILE=default
fi 
if test -z "${SIMBI_GPU_ARCH}"; then 
export SIMBI_GPU_ARCH=86
fi 
if test -z "${SIMBI_BUILDDIR}"; then 
export SIMBI_BUILDDIR=builddir
fi

configure_only=false
reconfigure=
if command meson introspect ${SIMBI_DIR}/${SIMBI_BUILDDIR} -i --targets &> /dev/null; then 
reconfigure="--reconfigure"
fi 

verbose=
while true
do
    case $1 in
        --gpu-compilation)
            [ "${--gpu-compilation}" = 2 ]
            export SIMBI_GPU_COMPILATION="enabled"
            shift
            ;;
        --cpu-compilation)
            [ "${--cpu-compilation}" = 2 ]
            export SIMBI_GPU_COMPILATION="disabled"
            shift
            ;;
        --float-precision)
            [ "${--double-precision}" = 2 ]
            export SIMBI_FLOAT_PRECISION=true
            shift
            ;;
        --double=precision)
            [ "${--float-precision}" = 2 ]
            export SIMBI_FLOAT_PRECISION=false
            shift
            ;;
        --column-major)
            [ "${--row-major}" = 2 ]
            export SIMBI_COLUMN_MAJOR=true
            shift
            ;;
        --row-major)
          [ "${--column-major}" = 2 ]
            export SIMBI_COLUMN_MAJOR=false
            shift
            ;;
        --develop)
            [ "${--default}" = 2 ]
            export SIMBI_PROFILE="develop"
            shift
            ;;
        --default)
          [ "${--develop}" = 2 ]
          export SIMBI_PROFILE="default"
          shift
          ;;
        --dev-arch)
            export SIMBI_GPU_ARCH=("$2")
            shift 2
            ;;
        --oned)
            export SIMBI_ONE_BLOCK_SIZE=("$2")
            shift 2
            ;;
        --twod)
            export SIMBI_TWOD_BLOCK_SIZE=("$2")
            shift 2
            ;;
        --threed)
            export SIMBI_THREED_BLOCK_SIZE=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            break
            ;;
        -v|--verbose)
            verbose='--verbose'
            shift
            ;;
        --configure)
            configure_only=true
            shift 
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            break
            ;;
    esac
done



if [ ${ERROR_CODE} = 0 ]; then
    if test -z "${CXX}"; then 
        CXX="$(echo $(command -v c++) )" 
        printf "${YELLOW}WRN${RST}: C++ compiler not set.\n"
        printf "Using symbolic link ${CXX} as default.\n"
    fi 

    function configure()
    {
    CXX=${CXX} meson setup ${SIMBI_BUILDDIR} -Dgpu_compilation=${SIMBI_GPU_COMPILATION} -Dhdf5_include_dir=${HDF5_INCLUDE} -Dgpu_include_dir=${GPU_INCLUDE} \
    -D1d_block_size=${SIMBI_ONE_BLOCK_SIZE} -D2d_block_size=${SIMBI_TWOD_BLOCK_SIZE} -D3d_block_size=${SIMBI_THREED_BLOCK_SIZE} \
    -Dcolumn_major=${SIMBI_COLUMN_MAJOR} -Dfloat_precision=${SIMBI_FLOAT_PRECISION} \
    -Dprofile=${SIMBI_PROFILE} -Dgpu_arch=${SIMBI_GPU_ARCH} ${reconfigure}
    }

    function install_simbi()
    {
    ( cd ${SIMBI_DIR}/${SIMBI_BUILDDIR} && meson compile ${verbose} && meson install )
    }

    function cleanup()
    {
    (cd ${SIMBI_DIR} && ./cleanup.sh)
    }

    if ! command -v meson &> /dev/null; then
        pip3 install meson
    fi 
    configure
    if [ ${configure_only} = false ]; then
    install_simbi
    cleanup
    fi
fi