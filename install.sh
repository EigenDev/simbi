#!/usr/bin/env bash
function usage {
    echo "usage: CXX=<cpp_compiler> $0 [options]"
    echo ""
    echo "   -h --help                     print help message"
    echo "  --oned                         block size for 1D simulations (only set for gpu compilation)"
    echo "  --twod                         block size for 2D simulations (only set for gpu compilation)"
    echo "  --threed                       block size for 3D simulations (only set for gpu compilation)"
    echo "  --gpu-arch                     SM architecture specification for gpu compilation"
    echo "  --verbose                      flag for verbose compilation output"
    echo "  --[float | double]-precision   floating point precision"
    echo "  --[column | row]-major         memory layout for multi-dimensional arrays"
    echo "  --[gpu | cpu]-compilation      compilation mode"
    echo "  --[default | develop]          install mode (normal or editable)"
    exit 1
}

params="$(getopt -o hv -l oned:,twod:,threed:,gpu-arch,help,verbose,\
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
HDF5_HEADER="$(echo '#include <H5Cpp.h>' | cpp -H -o /dev/null 2>&1 | head -n1 | sed "s/^.//;s/^.//")"
if [ $? -ne 0 ]
then
HDF5_RUNTIME_DIR="$(echo $PATH | sed "s/:/\n/g" | grep "hdf5" | sed "s/\/bin//g" |  head -n 1)"
HDF5_INCLUDE="${HDF5_RUNTIME_DIR}/include"
else
HDF5_INCLUDE="$(dirname "${HDF5_HEADER}")"
fi


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
export SIMBI_THREED_BLOCK_SIZE=8
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
not_configured=true
reconfigure=""
if [ -d "${SIMBI_DIR}/${SIMBI_BUILDDIR}/meson-logs" ]; then 
not_configured=false
reconfigure="--reconfigure"
fi 
verbose=""
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
        --gpu-arch)
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
            ;;
    esac
done

function configure()
{
  RED='\033[0;31m'
  RST='\033[0m' # No Color
  if test -z "${CXX}"; then 
    printf "${RED}ERROR${RST}: please provie a c++ compiler\n"
    usage
  fi 
  if [ ${not_configured} = true ] || [ -z "${reconfigure}" ] ; then
  CXX=${CXX} meson setup ${SIMBI_BUILDDIR} -Dgpu_compilation=${SIMBI_GPU_COMPILATION} -Dhdf5_include_dir=${HDF5_INCLUDE} -Dgpu_include_dir=${GPU_INCLUDE} \
  -D1d_block_size=${SIMBI_ONE_BLOCK_SIZE} -D2d_block_size=${SIMBI_TWOD_BLOCK_SIZE} -D3d_block_size=${SIMBI_THREED_BLOCK_SIZE} \
  -Dcolumn_major=${SIMBI_COLUMN_MAJOR} -Dfloat_precision=${SIMBI_FLOAT_PRECISION} \
  -Dprofile=${SIMBI_PROFILE} -Dgpu_arch=${SIMBI_GPU_ARCH} ${reconfigure}
  fi
}

function install_simbi()
{
  ( cd ${SIMBI_DIR}/${SIMBI_BUILDDIR} && meson compile ${verbose} && meson install )
}

function cleanup()
{
  (cd ${SIMBI_DIR} && ./cleanup.sh)
}

configure
if [ ${configure_only} = false ]; then
install_simbi
cleanup
fi