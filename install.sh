#!/bin/bash
params="$(getopt -o e:hv -l oned:,twod:,threed:,gpu_arch,help,verbose,float-precision,double-precision,column-major,row-major,gpu-compilation,cpu-compilation,develop,default --name "$(basename "$0")" -- "$@")"
SIMBI_DIR="$( cd "$( dirname "$0" )" && pwd )"

usage()  
{  
echo "Usage: $0 [options]"  
exit 1  
} 



if which nvcc -v &> /dev/null; then 
gpu_runtime_dir="$(echo $PATH | sed "s/:/\n/g" | grep "cuda/bin" | sed "s/\/bin//g" |  head -n 1)"
elif which hipcc -v &> /dev/null; then
gpu_runtime_dir=("$(hipconfig --rocmpath)")
fi 
if [ $? -ne 0 ]
then
    usage
fi
eval set -- "$params"
unset params

gpu_include="${gpu_runtime_dir}/include"
hdf5_header="$(echo '#include <H5Cpp.h>' | cpp -H -o /dev/null 2>&1 | head -n1 | sed "s/^.//;s/^.//")"
hdf5_include="$(dirname "${hdf5_header}")"
if test -z "${CXX}"; then 
echo "please provie a c++ compiler"
exit 1
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
        --gpu_arch)
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
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done

configure()
{
  if [ ${not_configured}=true ] || [ -z "${reconfigure}" ] ; then
  CXX=${CXX} meson setup ${SIMBI_BUILDDIR} -Dgpu_compilation=${SIMBI_GPU_COMPILATION} -Dhdf5_include_dir=${hdf5_include} -Dgpu_include_dir=${gpu_include} \
  -D1d_block_size=${SIMBI_ONE_BLOCK_SIZE} -D2d_block_size=${SIMBI_TWOD_BLOCK_SIZE} -D3d_block_size=${SIMBI_THREED_BLOCK_SIZE} \
  -Dcolumn_major=${SIMBI_COLUMN_MAJOR} -Dfloat_precision=${SIMBI_FLOAT_PRECISION} \
  -Dprofile=${SIMBI_PROFILE} -Dgpu_arch=${SIMBI_GPU_ARCH} ${reconfigure}
  fi
}

install_simbi()
{
  ( cd ${SIMBI_DIR}/${SIMBI_BUILDDIR} && meson compile ${verbose} && meson install )
}

cleanup()
{
  (cd ${SIMBI_DIR} && ./cleanup.sh)
}

configure
install_simbi
cleanup
