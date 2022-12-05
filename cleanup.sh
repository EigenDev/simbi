#!/bin/bash
SIMBI_DIR="$( cd "$( dirname "$0" )" && pwd )"

BUILD_DIR="${SIMBI_DIR}/build"
EGG_DIR="${SIMBI_DIR}/pysimbi.egg-info"
if [ -d "${EGG_DIR}" ]; then
  while true; do
    read -p "Do you wish to delete the ${EGG_DIR} directory (it is safe to do so)? " yn
    case $yn in
        [Yy]* ) rm -rf ${EGG_DIR}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done
fi

if [ -d "${BUILD_DIR}" ]; then
  while true; do
    read -p "Do you wish to delete the ${BUILD_DIR} directory (it is safe to do so)? " yn
    case $yn in
        [Yy]* ) rm -rf ${BUILD_DIR}; break;;
        [Nn]* ) break;;
        * ) echo "Please answer yes or no.";;
    esac
done
fi