#!/usr/bin/env bash
SIMBI_DIR="$( cd "$( dirname "$0" )" && pwd )"

BUILD_DIR="${SIMBI_DIR}/build"
EGG_DIR="${SIMBI_DIR}/simbi.egg-info"
if [ -d "${EGG_DIR}" ]; then
  rm -rf ${EGG_DIR}
fi

if [ -d "${BUILD_DIR}" ]; then
  rm -rf ${BUILD_DIR}
fi