#!/bin/bash

set -e

BUILD_PATH=$(pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

BUILD_ARGS=$(getopt -o ha:v:m:j:ft --long \
    help,release,debug,arch:,python-version:,include-mod:,CANN-path:,jobs:,force-rebuild,local,test-cases -- "$@")
eval set -- "${BUILD_ARGS}"

ARCH_TYPE=$(uname -m)
BUILD_TYPE=release
CANN_PATH=""
CONCURRENT_JOBS=16
BUILD_TEST_CASE=False
USE_LOCAL_FIRST=False
PYTHON_VERSION=""
INCLUDE_MOD=""
ADUMP_MOD="'adump'"
ATB_PROBE_MOD="'atb_probe'"
ACLGRAPH_DUMP_MOD="'aclgraph_dump'"
NAN_CHECK_MOD="'nan_check'"
PYTHON_NAN_CHECK_VENDOR_DIR="${BUILD_PATH}/python/msprobe/vendors"
XOR_CHECKSUM_MOD="'xor_checksum'"

HELP_DOC=$(cat << EOF
Usage: build.sh [OPTION]...\n
Build the C++ part of MsProbe.\n
\n
Arguments:\n
    -a, --arch                    Specify the schema, which generally does not need to be set up.\n
        --CANN-path               Specify the CANN path. When set, the build script will find the dependent files in\n
                                  the specified path.\n
    -j, --jobs                    Specify the number of compilation jobs(default 16).\n
    -f, --force-rebuild           Clean up the cache before building.\n
    -t, --test-cases              Build test cases.\n
        --local                   Prioritize the use of on-premises, third-party resources as dependencies.\n
        --release                 Build the release version(default).\n
        --debug                   Build the debug version.
    -v, --python-version          Specify version of python.
    -m, --include-mod             Specify the modules which need to be built.
EOF
)

detect_xor_checksum_npu_arch() {
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "dav-2201"
        return 0
    fi

    local smi_out
    smi_out=$(npu-smi info 2>/dev/null || true)

    if echo "${smi_out}" | grep -Eq 'Ascend950|(^|[^[:digit:]])950([[:alnum:]_-]*)([^[:alnum:]_]|$)'; then
        echo "dav-3510"
        return 0
    fi

    if echo "${smi_out}" | grep -Eq 'Ascend910B|(^|[^[:digit:]])910B([[:alnum:]_-]*)([^[:alnum:]_]|$)|Ascend910([^[:alnum:]_]|$)|(^|[^[:digit:]])910([^[:alnum:]_]|$)'; then
        echo "dav-2201"
        return 0
    fi

    echo "dav-2201"
}

while true; do
    case "$1" in
        -h | --help)
            echo -e ${HELP_DOC}
            exit 0 ;;
        -a | --arch)
            ARCH_TYPE="$2" ; shift 2 ;;
        -v | --python-version)
            PYTHON_VERSION="$2" ; shift 2 ;;
        -m | --include-mod)
            INCLUDE_MOD="$2" ; shift 2 ;;
        --release)
            BUILD_TYPE=release ; shift ;;
        --debug)
            BUILD_TYPE=debug ; shift ;;
        --CANN-path)
            CANN_PATH="$2" ; shift 2 ;;
        -j | --jobs)
            CONCURRENT_JOBS="$2"  ; shift 2 ;;
        --local)
            USE_LOCAL_FIRST=True ; shift ;;
        -f | --force-rebuild)
            rm -rf "${BUILD_PATH}/build_dependency" "${BUILD_PATH}/lib" "${BUILD_PATH}/output" "${BUILD_PATH}/third_party" \
                   "${BUILD_PATH}/python/msprobe/lib/*" "${BUILD_PATH}/ccsrc/xor_checksum/build" \
                   "${BUILD_PATH}/ccsrc/xor_checksum/dist" "${BUILD_PATH}/ccsrc/xor_checksum"/*.egg-info \
                   "${BUILD_PATH}/python/msprobe/lib/xor_checksum_ext.so"
            shift ;;
        -t | --test-cases)
            BUILD_TEST_CASE=True ; shift ;;
        --)
            shift ; break ;;
        *)
            echo "Unknown argument $1"
            exit 1 ;;
    esac
done

BUILD_OUTPUT_PATH=${BUILD_PATH}/output/${BUILD_TYPE}

if [[ "${INCLUDE_MOD}" == *"${ADUMP_MOD}"* ]]; then
    export MSPROBE_INCLUDE_MOD="adump"
    cmake -B ${BUILD_OUTPUT_PATH} -S . -DARCH_TYPE=${ARCH_TYPE} -DBUILD_TYPE=${BUILD_TYPE} -DCANN_PATH=${CANN_PATH} \
                                  -DUSE_LOCAL_FIRST=${USE_LOCAL_FIRST} -DBUILD_TEST_CASE=${BUILD_TEST_CASE} \
                                  -DPYTHON_VERSION=${PYTHON_VERSION}
    cd ${BUILD_OUTPUT_PATH}
    make -j${CONCURRENT_JOBS}

    if [[ ! -e ${BUILD_OUTPUT_PATH}/ccsrc/adump/lib_msprobe_c.so ]]; then
        echo "Failed to build lib_msprobe_c.so."
        exit 1
    fi
fi

if [[ "${INCLUDE_MOD}" == *"${ATB_PROBE_MOD}"* ]]; then
    export MSPROBE_INCLUDE_MOD="atb_probe"
    cd ${BUILD_PATH}
    cmake -B ${BUILD_OUTPUT_PATH} -S . -DARCH_TYPE=${ARCH_TYPE} -DBUILD_TYPE=${BUILD_TYPE} -DCANN_PATH=${CANN_PATH} \
                                  -DUSE_LOCAL_FIRST=${USE_LOCAL_FIRST} -DBUILD_TEST_CASE=${BUILD_TEST_CASE} \
                                  -DPYTHON_VERSION=${PYTHON_VERSION}
    cd ${BUILD_OUTPUT_PATH}
    make -j${CONCURRENT_JOBS}

    if [[ ! -e ${BUILD_OUTPUT_PATH}/ccsrc/atb_probe/libatb_probe_abi0.so ]]; then
        echo "Failed to build libatb_probe_abi0.so."
        exit 1
    fi

    export ATB_PROBE_ABI="1"
    cd ${BUILD_PATH}
    cmake -B ${BUILD_OUTPUT_PATH} -S . -DARCH_TYPE=${ARCH_TYPE} -DBUILD_TYPE=${BUILD_TYPE} -DCANN_PATH=${CANN_PATH} \
                                  -DUSE_LOCAL_FIRST=${USE_LOCAL_FIRST} -DBUILD_TEST_CASE=${BUILD_TEST_CASE} \
                                  -DPYTHON_VERSION=${PYTHON_VERSION}
    cd ${BUILD_OUTPUT_PATH}
    make -j${CONCURRENT_JOBS}

    if [[ ! -e ${BUILD_OUTPUT_PATH}/ccsrc/atb_probe/libatb_probe_abi1.so ]]; then
        echo "Failed to build libatb_probe_abi1.so."
        exit 1
    fi
fi

if [[ "${INCLUDE_MOD}" == *"${ACLGRAPH_DUMP_MOD}"* ]]; then
    export MSPROBE_INCLUDE_MOD="aclgraph_dump"
    cd ${BUILD_PATH}
    cmake -B ${BUILD_OUTPUT_PATH} -S . -DARCH_TYPE=${ARCH_TYPE} -DBUILD_TYPE=${BUILD_TYPE} -DCANN_PATH=${CANN_PATH} \
                                  -DUSE_LOCAL_FIRST=${USE_LOCAL_FIRST} -DBUILD_TEST_CASE=${BUILD_TEST_CASE} \
                                  -DPYTHON_VERSION=${PYTHON_VERSION}
    cd ${BUILD_OUTPUT_PATH}
    make -j${CONCURRENT_JOBS}

    if [[ ! -e ${BUILD_OUTPUT_PATH}/ccsrc/aclgraph_dump/aclgraph_dump_ext.so ]]; then
        echo "Failed to build aclgraph_dump_ext.so."
        exit 1
    fi
fi

if [[ "${INCLUDE_MOD}" == *"${NAN_CHECK_MOD}"* ]]; then
    cd ${BUILD_PATH}
    bash ${BUILD_PATH}/ccsrc/nan_check/build_nan_test.sh
    export MSPROBE_INCLUDE_MOD="nan_check"
    cd ${BUILD_PATH}
    cmake -B ${BUILD_OUTPUT_PATH} -S . -DARCH_TYPE=${ARCH_TYPE} -DBUILD_TYPE=${BUILD_TYPE} -DCANN_PATH=${CANN_PATH} \
                                  -DUSE_LOCAL_FIRST=${USE_LOCAL_FIRST} -DBUILD_TEST_CASE=${BUILD_TEST_CASE} \
                                  -DPYTHON_VERSION=${PYTHON_VERSION}
    cd ${BUILD_OUTPUT_PATH}
    make -j${CONCURRENT_JOBS}

    if [[ ! -e ${BUILD_OUTPUT_PATH}/ccsrc/nan_check/nan_check_ext.so ]]; then
        echo "Failed to build nan_check_ext.so."
        exit 1
    fi
fi

if [[ "${INCLUDE_MOD}" == *"${XOR_CHECKSUM_MOD}"* ]]; then
    export MSPROBE_INCLUDE_MOD="xor_checksum"
    XOR_CHECKSUM_OUTPUT_PATH=${BUILD_OUTPUT_PATH}/xor_checksum
    XOR_CHECKSUM_NPU_ARCH=$(detect_xor_checksum_npu_arch)
    echo "[xor_checksum_build] Using NPU_ARCH=${XOR_CHECKSUM_NPU_ARCH}"
    cd ${BUILD_PATH}
    cmake -B ${XOR_CHECKSUM_OUTPUT_PATH} -S ${BUILD_PATH}/ccsrc/xor_checksum \
          -DPython3_EXECUTABLE=${PYTHON_BIN} \
          -DNPU_ARCH=${XOR_CHECKSUM_NPU_ARCH}
    cd ${XOR_CHECKSUM_OUTPUT_PATH}
    make -j${CONCURRENT_JOBS}

    if [[ ! -e ${XOR_CHECKSUM_OUTPUT_PATH}/xor_checksum_ext.so ]]; then
        echo "Failed to build xor_checksum_ext.so."
        exit 1
    fi
fi

if [ ! -d ${BUILD_PATH}/python/msprobe/lib ]; then
    mkdir ${BUILD_PATH}/python/msprobe/lib
fi

if [[ "${INCLUDE_MOD}" == *"${ADUMP_MOD}"* ]]; then
    cp -f ${BUILD_OUTPUT_PATH}/ccsrc/adump/lib_msprobe_c.so ${BUILD_PATH}/python/msprobe/lib/_msprobe_c.so
fi

if [[ "${INCLUDE_MOD}" == *"${ATB_PROBE_MOD}"* ]]; then
    cp -f ${BUILD_OUTPUT_PATH}/ccsrc/atb_probe/libatb_probe_abi0.so ${BUILD_PATH}/python/msprobe/lib/libatb_probe_abi0.so
    cp -f ${BUILD_OUTPUT_PATH}/ccsrc/atb_probe/libatb_probe_abi1.so ${BUILD_PATH}/python/msprobe/lib/libatb_probe_abi1.so
fi

if [[ "${INCLUDE_MOD}" == *"${ACLGRAPH_DUMP_MOD}"* ]]; then
    cp -f ${BUILD_OUTPUT_PATH}/ccsrc/aclgraph_dump/aclgraph_dump_ext.so ${BUILD_PATH}/python/msprobe/lib/aclgraph_dump_ext.so
fi

if [[ "${INCLUDE_MOD}" == *"${NAN_CHECK_MOD}"* ]]; then
    cp -f ${BUILD_OUTPUT_PATH}/ccsrc/nan_check/nan_check_ext.so ${BUILD_PATH}/python/msprobe/lib/nan_check_ext.so
    rm -rf "${PYTHON_NAN_CHECK_VENDOR_DIR}"
    if [[ -d "${BUILD_PATH}/vendors" ]]; then
        cp -a "${BUILD_PATH}/vendors" "${PYTHON_NAN_CHECK_VENDOR_DIR}"
    fi
    if [[ ! -f "${PYTHON_NAN_CHECK_VENDOR_DIR}/customize/op_api/lib/libcust_opapi.so" ]]; then
        echo "Failed to prepare nan_check runtime assets under ${PYTHON_NAN_CHECK_VENDOR_DIR}." >&2
        exit 1
    fi
elif [[ -d "${PYTHON_NAN_CHECK_VENDOR_DIR}" ]]; then
    rm -rf "${PYTHON_NAN_CHECK_VENDOR_DIR}"
fi

if [[ "${INCLUDE_MOD}" == *"${XOR_CHECKSUM_MOD}"* ]]; then
    cp -f ${BUILD_OUTPUT_PATH}/xor_checksum/xor_checksum_ext.so ${BUILD_PATH}/python/msprobe/lib/xor_checksum_ext.so
fi
