#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAN_CHECK_DIR="${SCRIPT_DIR}"
SRC_TEMPLATE_DIR="${NAN_CHECK_DIR}/nan_test"
JSON_PATH="${NAN_CHECK_DIR}/nan_test.json"
GEN_DIR="${NAN_CHECK_DIR}/NanTest"
SAFE_GEN_BASE="${NAN_CHECK_DIR}/.msopgen_build"
FRAMEWORK="${MSOPGEN_FRAMEWORK:-pytorch}"
TARGET_SOC_VERSION=""

ORIG_NAN_CHECK_MODE=""

restore_nan_check_mode() {
    if [[ -n "${ORIG_NAN_CHECK_MODE}" ]]; then
        chmod "${ORIG_NAN_CHECK_MODE}" "${NAN_CHECK_DIR}" || true
    fi
}

secure_nan_check_dir() {
    ORIG_NAN_CHECK_MODE=$(stat -c '%a' "${NAN_CHECK_DIR}")
    # msopgen requires path not writable by group/others.
    chmod go-w "${NAN_CHECK_DIR}"
    trap restore_nan_check_mode EXIT
}

secure_json_file() {
    # msopgen rejects overly permissive input json permissions.
    chmod 640 "${JSON_PATH}"
}

detect_soc_version() {
    if [[ -n "${SOC_VERSION:-}" ]]; then
        case "${SOC_VERSION}" in
            Ascend950|Ascend950PR|Ascend950DT) echo "Ascend950" ;;
            *) echo "${SOC_VERSION}" ;;
        esac
        return 0
    fi

    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "Ascend910B1"
        return 0
    fi

    local smi_out
    smi_out=$(npu-smi info 2>/dev/null || true)

    if echo "${smi_out}" | grep -Eq '910[_-]?93|910[Cc]|Ascend910([^[:alnum:]_]|$)'; then
        echo "Ascend910_93"
        return 0
    fi

    if echo "${smi_out}" | grep -Eq 'Ascend950|[^[:digit:]]950([^[:digit:]]|$)'; then
        echo "Ascend950"
        return 0
    fi

    local tag
    tag=$(echo "${smi_out}" | grep -Eo '910[Bb][0-9]+' | head -n 1 || true)
    if [[ -z "${tag}" ]]; then
        tag=$(echo "${smi_out}" | grep -Eo '910[A-Za-z0-9]+' | head -n 1 || true)
    fi

    if [[ -n "${tag}" ]]; then
        tag=$(echo "${tag}" | tr '[:lower:]' '[:upper:]')
        echo "Ascend${tag}"
    else
        echo "Ascend910B1"
    fi
}

target_op_host_soc_config() {
    case "${1}" in
        Ascend910_93) echo "ascend910_93" ;;
        Ascend950) echo "ascend950" ;;
        Ascend910A) echo "ascend910a" ;;
        Ascend910B*) echo "ascend910b" ;;
        *) echo "ascend910b" ;;
    esac
}

log() {
    echo "[nan_test_build] $*"
}

ensure_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "Command not found: ${cmd}" >&2
        return 1
    fi
}

gen_project() {
    secure_nan_check_dir
    secure_json_file
    TARGET_SOC_VERSION=$(detect_soc_version)
    log "Using msopgen args: -f ${FRAMEWORK} -c ai_core-${TARGET_SOC_VERSION}"

    mkdir -p "${SAFE_GEN_BASE}"
    chmod 700 "${SAFE_GEN_BASE}"

    local tmp_out
    tmp_out=$(mktemp -d "${SAFE_GEN_BASE}/nan_test_gen.XXXXXX")

    local -a try_cmds=(
        "msopgen gen -i \"${JSON_PATH}\" -f \"${FRAMEWORK}\" -c ai_core-\"${TARGET_SOC_VERSION}\" -lan cpp -out \"${tmp_out}\""
        "msopgen gen -i \"${JSON_PATH}\" -f \"${FRAMEWORK}\" -c ai_core-\"${TARGET_SOC_VERSION}\" -out \"${tmp_out}\""
    )

    for raw_cmd in "${try_cmds[@]}"; do
        local cmd
        cmd=$(eval "echo ${raw_cmd}")
        log "Trying: ${cmd}"
        if eval "${cmd}"; then
            local generated_root="${tmp_out}"
            if [[ -d "${tmp_out}/NanTest" ]]; then
                generated_root="${tmp_out}/NanTest"
            fi
            rm -rf "${GEN_DIR}"
            mkdir -p "${GEN_DIR}"
            cp -a "${generated_root}/." "${GEN_DIR}/"
            return 0
        fi
    done
    return 1
}

replace_sources() {
    log "Replacing generated op_host/op_kernel(op_device) with local sources."
    mkdir -p "${GEN_DIR}/op_host" "${GEN_DIR}/op_kernel" "${GEN_DIR}/op_device"
    rm -f "${GEN_DIR}/op_host"/*.cpp "${GEN_DIR}/op_host"/*.h \
        "${GEN_DIR}/op_kernel"/*.cpp "${GEN_DIR}/op_kernel"/*.h \
        "${GEN_DIR}/op_device"/*.cpp

    cp -a "${SRC_TEMPLATE_DIR}/op_host"/. "${GEN_DIR}/op_host"/
    local host_soc_config
    host_soc_config=$(target_op_host_soc_config "${TARGET_SOC_VERSION}")
    sed -i "s/@NAN_TEST_SOC_CONFIG@/${host_soc_config}/g" "${GEN_DIR}/op_host/nan_test.cpp"
    log "Restricted NanTest host config to ${host_soc_config} for ${TARGET_SOC_VERSION}."

    if [[ -d "${SRC_TEMPLATE_DIR}/op_device" ]]; then
        cp -a "${SRC_TEMPLATE_DIR}/op_device"/. "${GEN_DIR}/op_device"/
    elif [[ -d "${SRC_TEMPLATE_DIR}/op_kernel" ]]; then
        cp -a "${SRC_TEMPLATE_DIR}/op_kernel"/. "${GEN_DIR}/op_kernel"/
    else
        echo "Neither ${SRC_TEMPLATE_DIR}/op_device nor ${SRC_TEMPLATE_DIR}/op_kernel exists." >&2
        return 1
    fi
}

build_and_install() {
    if [[ ! -f "${GEN_DIR}/build.sh" ]]; then
        echo "Generated build script not found: ${GEN_DIR}/build.sh" >&2
        return 1
    fi
    log "Building operator package from ${GEN_DIR}/build.sh"
    (cd "${GEN_DIR}" && bash ./build.sh)

    local run_pkg
    run_pkg=$(find "${GEN_DIR}" -type f -name "*.run" | head -n 1 || true)
    if [[ -z "${run_pkg}" ]]; then
        echo "No .run package produced under ${GEN_DIR}" >&2
        return 1
    fi

    log "Installing run package: ${run_pkg}"
    "${run_pkg}" --install-path="$PWD"
}

main() {
    ensure_cmd msopgen
    [[ -f "${JSON_PATH}" ]] || { echo "Missing ${JSON_PATH}" >&2; exit 1; }
    [[ -d "${SRC_TEMPLATE_DIR}" ]] || { echo "Missing ${SRC_TEMPLATE_DIR}" >&2; exit 1; }

    gen_project
    replace_sources
    build_and_install
    log "NanTest operator build+install finished."
}

main "$@"
