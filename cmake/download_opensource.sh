#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <pkg_name> <path> <sha256_value> [ <tag> ]"
    exit 1
fi

pkg_name=$1
path=$2
sha256_value=$3

if [ "$#" -ge 4 ]; then
    tag=$4
fi

url=$(awk -F " = " '/\['${pkg_name}'\]/{a=1}a==1&&$1~/url/{print $2;exit}' config.ini)
lib_path=$MSTT_LIB_PATH
if [ -n "$lib_path" ]; then
    url=${lib_path}$(echo $url | awk -F '/' -v OFS='/' '{print $5,$8}')
fi
if [[ ! $url = https* ]]; then
    echo "The URL of $pkg_name is illegal."
    exit 1
fi

echo "Start to download ${url}..."

if [ ! -d "$path" ]; then
    echo "The specified path does not exist: $path"
    exit 1
fi
cd "${path}"

fullname="${path}/$(basename "${url}")"
extension=$(echo "${url}" | awk -F'[./]' '{print $NF}')
if [[ "${extension}" == "gz" || "${extension}" == "zip" ]]; then
    if [[ -e "${fullname}" ]]; then
        echo "Source ${fullname} is exists, will not download again."
    else
        if [ "${INSTALL_WITHOUT_CHECK}" == "1" ]; then
            echo "[Warning] Downloading ${url} with insecure mode (-k). SSL verification disabled."
            curl -L "${url}" -o "${fullname}" -k
        else
            curl -L "${url}" -o "${fullname}"
        fi
        if [ $? -eq 0 ]; then
            echo "Download successful: ${url}"
        else
            echo "Download failed: ${url}"
            exit 1
        fi
    fi

    if [[ -z "${sha256_value}" ]]; then
        echo "Missing sha256 checksum for ${url}"
        exit 1
    fi

    sha256data=$(sha256sum "${fullname}" | cut -d' ' -f1)
    if [[ "${sha256data}" != "${sha256_value}" ]]; then
        echo "Failed to verify sha256: ${url}"
        exit 1
    fi

    if [[ "${extension}" == "gz" ]]; then
        tar -zxvf "${fullname}" -C ./ -n > /dev/null
    elif [[ "${extension}" == "zip" ]]; then
        unzip -n "${fullname}" -d ./ > /dev/null
    fi
elif [[ "${extension}" == "git" ]]; then
    if [[ -z "${sha256_value}" ]]; then
        echo "Missing commit id for ${url}"
        exit 1
    fi
    if [[ -d "${fullname%\.*}" ]]; then
        cd "${fullname%\.*}"
        is_clean=$(git status | grep 'nothing to commit, working tree clean' |wc -l)
        if [ $? -eq 0 ]&&[ $is_clean -eq 1 ]; then
            commit_id=$(git log -n 1 | awk 'NR==1 {print $2}')
            if [ $? -eq 0 ]&&[[ "${commit_id}" == "${sha256_value}" ]]; then
                echo "Source ${fullname} is exists, will not download again."
                cd "${path}"
            else
                rm -rf "${fullname%\.*}"
            fi
        else
            rm -rf "${fullname%\.*}"
        fi
    fi
    cd "${path}"
    if [ ! -d "${fullname%\.*}" ]; then
        if [[ -z "${tag}" ]]; then
            git clone "${url}"
        else
            git clone "${url}" -b "${tag}"
        fi
        if [ $? -eq 0 ]; then
            cd "${fullname%\.*}"
            git checkout --detach "${sha256_value}"
            if [[ "$(git rev-parse HEAD)" == "${sha256_value}" ]]; then
                cd "${path}"
                echo "Download successful: ${url}"
            else
                cd "${path}"
                echo "Download failed: ${url}"
                exit 1
            fi
        else
            echo "Download failed: ${url}"
            exit 1
        fi
    fi
else
    echo "Unknown url ${url}"
    exit 1
fi
