#!/usr/bin/env bash
# set -x             # for debug
set -euo pipefail  # fail early
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

docker build . \
-t cr-cn-beijing.volces.com/hpcaitech/hyadis-runtime-benchmark-dependency:latest
