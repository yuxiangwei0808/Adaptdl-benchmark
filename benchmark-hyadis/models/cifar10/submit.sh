#!/usr/bin/env bash
# set -x             # for debug
set -euo pipefail  # fail early
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd $SCRIPT_DIR

# kubectl -n ray port-forward example-cluster-ray-head-type-xxxxx 8265

export RAY_ADDRESS="http://127.0.0.1:8265"

ray job submit --runtime-env-json='{"working_dir": "./", "pip": []}' -- \
python main.py \
--data_path=/data/cifar-10/ \
--num_workers=2 \
--batch_size=4
