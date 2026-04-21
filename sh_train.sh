

## Usage
tmp="
conda activate robodiff
cd /mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/dp23rss
export PYTHONPATH=/mnt/dongxu-fs1/data-ssd/qiyuanqiao/workspace/dp23rss:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 bash sh_train.sh
"

CONFIG_DIR="./"
#CONFIG_NAME="tcl_dp_transformer.yaml"
#CONFIG_NAME="tcl_hdfree_shovel.yaml"
#CONFIG_NAME="tcl_hdfree_dp.yaml"
#CONFIG_NAME="tcl_dp_force.yaml"
#CONFIG_NAME="libero_force_dp.yaml"

### Reverse Collect ###
CONFIG_NAME="reverse_dp_force.yaml"

DEVICE="cuda"

export HYDRA_FULL_ERROR=1

set -e
set -x

wandb online

### DDP training with accelerate
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[ERROR] CUDA_VISIBLE_DEVICES is empty. Please set it before running."
  exit 1
fi

IFS=',' read -r -a _raw_gpus <<< "${CUDA_VISIBLE_DEVICES}"
_gpus=()
for _gpu in "${_raw_gpus[@]}"; do
  _gpu="${_gpu//[[:space:]]/}"
  if [[ -n "${_gpu}" ]]; then
    _gpus+=("${_gpu}")
  fi
done

if [[ ${#_gpus[@]} -eq 0 ]]; then
  echo "[ERROR] No valid GPU id parsed from CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'."
  exit 1
fi

NUM_GPUS=${#_gpus[@]}
_last_gpu="${_gpus[$((NUM_GPUS-1))]}"
_last_gpu_last_digit="${_last_gpu: -1}"
MAIN_PORT="2234${_last_gpu_last_digit}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}, MAIN_PORT=${MAIN_PORT}"

# python train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42  \
#   training.device=${DEVICE}  \
#   hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MAIN_PORT} \
  train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42 \
  training.device=${DEVICE} \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
