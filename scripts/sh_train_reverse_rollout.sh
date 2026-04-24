
## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/sh_train_reverse_rollout.sh
"

### Most configs are inherited from the iter_0 reverse config
### But we'll use a different dataset class: `TCLMasterSlaveDataset` to load both master and slave data together
CONFIG_DIR="./"
CONFIG_NAME="reverse_dp_force.yaml"


#DATA_ROOT="/home/geyuan/datasets/reverse/0209_tower_boby_easy_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0209_tower_boby_easy_reversed_240p.h5"
#SLAVE_DATA_ROOTS=(
#  "/home/geyuan/datasets/reverse/tower_boby_easy_reversed_filtered_iter1/"
#)
#SLAVE_H5_PATHS=(
#  "/home/geyuan/datasets/reverse/hdf5/tower_boby_easy_reversed_filtered_iter1_240p.h5"
#)
#PRETRAINED_TIME_LOG="2026.03.18-22.40.53"


#DATA_ROOT="/home/geyuan/datasets/reverse/0417_put_mouse_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0417_put_mouse_reversed_240p.h5"
#SLAVE_DATA_ROOTS=(
#  "/home/geyuan/datasets/reverse/put_mouse_reversed_filtered_iter1/"
#)
#SLAVE_H5_PATHS=(
#  "/home/geyuan/datasets/reverse/hdf5/put_mouse_reversed_filtered_iter1_240p.h5"
#)
#PRETRAINED_TIME_LOG="2026.04.19-21.57.24"


DATA_ROOT="/home/geyuan/datasets/reverse/0417_french_press_reversed/"
H5_PATH="/home/geyuan/datasets/reverse/hdf5/0417_french_press_reversed_240p.h5"
SLAVE_DATA_ROOTS=(
  "/home/geyuan/datasets/reverse/french_press_reversed_filtered_iter1/"
)
SLAVE_H5_PATHS=(
  "/home/geyuan/datasets/reverse/hdf5/french_press_reversed_filtered_iter1_240p.h5"
)
PRETRAINED_TIME_LOG="2026.04.22-18.13.59"


#DATA_ROOT="/home/geyuan/datasets/reverse/0417_test_tube_reversed/"
#H5_PATH="/home/geyuan/datasets/reverse/hdf5/0417_test_tube_reversed_240p.h5"


to_path() {
  local s="$1"
  printf '%s\n' "${s/-/\/}"   # 只替换第一个 '-'
}

PRETRAINED_CKPT="data/outputs/$(to_path "$PRETRAINED_TIME_LOG")_train_diffusion_transformer_hybrid_pusht_image/checkpoints/latest.ckpt"
SLAVE_DATA_ROOTS_OVERRIDE="[$(IFS=,; echo "${SLAVE_DATA_ROOTS[*]}")]"
SLAVE_H5_PATHS_OVERRIDE="[$(IFS=,; echo "${SLAVE_H5_PATHS[*]}")]"

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
MAIN_PORT="1234${_last_gpu_last_digit}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NUM_GPUS=${NUM_GPUS}, MAIN_PORT=${MAIN_PORT}"

accelerate launch \
  --multi_gpu \
  --num_processes=${NUM_GPUS} \
  --main_process_port=${MAIN_PORT} \
  train.py --config-dir=${CONFIG_DIR} --config-name=${CONFIG_NAME} training.seed=42 \
  training.device=${DEVICE} \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  task.dataset._target_="diffusion_policy.dataset.tcl_dataset.TCLMasterSlaveDataset"  \
  task.dataset.data_root="${DATA_ROOT}" \
  task.dataset.h5_path="${H5_PATH}"  \
  task.dataset.slave_data_roots="${SLAVE_DATA_ROOTS_OVERRIDE}" \
  task.dataset.slave_h5_paths="${SLAVE_H5_PATHS_OVERRIDE}" \
  task.dataset.zero_force=false \
  pretrained_ckpt="${PRETRAINED_CKPT}" \
  training.checkpoint_every=20 \
  training.num_epochs=100 \
  task.dataset.pad_after=0
