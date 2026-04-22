## Usage
tmp="
conda activate robodiff
cd ~/code/dp23rss_fork
export PYTHONPATH=~/code/dp23rss_fork:$PYTHONPATH
CUDA_VISIBLE_DEVICES=2,3,4,5 EN_FREEZE_OBS_ENCODER=false bash scripts/sh_train_moe.sh
"

### Merge MoE ###
CONFIG_DIR="configs/"
CONFIG_NAME="pusht256_moe.yaml"

domain_shift="goal"
zarr_paths=(
  "data/pusht/pusht_256_block.zarr"
  "data/pusht/pusht_256_goal.zarr"
  "data/pusht/pusht_256_light.zarr"
)
ffn_expand_factor=(2 4 6)
teacher_ckpts=(
#  "data/outputs/2026.04.15/17.39.01_train_diffusion_transformer_hybrid_pusht256/checkpoints/latest.ckpt"
#  "data/outputs/2026.04.15/16.44.03_train_diffusion_transformer_hybrid_pusht256/checkpoints/latest.ckpt"
#  "data/outputs/2026.04.15/16.59.43_train_diffusion_transformer_hybrid_pusht256/checkpoints/latest.ckpt"
)

zarr_paths_override="[$(IFS=,; echo "${zarr_paths[*]}")]"
ffn_expand_factor_override="[$(IFS=,; echo "${ffn_expand_factor[*]}")]"
teacher_ckpts_override="[$(IFS=,; echo "${teacher_ckpts[*]}")]"

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
  task.env_runner.domain_shift=${domain_shift} \
  task.dataset.zarr_paths=${zarr_paths_override} \
  policy.ffn_expand_factor=${ffn_expand_factor_override} \
  policy.teacher_ckpts=${teacher_ckpts_override} \
  policy.en_freeze_obs_encoder=${EN_FREEZE_OBS_ENCODER:-true}
