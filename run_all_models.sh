#!/bin/bash
# SLURM job launcher for Winter RSC SSL + Meta-Learning.
# Submit with:  sbatch run_all_models.sh
# Run locally with: bash run_all_models.sh
#SBATCH --job-name=RSC_SSL_META_Hybrid_HardEpisodic
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:4
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --time=6-23:30:00
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=errors/%x-%j.err

set -euo pipefail

mkdir -p Output outputs errors

EXPERIMENT_NAME="${EXPERIMENT_NAME:-SSL_Hybrid_FineTune_Episodic}"
RUN_ID="${RUN_ID:-${EXPERIMENT_NAME}_${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-Output/$RUN_ID}"
WARMUP_DIR="${WARMUP_DIR:-Warm-up Dataset}"
DATA_DIR="${DATA_DIR:-Dataset_classes/1 Defined}"
MODELS="${MODELS:-convnext dino}"
CONVNEXT_NAME="${CONVNEXT_NAME:-convnext_base_in22k}"
DINO_NAME="${DINO_NAME:-vit_base_patch14_dinov2.lvd142m}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SUPPORT_PER_CLASS="${SUPPORT_PER_CLASS:-80}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-80}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
EPOCHS_SSL="${EPOCHS_SSL:-100}"
EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-70}"
EPOCHS_META="${EPOCHS_META:-70}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-80}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-true}"
VENV_PATH="${VENV_PATH:-${VENV_DIR:-$HOME/projects/p65425/mmabdela/mma_venv}}"

emit_job_metadata() {
    echo "========================================================================"
    echo "Winter RSC SSL + Meta-Learning Experiment Metadata"
    echo "========================================================================"
    echo "Job id:             ${SLURM_JOB_ID:-local}"
    echo "Job name:           ${SLURM_JOB_NAME:-local}"
    echo "Run id:             $RUN_ID"
    echo "Host:               $(hostname)"
    echo "Working dir:        $(pwd)"
    echo "Date:               $(date)"
    echo "Experiment set:     $EXPERIMENT_NAME"
    echo "Models:             $MODELS"
    echo "ConvNeXt name:      $CONVNEXT_NAME"
    echo "DINO name:          $DINO_NAME"
    echo "Warm-up dir:        $WARMUP_DIR"
    echo "Data dir:           $DATA_DIR"
    echo "Output root:        $OUTPUT_ROOT"
    echo "Image size:         $IMAGE_SIZE"
    echo "Batch size:         $BATCH_SIZE"
    echo "Support per class:  $SUPPORT_PER_CLASS"
    echo "Query per class:    $QUERY_PER_CLASS"
    echo "Num workers:        $NUM_WORKERS"
    echo "Device:             $DEVICE"
    echo "SSL epochs:         $EPOCHS_SSL"
    echo "Fine-tune epochs:   $EPOCHS_FINETUNE"
    echo "Meta epochs:        $EPOCHS_META"
    echo "Episodes/epoch:     $EPISODES_PER_EPOCH"
    echo "Process plan:"
    echo "  1. Validate Warm-Up and 5-class RSC folders"
    echo "  2. Run SSL + class-balanced fine-tuning for ConvNeXt and DINO"
    echo "  3. Run balanced and hard episodic meta-learning"
    echo "  4. Save isolated outputs under $OUTPUT_ROOT"
    echo "  5. Emit metrics, figures, checkpoints, predictions, and reports"
    echo "========================================================================"
}

# Print metadata at the top of both SLURM .out and .err files.
emit_job_metadata
emit_job_metadata >&2

if command -v module >/dev/null 2>&1; then
    module reset
    module load StdEnv/2023
    module load gcc/12.3
    module load python/3.10
    module load opencv/4.10.0
fi

if [[ -f "$VENV_PATH/bin/activate" ]]; then
    source "$VENV_PATH/bin/activate"
else
    echo "WARNING: virtual environment not found at $VENV_PATH; using current Python." >&2
fi

if [[ "$INSTALL_REQUIREMENTS" == "true" ]]; then
    python -m pip install --no-index -r requirements.txt
fi

# ---------------------------------------------------------------------------
# Thread & buffering settings — critical for flushing epoch progress to .out
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ---------------------------------------------------------------------------
# Pretrained / cache paths for offline clusters
# ---------------------------------------------------------------------------
export RSC_PRETRAINED_CACHE_DIR="${RSC_PRETRAINED_CACHE_DIR:-$PWD/weights/pretrained_cache}"
export HF_HOME="$RSC_PRETRAINED_CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$RSC_PRETRAINED_CACHE_DIR/torch"
export TIMM_HOME="$RSC_PRETRAINED_CACHE_DIR/timm"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

mkdir -p weights "$HF_HUB_CACHE" "$TORCH_HOME" "$TIMM_HOME" "$TRANSFORMERS_CACHE" "$OUTPUT_ROOT"

# ---------------------------------------------------------------------------
# Environment diagnostics printed into the .out file
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Winter RSC SSL + Meta-Learning - Job ${SLURM_JOB_ID:-local}"
echo "========================================================================"
echo "Host:              $(hostname)"
echo "Working dir:       $(pwd)"
echo "Date:              $(date)"
echo "Python:            $(which python)"
echo "Output root:       $OUTPUT_ROOT"
echo "Models:            $MODELS"
echo "ConvNeXt name:     $CONVNEXT_NAME"
echo "DINO name:         $DINO_NAME"
echo "Image size:        $IMAGE_SIZE"
echo "Batch size:        $BATCH_SIZE"
echo "Support per class: $SUPPORT_PER_CLASS"
echo "Query per class:   $QUERY_PER_CLASS"
echo "Num workers:       $NUM_WORKERS"
echo "Device:            $DEVICE"
echo "SSL epochs:        $EPOCHS_SSL"
echo "Fine-tune epochs:  $EPOCHS_FINETUNE"
echo "Meta epochs:       $EPOCHS_META"
echo "Episodes/epoch:    $EPISODES_PER_EPOCH"
nvidia-smi -L || true
python - <<'PY'
import sys
import torch
print(f"Python:            {sys.version}")
print(f"PyTorch:           {torch.__version__}")
print(f"CUDA available:    {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PY
echo "========================================================================"
echo ""

# ---------------------------------------------------------------------------
# Run the RSC SSL + Meta-Learning pipeline.
#
# This runs the recommended best-overall setting:
#   - selected experiment: SSL_Hybrid_FineTune_Episodic
#   - models: convnext dino
#   - ConvNeXt: convnext_base_in22k
#   - image size: 512
#   - batch size: 32
#   - support/query per class: 80/80
#   - loss: weighted_ce
#   - sampler: balanced
#   - prototype distance: euclidean
#   - purpose: combine SSL, balanced supervised fine-tuning, balanced episodes, and hard One Track episodes
#
# PYTHONUNBUFFERED=1 ensures progress is flushed into the SLURM .out file.
# ---------------------------------------------------------------------------
python run_ssl_meta_rsc.py \
    --experiment SSL_Hybrid_FineTune_Episodic \
    --output_dir "$OUTPUT_ROOT" \
    --warmup_dir "$WARMUP_DIR" \
    --data_dir "$DATA_DIR" \
    --models $MODELS \
    --convnext_name "$CONVNEXT_NAME" \
    --dino_name "$DINO_NAME" \
    --image_size "$IMAGE_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --support_per_class "$SUPPORT_PER_CLASS" \
    --query_per_class "$QUERY_PER_CLASS" \
    --num_workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    --loss weighted_ce \
    --sampler balanced \
    --prototype_distance euclidean \
    --epochs_ssl "$EPOCHS_SSL" \
    --epochs_finetune "$EPOCHS_FINETUNE" \
    --epochs_meta "$EPOCHS_META" \
    --episodes_per_epoch "$EPISODES_PER_EPOCH" \
    --augmentation_strength medium

echo ""
echo "========================================================================"
echo "  Job ${SLURM_JOB_ID:-local} completed at $(date)"
echo "========================================================================"
