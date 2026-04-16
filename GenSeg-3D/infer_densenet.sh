#!/usr/bin/env bash
# Run inference for DenseNet using the vscsv dataset loader (CSV-driven)
# Edit the variables below to point to your CSV, data root and checkpoint before running.

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
export CUDA_VISIBLE_DEVICES=0

CSV="/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg_VS_with_brainmask.csv"            # CSV with header: Patient_ID,image_nifti,mask_nifti,y_value
DATAROOT="/home/ubuntu/Documents/Nafisha/VS_data_nifti_Genseg/"           # root folder for nifti files (used by dataset loader)
CHECKPOINT="/home/ubuntu/Documents/Nafisha/GenSeg-VS/GenSeg-3D/checkpoint_e2e/vs-128-model-20250915-122955/densenet.pkl"  # path to densenet checkpoint
OUT_DIR="./inference_results"
BATCH_SIZE=4
NUM_WORKERS=4
THRESH=0.5
MODEL="pix2pix3d"            # use pix2pix3d for 3D behavior in vscsv_dataset
DATASET_MODE="vscsv"


mkdir -p "$OUT_DIR"

# resolve script directory so this works from any cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/infer_densenet.py" \
  --checkpoint $CHECKPOINT \
  --batch_size $BATCH_SIZE \
  --num_threads $NUM_WORKERS \
  --dataset_mode $DATASET_MODE \
  --csv_file "$CSV" \
  --dataroot "$DATAROOT" \
  --model $MODEL \
  "$@"

echo "Inference finished. Results (metrics, plots, predictions.csv) are in: $OUT_DIR"


