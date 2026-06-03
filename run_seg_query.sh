#!/bin/bash

# Default values
GPU_ID="0,1"
MODEL_NAME="Gemma-4-26B-A4B-it"
TASK="balloon"
QUERY_IMAGE_PATH="few-shot_episode/${TASK}/query_img.jpg"
MLLM_RESPONSE_PATH="outputs/fsss_outputs/${MODEL_NAME}-responses/test_sample_fsss.txt"
SAM2_MODEL="./checkpoints/sam_weights/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SEGZERO_MODEL="checkpoints/seg-zero_weights-cache/models--Ricky06662--Seg-Zero-7B/snapshots/7764ee846ff4d68e7c65ede1e63531302f63860a"
OUTPUT_DIR="./outputs/fsss_outputs/query_masks/${TASK}/${MODEL_NAME}"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift ;;
        --query_image_path) QUERY_IMAGE_PATH="$2"; shift ;;
        --mllm_response_path) MLLM_RESPONSE_PATH="$2"; shift ;;
        --sam2_model) SAM2_MODEL="$2"; shift ;;
        --sam2_model_cfg) SAM2_MODEL_CFG="$2"; shift ;;
        --segzero_model) SEGZERO_MODEL="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --gpu <id>                  Specify GPU ID(s) to use, e.g., 0 or 0,1 (default: 0,1)"
            echo "  --query_image_path <path>   Path to the single query image"
            echo "  --mllm_response_path <path> Path to the MLLM response file containing target descriptions"
            echo "  --sam2_model <path>         Path to SAM2 model (default: checkpoints/sam_weights/sam2.1_hiera_large.pt)"
            echo "  --sam2_model_cfg <path>     Path to SAM2 model config (default: configs/sam2.1/sam2.1_hiera_l.yaml)"
            echo "  --segzero_model <path>      Path to SegZero model (default: Ricky06662/Seg-Zero-7B)"
            echo "  --output_dir <dir>          Output directory for generated masks (default: ./vis_output/query_masks)"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$QUERY_IMAGE_PATH" ] || [ -z "$MLLM_RESPONSE_PATH" ]; then
    echo "Error: --query_image_path and --mllm_response_path are required parameters."
    echo "Use --help for more information."
    exit 1
fi

echo "========================================"
echo "Running seg_query.py with configuration:"
echo "GPU ID:              $GPU_ID"
echo "Query Image Path:    $QUERY_IMAGE_PATH"
echo "MLLM Response Path:  $MLLM_RESPONSE_PATH"
echo "SAM2 Model:          $SAM2_MODEL"
echo "SAM2 Model Cfg:      $SAM2_MODEL_CFG"
echo "SegZero Model:       $SEGZERO_MODEL"
echo "Output Directory:    $OUTPUT_DIR"
echo "========================================"

# Execute python script with specified GPU(s) and parameters
export PATH="/data/yukun/miniconda3/envs/cot-rvs/bin:$PATH"
CUDA_VISIBLE_DEVICES=$GPU_ID python seg_query.py \
    --query_image_path "$QUERY_IMAGE_PATH" \
    --mllm_response_path "$MLLM_RESPONSE_PATH" \
    --sam2_model "$SAM2_MODEL" \
    --sam2_model_cfg "$SAM2_MODEL_CFG" \
    --segzero_model "$SEGZERO_MODEL" \
    --output_dir "$OUTPUT_DIR"
