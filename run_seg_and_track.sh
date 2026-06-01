#!/bin/bash

# Default values
GPU_ID="0,1"
VIDEO_DIR="test_sample/26d141ec-f952-3908-b4cc-ae359377424e"
MLLM_RESPONSE_PATH="outputs/Gemma-4-E4B-it-responses/test_1GPU.txt"
SAM2_MODEL="./checkpoints/sam_weights/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
SEGZERO_MODEL="checkpoints/seg-zero_weights-cache/models--Ricky06662--Seg-Zero-7B/snapshots/7764ee846ff4d68e7c65ede1e63531302f63860a"
OUTPUT_DIR="./outputs/Gemma-4-E4B-it-responses"
NUM_CANDIDATES=8

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift ;;
        --video_dir) VIDEO_DIR="$2"; shift ;;
        --mllm_response_path) MLLM_RESPONSE_PATH="$2"; shift ;;
        --sam2_model) SAM2_MODEL="$2"; shift ;;
        --sam2_model_cfg) SAM2_MODEL_CFG="$2"; shift ;;
        --segzero_model) SEGZERO_MODEL="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --num_candidates) NUM_CANDIDATES="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --gpu <id>                  Specify GPU ID(s) to use, e.g., 0 or 0,1 (default: 0)"
            echo "  --video_dir <dir>           Path to the video directory"
            echo "  --mllm_response_path <path> Path to the MLLM response file (from run_gemma.py)"
            echo "  --sam2_model <path>         Path to SAM2 model (default: checkpoints/sam_weights/sam2.1_hiera_large.pt)"
            echo "  --sam2_model_cfg <path>     Path to SAM2 model config (default: configs/sam2.1/sam2.1_hiera_l.yaml)"
            echo "  --segzero_model <path>      Path to SegZero model (default: Ricky06662/Seg-Zero-7B)"
            echo "  --output_dir <dir>          Output directory (default: ./vis_output)"
            echo "  --num_candidates <num>      Number of candidates (default: 8)"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$VIDEO_DIR" ] || [ -z "$MLLM_RESPONSE_PATH" ]; then
    echo "Error: --video_dir and --mllm_response_path are required parameters."
    echo "Use --help for more information."
    exit 1
fi

echo "========================================"
echo "Running seg_and_track.py with configuration:"
echo "GPU ID:              $GPU_ID"
echo "Video Dir:           $VIDEO_DIR"
echo "MLLM Response Path:  $MLLM_RESPONSE_PATH"
echo "SAM2 Model:          $SAM2_MODEL"
echo "SAM2 Model Cfg:      $SAM2_MODEL_CFG"
echo "SegZero Model:       $SEGZERO_MODEL"
echo "Output Directory:    $OUTPUT_DIR"
echo "Num Candidates:      $NUM_CANDIDATES"
echo "========================================"

# Execute python script with specified GPU(s) and parameters
export PATH="/data/yukun/miniconda3/envs/cot-rvs/bin:$PATH"
CUDA_VISIBLE_DEVICES=$GPU_ID python seg_and_track.py \
    --video_dir "$VIDEO_DIR" \
    --mllm_response_path "$MLLM_RESPONSE_PATH" \
    --sam2_model "$SAM2_MODEL" \
    --sam2_model_cfg "$SAM2_MODEL_CFG" \
    --segzero_model "$SEGZERO_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_candidates "$NUM_CANDIDATES"