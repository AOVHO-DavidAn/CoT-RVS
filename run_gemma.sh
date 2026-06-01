#!/bin/bash

# Default values
GPU_ID="0,1,2,3"
# GEMMA_MODEL_NAME="gemma-3-12b-it"
GEMMA_MODEL_NAME="Gemma-4-26B-A4B-it"
MODEL_PATH="./checkpoints/gemma_weights/${GEMMA_MODEL_NAME}"
OUTPUT_DIR="./outputs"
NUM_CANDIDATES=8
SAVE_NAME="test_4GPU"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift ;;
        --gemma_model_name) GEMMA_MODEL_NAME="$2"; shift ;;
        --model_path) MODEL_PATH="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --num_candidates) NUM_CANDIDATES="$2"; shift ;;
        --save_name) SAVE_NAME="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --gpu <id>               Specify GPU ID(s) to use, e.g., 0 or 0,1 (default: 0,1)"
            echo "  --gemma_model_name <name> Model name to display (default: gemma-3-12b-it)"
            echo "  --model_path <path>      Path to Gemma model weights (default: ./checkpoints/gemma_weights/gemma-3-12b-it)"
            echo "  --output_dir <dir>       Output directory (default: ./outputs)"
            echo "  --num_candidates <num>   Number of candidates (default: 8)"
            echo "  --save_name <name>       Save name for the output test (default: test_sample)"
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================"
echo "Running run_gemma.py with configuration:"
echo "GPU ID:             $GPU_ID"
echo "Gemma Model Name:   $GEMMA_MODEL_NAME"
echo "Model Path:         $MODEL_PATH"
echo "Output Directory:   $OUTPUT_DIR"
echo "Num Candidates:     $NUM_CANDIDATES"
echo "Save Name:          $SAVE_NAME"
echo "========================================"

# Execute python script with specified GPU(s) and parameters
CUDA_VISIBLE_DEVICES=$GPU_ID python run_gemma.py \
    --gemma_model_name "$GEMMA_MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_candidates "$NUM_CANDIDATES" \
    --save_name "$SAVE_NAME"
