#!/bin/bash

# Default values
GPU_ID="0,1,2,3"
GEMMA_MODEL_NAME="Gemma-4-26B-A4B-it"
# GEMMA_MODEL_NAME="Gemma-4-E4B-it"
# GEMMA_MODEL_NAME="Gemma-3-12B-it"
MODEL_PATH="./checkpoints/gemma_weights/${GEMMA_MODEL_NAME}"
IMAGE_PATH="./outputs/merged_episode/test_input.jpg"
K_SHOT=1
OUTPUT_DIR="./outputs/fsss_outputs"
SAVE_NAME="test_sample_fsss"
TASK_NOTE="FSSS offline merged image"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
	case $1 in
		--gpu) GPU_ID="$2"; shift ;;
		--gemma_model_name) GEMMA_MODEL_NAME="$2"; shift ;;
		--model_path) MODEL_PATH="$2"; shift ;;
		--image_path) IMAGE_PATH="$2"; shift ;;
		--k_shot) K_SHOT="$2"; shift ;;
		--output_dir) OUTPUT_DIR="$2"; shift ;;
		--save_name) SAVE_NAME="$2"; shift ;;
		--task_note) TASK_NOTE="$2"; shift ;;
		-h|--help)
			echo "Usage: $0 [options]"
			echo "Options:"
			echo "  --gpu <id>               Specify GPU ID(s) to use, e.g., 0 or 0,1 (default: 0)"
			echo "  --gemma_model_name <name> Model name to display (default: Gemma-3-12B-it)"
			echo "  --model_path <path>      Path to Gemma model weights (default: ./checkpoints/gemma_weights/<gemma_model_name>)"
			echo "  --image_path <path>      Path to merged image (default: ./outputs/merged_episode/test_input.jpg)"
			echo "  --k_shot <num>           Number of shots (default: 1)"
			echo "  --output_dir <dir>       Output directory (default: ./vis_output)"
			echo "  --save_name <name>       Save name for the output test (default: test_sample_fsss)"
			echo "  --task_note <text>       Task note saved in the output file"
			exit 0
			;;
		*) echo "Unknown parameter passed: $1"; exit 1 ;;
	esac
	shift
done

echo "========================================"
echo "Running run_gemma_fsss.py with configuration:"
echo "GPU ID:             $GPU_ID"
echo "Gemma Model Name:   $GEMMA_MODEL_NAME"
echo "Model Path:         $MODEL_PATH"
echo "Image Path:         $IMAGE_PATH"
echo "K-shot:             $K_SHOT"
echo "Output Directory:   $OUTPUT_DIR"
echo "Save Name:          $SAVE_NAME"
echo "Task Note:          $TASK_NOTE"
echo "========================================"

# Execute python script with specified GPU(s) and parameters
CUDA_VISIBLE_DEVICES=$GPU_ID python run_gemma_fsss.py \
	--gemma_model_name "$GEMMA_MODEL_NAME" \
	--model_path "$MODEL_PATH" \
	--image_path "$IMAGE_PATH" \
	--k_shot "$K_SHOT" \
	--output_dir "$OUTPUT_DIR" \
	--save_name "$SAVE_NAME" \
	--task_note "$TASK_NOTE"
