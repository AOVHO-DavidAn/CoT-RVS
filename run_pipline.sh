#!/bin/bash

# Default values
GPU_ID="0,1,2,3"
GEMMA_MODEL_NAME="Gemma-4-26B-A4B-it"
# GEMMA_MODEL_NAME="Gemma-4-E4B-it"
# GEMMA_MODEL_NAME="Gemma-3-12B-it"
MODEL_PATH="./checkpoints/gemma_weights/${GEMMA_MODEL_NAME}"
K_SHOT=1


CUDA_VISIBLE_DEVICES=$GPU_ID python run_fsss_pipeline.py \
  --dataset fss-1000 \
  --dataset_root ./datasets \
  --k_shot ${K_SHOT} \
  --num_episodes 1 \
  --gemma_model_name ${GEMMA_MODEL_NAME} \
  --gemma_model_path ${MODEL_PATH} \
  --sam2_model ./checkpoints/sam_weights/sam2.1_hiera_large.pt \
  --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  # --segzero_model Ricky06662/Seg-Zero-7B \ 
  --