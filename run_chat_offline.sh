#!/bin/bash

# ==========================================
# chat_offline.py 运行脚本
# ==========================================

# 1. 指定运行的 GPU ID
# 如果有多个 GPU，可以使用逗号分隔，如 "0,1"
# 如果想使用 CPU，可以将其设置为 ""
export CUDA_VISIBLE_DEVICES="0"

# 2. 参数设置
# SAM2 模型权重路径
SAM2_MODEL="checkpoints/sam_weights/sam2.1_hiera_large.pt"

# SAM2 模型配置路径
SAM2_MODEL_CFG="checkpoints/sam_weights/config/sam2.1_hiera_l.yaml"

# Seg-Zero 模型路径 (HuggingFace 仓库名或本地路径)
SEGZERO_MODEL="checkpoints/seg-zero_weights-cache"

# 预测结果和关键帧输出目录
OUTPUT_DIR="./vis_output"

# 供大模型分析的候选帧数量
NUM_CANDIDATES=8

# ==========================================
# 运行命令
# ==========================================
echo "=========================================="
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo "SAM2 模型: $SAM2_MODEL"
echo "Seg-Zero 模型: $SEGZERO_MODEL"
echo "候选帧数量: $NUM_CANDIDATES"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="

python chat_offline.py \
    --sam2_model "$SAM2_MODEL" \
    --sam2_model_cfg "$SAM2_MODEL_CFG" \
    --segzero_model "$SEGZERO_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_candidates "$NUM_CANDIDATES"
