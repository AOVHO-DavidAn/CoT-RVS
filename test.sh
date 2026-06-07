#!/bin/bash

# Test One-shot Semantic Segmentation on COCO-20i (Fold 0)
gpus="0,1,2,3"
BENCHMARK="coco"
FOLD=0

CUDA_VISIBLE_DEVICES=$gpus python run_fsss_pipeline.py \
    --benchmark $BENCHMARK \
    --nshot 1 \
    --fold $FOLD \
    --log-root "outputs/logs/$BENCHMARK/fold$FOLD" \
