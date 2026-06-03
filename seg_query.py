import argparse
import os
import sys

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 假设利用现有的 utils 和 api
from utils.util import *
# from infer.prompt_api import parse_gpt_output
from infer.seg_zero import generate_mask

def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-FSSS Query Segmentation")
    parser.add_argument("--query_image_path", type=str, required=True, help="Path to the query image")
    parser.add_argument("--mllm_response_path", type=str, required=True, help="Path to the MLLM generated target descriptions")
    parser.add_argument("--sam2_model", default="checkpoints/sam2.1_hiera_large.pt", type=str)
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml", type=str)
    parser.add_argument("--segzero_model", default="Ricky06662/Seg-Zero-7B", type=str)
    parser.add_argument("--output_dir", default="./vis_output/query_masks", type=str)
    return parser.parse_args(args)

def seg_query_objects(args, reasoning_model, segmentation_model, processor, query_image_path, target_objects, save_mask=True, save_name="query_obj_"):
    """
    针对 FSSS 的 query 图像进行逐目标分割
    """
    masks = []
    for target in target_objects:
        # FSSS任务：不再需要找keyframe，只需遍历每个解析出的物体即可
        object_index = target.get("object_index", 0)
        object_desc = target.get("object_description", "")
        
        # 构造 Seg-Zero 所需的 Prompt
        prompt = f"Please segment {object_desc}."
        print(f"[{object_index}] Processing prompt: {prompt}")
        
        # 调用 Seg-Zero 和 SAM2 生成这一个物体在该 query 图上的 mask
        # 保存下来作为中间分析结果
        current_save_name = f"{save_name}{object_index}"
        
        mask = generate_mask(
            args=args,
            reasoning_model=reasoning_model,
            segmentation_model=segmentation_model,
            processor=processor,
            image_path=query_image_path,
            query=prompt,
            save_mask=save_mask,  # 设定为True，输出每个物体单独的分割mask
            save_name=current_save_name
        )
        
        if save_mask:
            # 读取原始图像
            image_cv = cv2.imread(query_image_path)
            if image_cv is not None:
                # BGR 格式，红色为 [0, 0, 255]
                color = np.array([0, 0, 255], dtype=np.float32)
                alpha = 1
                
                # 叠加半透明红色遮罩
                overlay = image_cv.copy()
                overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
                
                # 保存可视化结果
                vis_save_path = f"{current_save_name}_vis.jpg"
                cv2.imwrite(vis_save_path, overlay.astype(np.uint8))
                print(f"Saved mask visualization to {vis_save_path}")
        
        masks.append({
            "object_index": object_index,
            "description": object_desc,
            "mask": mask
        })
        
    return masks

def main(args):
    args = parse_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 确定设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2. 初始化 Seg-Zero (Qwen2.5-VL)
    print("Loading reasoning model (Seg-Zero) ...")
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.segzero_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(args.segzero_model, padding_side="left")

    # 3. 初始化 SAM 2 Image Predictor
    print("Loading segmentation model (SAM 2) ...")
    sam2_model = build_sam2(args.sam2_model_cfg, args.sam2_model, device=device)
    segmentation_model = SAM2ImagePredictor(sam2_model)

    # 4. 解析中间文字描述 (包含物体列表)
    if not os.path.exists(args.mllm_response_path):
        raise FileNotFoundError(f"MLLM response not found at: {args.mllm_response_path}")
    
    with open(args.mllm_response_path, "r", encoding="utf-8") as f:
        mllm_answer = f.read()
        
    # 利用预先定义好的 parse_gpt_output 获取对象列表
    # 期望的格式类似于 [{'object_index': 1, 'object_description': 'the wheel of the car'}, ...]
    target_objects = parse_gpt_output(mllm_answer)
    print(f"Found {len(target_objects)} objects to segment.")

    # 5. 执行 query 图像的分割逻辑，并输出每个物体单独的 mask 作为中间结果
    base_save_name = os.path.join(args.output_dir, "query_mask_obj_")
    
    # 生成中间结果
    masks_info = seg_query_objects(
        args=args,
        reasoning_model=reasoning_model,
        segmentation_model=segmentation_model,
        processor=processor,
        query_image_path=args.query_image_path,
        target_objects=target_objects,
        save_mask=True,             # 显式保存 mask 作为分析结果
        save_name=base_save_name
    )
    
    print(f"Segmentation finished. Intermediate masks saved to {args.output_dir}")

if __name__ == "__main__":
    main(sys.argv[1:])