import argparse
import os
import sys
import time
import yaml

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from utils.util import *
from infer.prompt_gemma import *

from transformers import Gemma4ForConditionalGeneration, AutoProcessor, Gemma3ForConditionalGeneration
from infer.seg_zero import generate_mask


COLOR_MAP = [(255,0,0),(0,255,255),(128,255,0),(187, 19, 208),(222, 148, 80),(147, 71, 238),(98, 43, 249)]
def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS-Gemma stage 1")
    # parser.add_argument("--gemma3_model", default="../gemma3/models/gemma-3-12b-it", type=str)
    parser.add_argument("--gemma_model_name", default="Gemma-3-12B-it", type=str)
    parser.add_argument("--model_path", default=f"./checkpoints/gemma_weights/gemma-3-12b-it", type=str)
    parser.add_argument("--output_dir", default="./vis_output",type=str)
    parser.add_argument("--num_candidates", default=8, type=int)
    parser.add_argument("--save_name", default="test_sample", type=str)
    return parser.parse_args(args)
    
def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(os.path.join(args.output_dir,f"{args.gemma_model_name}-responses"),exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    gemma_model_id = args.model_path
    model_name = args.gemma_model_name.strip().lower()

    if model_name == "gemma-3-12b-it":
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_model_id, device_map="auto"
        ).eval()
    elif model_name in {"gemma-4-e4b-it", "gemma-4-26b-a4b-it", "gema-4-26b-a4b-it"}:
        gemma_model = Gemma4ForConditionalGeneration.from_pretrained(
            gemma_model_id, device_map="auto"
        ).eval()
    else:
        raise ValueError(
            f"Unsupported gemma model name: {args.gemma_model_name}. "
            "Expected gemma-3-12b-it, gemma-4-e4b-it, or gemma-4-26b-a4b-it."
        )

    gemma_processor = AutoProcessor.from_pretrained(gemma_model_id)
    
    # video_dir = input('Please input the directory of image sequence: ')
    video_dir = "test_sample/26d141ec-f952-3908-b4cc-ae359377424e"
    print("Using video directory:", video_dir)
    video_name = video_dir.split("/")[-1] if video_dir[-1]!="/" else video_dir.split("/")[-2]
    # user_query = input("Please input the user query: ")
    user_query = "My friend and I each drove our cars to another city. He was driving a white car and leading the way in front of me, but he drove too fast and I lost him. He called me to say that he had just been waiting at a traffic light and then crossed an intersection. Which one is most likely to be my friend's car?"
    print("Using user query:", user_query)
    
    if (not os.path.exists(video_dir)):
        raise RuntimeError("Input path not found.")
    
            
    # save_name = input("Please input the save name: ")
    save_name = args.save_name
    print("Using save name:", save_name)
    
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1].split('frame')[-1]))
    
    T = len(frame_names)
    sample_every = (T-1)//args.num_candidates+1
    
    keyframes = [Image.open(os.path.join(video_dir,path)) for path in frame_names[::sample_every]] 
    merged_result_path = merge_keyframe(args,video_name, keyframes)
    print(f"Waiting for {args.gemma_model_name} response...")
    
    start_time = time.perf_counter()
    answer = prompt_gemma(
        model=gemma_model,
        processor=gemma_processor,
        image_path=merged_result_path,
        query=user_query,
        num_keyframes=len(keyframes),
    )
    inference_time = time.perf_counter() - start_time
    print(f"Inference time: {inference_time:.2f} seconds")
    save_answer(
        user_query,
        answer,
        os.path.join(args.output_dir, f"{args.gemma_model_name}-responses", f"{save_name}.txt"),
        model_name=args.gemma_model_name,
        inference_time=inference_time,
    )
        
if __name__ == "__main__":
    main(sys.argv[1:])
    