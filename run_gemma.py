import argparse
import os
import sys
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

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from seg_zero import generate_mask


COLOR_MAP = [(255,0,0),(0,255,255),(128,255,0),(187, 19, 208),(222, 148, 80),(147, 71, 238),(98, 43, 249)]
def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS-Gemma stage 1")
    parser.add_argument("--gemma3_model", default="../gemma3/models/gemma-3-12b-it", type=str)
    parser.add_argument("--output_dir", default="./vis_output",type=str)
    parser.add_argument("--num_candidates", default=8, type=int)
    return parser.parse_args(args)
    
def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(os.path.join(args.output_dir,"Gemma-3 responses"),exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    gemma_model_id = args.gemma3_model

    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_model_id, device_map="auto"
    ).eval()

    gemma_processor = AutoProcessor.from_pretrained(gemma_model_id)
    
    video_dir = input('Please input the directory of image sequence: ')
    video_name = video_dir.split("/")[-1] if video_dir[-1]!="/" else video_dir.split("/")[-2]
    user_query = input("Please input the user query: ")
    
    if (not os.path.exists(video_dir)):
        raise RuntimeError("Input path not found.")
    
            
    save_name = input("Please input the save name: ")
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1].split('frame')[-1]))
    
    T = len(frame_names)
    sample_every = (T-1)//args.num_candidates+1
    
    keyframes = [Image.open(os.path.join(video_dir,path)) for path in frame_names[::sample_every]] 
    merged_result_path = merge_keyframe(args,video_name, keyframes)
    print("Waiting for Gemma3 response...")
    answer = prompt_gemma(model=gemma_model,processor=gemma_processor,image_path=merged_result_path,query=user_query,num_keyframes=len(keyframes))
    save_answer(user_query, answer, os.path.join(args.output_dir,"Gemma-3-responses",f"{save_name}.txt"))
        
if __name__ == "__main__":
    main(sys.argv[1:])
    