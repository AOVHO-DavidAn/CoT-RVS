from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Gemma4ForConditionalGeneration
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import Qwen2_5_VLForConditionalGeneration

from fsss.datasets import EpisodeSample
from infer.prompt_gemma_fsss import prompt_gemma
from seg_query import seg_query_objects
from utils.util import parse_gpt_output


@dataclass
class FSSSResult:
    episode_id: str
    # class_name: str
    response_text: str
    target_objects: List[Dict[str, Any]]
    masks_info: List[Dict[str, Any]]


def _resolve_font(font_path: str, font_size: int) -> ImageFont.ImageFont:
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    width, height = image.size
    new_width = int(width * (target_height / height))
    return image.resize((new_width, target_height), Image.Resampling.BILINEAR)


def _add_header(image: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    width, height = image.size
    header_height = 60
    header = Image.new("RGB", (width, header_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(header)
    draw.text((10, 15), text, fill=(0, 0, 0), font=font)
    combined = Image.new("RGB", (width, header_height + height), color=(255, 255, 255))
    combined.paste(header, (0, 0))
    combined.paste(image, (0, header_height))
    return combined


def build_prompt_image(
    support_images: List[Image.Image],
    support_masks: List[np.ndarray],
    query_image: Image.Image,
    panel_height: int,
    font_path: str,
) -> Image.Image:
    font = _resolve_font(font_path, max(12, panel_height // 12))

    panels: List[Image.Image] = []
    for shot_index, (support_image, support_mask) in enumerate(
        zip(support_images, support_masks), start=1
    ):
        support_image = support_image.convert("RGB")
        support_image = _resize_to_height(support_image, panel_height)

        mask = Image.fromarray(support_mask.astype(np.uint8) * 255)
        mask = mask.resize(
            (int(mask.width * (panel_height / mask.height)), panel_height),
            Image.Resampling.NEAREST,
        )
        mask_np = np.array(mask) > 0

        # --- 抠出前景目标，背景置白，让模型只关注物体本身 ---
        support_np = np.array(support_image)
        bg_color = np.array([255, 255, 255], dtype=np.uint8)  # 白色背景
        support_np[~mask_np] = bg_color
        cutout_image = Image.fromarray(support_np)

        mask_rgb = Image.fromarray(mask_np.astype(np.uint8) * 255).convert("RGB")

        panels.append(
            _add_header(support_image, f"Support Image #{shot_index}", font)
        )
        panels.append(
            _add_header(cutout_image, f"Target Object Cutout #{shot_index}", font)
        )
        panels.append(
            _add_header(mask_rgb, f"Binary Mask #{shot_index}", font)
        )

    query_image = query_image.convert("RGB")
    query_image = _resize_to_height(query_image, panel_height)
    panels.append(_add_header(query_image, "Query Image", font))

    total_width = sum(panel.size[0] for panel in panels)
    max_height = max(panel.size[1] for panel in panels)
    merged = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for panel in panels:
        merged.paste(panel, (x_offset, 0))
        x_offset += panel.size[0]
    return merged


class FSSSPipeline:
    def __init__(
        self,
        gemma_model_name: str,
        gemma_model_path: str,
        sam2_model: str,
        sam2_model_cfg: str,
        segzero_model: str,
        output_dir: str,
        panel_height: int = 512,
        save_intermediates: bool = False,
        benchmark: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.panel_height = panel_height
        self.save_intermediates = save_intermediates
        self.font_path = os.path.join(os.getcwd(), "NotoSansMath-Regular.ttf")

        self.device = self._resolve_device()
        self.gemma_model_name = gemma_model_name
        self.gemma_model_path = gemma_model_path

        self.gemma_model, self.gemma_processor = self._load_gemma_model()
        self.reasoning_model, self.segzero_processor = self._load_segzero_model(
            segzero_model
        )
        self.segmentation_model = self._load_sam2_model(sam2_model_cfg, sam2_model)
        self.benchmark = benchmark

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            return device
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_gemma_model(self) -> Any:
        model_name = self.gemma_model_name.strip().lower()
        if model_name == "gemma-3-12b-it":
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.gemma_model_path, device_map="auto"
            ).eval()
        elif model_name in {"gemma-4-e4b-it", "gemma-4-26b-a4b-it"}:
            model = Gemma4ForConditionalGeneration.from_pretrained(
                self.gemma_model_path, device_map="auto"
            ).eval()
        else:
            raise ValueError(
                f"Unsupported Gemma model: {self.gemma_model_name}. "
                "Expected gemma-3-12b-it, gemma-4-e4b-it, or gemma-4-26b-a4b-it."
            )
        processor = AutoProcessor.from_pretrained(self.gemma_model_path)
        return model, processor

    def _load_segzero_model(self, segzero_model: str) -> Any:
        reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            segzero_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        reasoning_model.eval()
        processor = AutoProcessor.from_pretrained(segzero_model, padding_side="left")
        return reasoning_model, processor

    def _load_sam2_model(self, sam2_model_cfg: str, sam2_model: str) -> SAM2ImagePredictor:
        sam2 = build_sam2(sam2_model_cfg, sam2_model, device=self.device)
        return SAM2ImagePredictor(sam2)

    def run_episode(self, query_img, support_imgs, support_masks, k_shot: int, episode_id: str) -> FSSSResult:
        os.makedirs(self.output_dir, exist_ok=True)
        episode_dir = os.path.join(self.output_dir, self.benchmark, f"episode_{episode_id}")
        if self.save_intermediates:
            os.makedirs(episode_dir, exist_ok=True)

        # ---- Tensor → PIL 转换 ----
        # dataloader 输出的 query_img 形状为 [1, 3, H, W] 或 [3, H, W]
        # support_imgs 形状为 [1, N, 3, H, W] 或 [N, 3, H, W]
        # support_masks 形状为 [1, N, H, W] 或 [N, H, W]
        # 统一去掉 batch 维度，转换为 PIL Image / np.ndarray

        to_pil = T.ToPILImage()

        if query_img.dim() == 4:
            query_img = query_img.squeeze(0)          # [1, 3, H, W] → [3, H, W]
        query_pil = to_pil(query_img.cpu())            # Tensor → PIL Image

        if support_imgs.dim() == 5:
            support_imgs = support_imgs.squeeze(0)     # [1, N, 3, H, W] → [N, 3, H, W]
        support_pils = [to_pil(img.cpu()) for img in support_imgs]

        if support_masks.dim() == 4:
            support_masks = support_masks.squeeze(0)   # [1, N, H, W] → [N, H, W]
        support_masks_np = [mask.cpu().numpy() for mask in support_masks]

        prompt_image = build_prompt_image(
            support_pils,
            support_masks_np,
            query_pil,
            panel_height=self.panel_height,
            font_path=self.font_path,
        )

        if self.save_intermediates:
            prompt_path = os.path.join(episode_dir, "prompt_image.jpg")
            prompt_image.save(prompt_path)

        start_time = time.perf_counter()
        response_text = prompt_gemma(
            model=self.gemma_model,
            processor=self.gemma_processor,
            image=prompt_image,
            k_shot=k_shot,
        )
        inference_time = time.perf_counter() - start_time

        if self.save_intermediates:
            response_path = os.path.join(episode_dir, "gemma_response.txt")
            with open(response_path, "w", encoding="utf-8") as f:
                # f.write(f"Class: {episode.class_name}\n")
                f.write(f"Inference time: {inference_time:.2f} seconds\n")
                f.write(response_text)

        target_objects = parse_gpt_output(response_text)

        if not target_objects:
            print(f"[WARN] Episode {episode_id}: MLLM returned no target objects. "
                  f"Returning zero mask to avoid downstream errors.")
            h, w = query_img.shape[-2:]  # query_img is [3, H, W] at this point
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            return FSSSResult(
                episode_id=episode_id,
                response_text=response_text,
                target_objects=[],
                masks_info=[{"object_index": 0, "description": "no objects found", "mask": zero_mask}],
            )

        seg_args = type("SegArgs", (), {})()
        seg_args.output_dir = episode_dir if self.save_intermediates else self.output_dir
        seg_args.segzero_model = ""
        seg_args.sam2_model = ""
        seg_args.sam2_model_cfg = ""

        save_prefix = f"query_mask_obj_{episode_id}_"
        # mask_info 包含每个目标的分割结果和相关信息，格式为 [{'object_description': str, 'mask': np.ndarray}, ...]
        masks_info = seg_query_objects(
            args=seg_args,
            reasoning_model=self.reasoning_model,
            segmentation_model=self.segmentation_model,
            processor=self.segzero_processor,
            query_img=query_pil,
            target_objects=target_objects,
            save_mask=self.save_intermediates,
            save_name=save_prefix,
        )

        return FSSSResult(
            episode_id=episode_id,
            # class_name=episode.class_name,
            response_text=response_text,
            target_objects=target_objects,
            masks_info=masks_info,
        )
