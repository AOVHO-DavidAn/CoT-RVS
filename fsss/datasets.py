from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@dataclass
class EpisodeSample:
    dataset: str
    class_name: str
    class_id: Optional[int]
    support_image_paths: List[str]
    support_masks: List[np.ndarray]
    query_image_path: str
    query_id: str

    def load_support_images(self) -> List[Image.Image]:
        return [Image.open(path).convert("RGB") for path in self.support_image_paths]

    def load_query_image(self) -> Image.Image:
        return Image.open(self.query_image_path).convert("RGB")


class FSS1000Dataset:
    def __init__(self, root: str) -> None:
        self.root = root
        self.class_names = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        )

    def sample_episode(
        self,
        k_shot: int,
        query_count: int,
        rng: random.Random,
        class_name: Optional[str] = None,
    ) -> EpisodeSample:
        if query_count != 1:
            raise ValueError("Only query_count=1 is supported right now.")
        if class_name is None:
            class_name = rng.choice(self.class_names)
        class_dir = os.path.join(self.root, class_name)
        image_names = sorted([f for f in os.listdir(class_dir) if f.endswith(".jpg")])
        if len(image_names) < k_shot + query_count:
            raise RuntimeError(
                f"Not enough images for class {class_name}: {len(image_names)}"
            )
        selected = rng.sample(image_names, k_shot + query_count)
        support_names = selected[:k_shot]
        query_name = selected[-1]

        support_paths = [os.path.join(class_dir, name) for name in support_names]
        support_masks = []
        for name in support_names:
            mask_path = os.path.join(class_dir, name.replace(".jpg", ".png"))
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            support_masks.append(mask)

        query_path = os.path.join(class_dir, query_name)
        query_id = os.path.splitext(query_name)[0]

        return EpisodeSample(
            dataset="fss-1000",
            class_name=class_name,
            class_id=None,
            support_image_paths=support_paths,
            support_masks=support_masks,
            query_image_path=query_path,
            query_id=query_id,
        )


class Pascal5iDataset:
    def __init__(self, root: str, fold: int, split: str) -> None:
        if fold not in {0, 1, 2, 3}:
            raise ValueError("fold must be one of {0,1,2,3} for PASCAL-5i.")
        self.root = root
        self.fold = fold
        self.split = split
        self.mask_dir = self._resolve_mask_dir()
        self.image_dir = os.path.join(root, "JPEGImages")
        self.image_sets_dir = os.path.join(root, "ImageSets", "Segmentation")
        self._class_cache: Dict[int, List[str]] = {}

    def _resolve_mask_dir(self) -> str:
        aug_dir = os.path.join(self.root, "SegmentationClassAug")
        if os.path.isdir(aug_dir):
            return aug_dir
        return os.path.join(self.root, "SegmentationClass")

    def _load_image_ids(self) -> List[str]:
        split_file = os.path.join(self.image_sets_dir, f"{self.split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _class_candidates(self) -> List[int]:
        if self.split.lower() in {"val", "test"}:
            return [i + 1 for i, _ in enumerate(PASCAL_CLASSES) if i % 4 == self.fold]
        return [i + 1 for i, _ in enumerate(PASCAL_CLASSES) if i % 4 != self.fold]

    def _get_images_for_class(self, class_id: int, image_ids: List[str]) -> List[str]:
        if class_id in self._class_cache:
            return self._class_cache[class_id]
        valid_ids = []
        for image_id in image_ids:
            mask_path = os.path.join(self.mask_dir, f"{image_id}.png")
            if not os.path.exists(mask_path):
                continue
            mask = np.array(Image.open(mask_path))
            if (mask == class_id).any():
                valid_ids.append(image_id)
        self._class_cache[class_id] = valid_ids
        return valid_ids

    def sample_episode(
        self,
        k_shot: int,
        query_count: int,
        rng: random.Random,
        class_name: Optional[str] = None,
    ) -> EpisodeSample:
        if query_count != 1:
            raise ValueError("Only query_count=1 is supported right now.")
        image_ids = self._load_image_ids()

        if class_name is None:
            candidates = self._class_candidates()
            class_id = rng.choice(candidates)
            class_name = PASCAL_CLASSES[class_id - 1]
        else:
            if class_name not in PASCAL_CLASSES:
                raise ValueError(f"Unknown PASCAL class: {class_name}")
            class_id = PASCAL_CLASSES.index(class_name) + 1

        valid_ids = self._get_images_for_class(class_id, image_ids)
        if len(valid_ids) < k_shot + query_count:
            raise RuntimeError(
                f"Not enough images for class {class_name}: {len(valid_ids)}"
            )
        chosen = rng.sample(valid_ids, k_shot + query_count)
        support_ids = chosen[:k_shot]
        query_id = chosen[-1]

        support_paths = [os.path.join(self.image_dir, f"{i}.jpg") for i in support_ids]
        support_masks = []
        for image_id in support_ids:
            mask_path = os.path.join(self.mask_dir, f"{image_id}.png")
            mask = np.array(Image.open(mask_path)) == class_id
            support_masks.append(mask)

        query_path = os.path.join(self.image_dir, f"{query_id}.jpg")

        return EpisodeSample(
            dataset="pascal",
            class_name=class_name,
            class_id=class_id,
            support_image_paths=support_paths,
            support_masks=support_masks,
            query_image_path=query_path,
            query_id=query_id,
        )


class CocoFSSDataset:
    def __init__(self, root: str, split: str, ann_path: Optional[str] = None) -> None:
        self.root = root
        self.split = split
        if ann_path is None:
            ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"COCO annotation not found: {ann_path}")
        self.coco = COCO(ann_path)
        self.image_dir = os.path.join(root, split)
        self.cat_name_to_id = {
            cat["name"]: cat["id"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

    def _build_mask(self, img_id: int, cat_id: int) -> np.ndarray:
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[cat_id])
        anns = self.coco.loadAnns(ann_ids)
        if not anns:
            raise RuntimeError("No annotations found for COCO support image.")
        mask = None
        for ann in anns:
            ann_mask = self.coco.annToMask(ann).astype(bool)
            mask = ann_mask if mask is None else (mask | ann_mask)
        return mask

    def sample_episode(
        self,
        k_shot: int,
        query_count: int,
        rng: random.Random,
        class_name: Optional[str] = None,
    ) -> EpisodeSample:
        if query_count != 1:
            raise ValueError("Only query_count=1 is supported right now.")
        if class_name is None:
            cat_id = rng.choice(self.coco.getCatIds())
            class_name = self.coco.loadCats([cat_id])[0]["name"]
        else:
            if class_name not in self.cat_name_to_id:
                raise ValueError(f"Unknown COCO class: {class_name}")
            cat_id = self.cat_name_to_id[class_name]

        img_ids = self.coco.getImgIds(catIds=[cat_id])
        if len(img_ids) < k_shot + query_count:
            raise RuntimeError(
                f"Not enough images for class {class_name}: {len(img_ids)}"
            )
        chosen = rng.sample(img_ids, k_shot + query_count)
        support_ids = chosen[:k_shot]
        query_id = chosen[-1]

        support_paths = []
        support_masks = []
        for img_id in support_ids:
            img_info = self.coco.loadImgs([img_id])[0]
            support_paths.append(os.path.join(self.image_dir, img_info["file_name"]))
            support_masks.append(self._build_mask(img_id, cat_id))

        query_info = self.coco.loadImgs([query_id])[0]
        query_path = os.path.join(self.image_dir, query_info["file_name"])

        return EpisodeSample(
            dataset="coco",
            class_name=class_name,
            class_id=cat_id,
            support_image_paths=support_paths,
            support_masks=support_masks,
            query_image_path=query_path,
            query_id=str(query_id),
        )
