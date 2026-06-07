import argparse
import os
import sys
import time

import torch
from transformers import Gemma4ForConditionalGeneration, AutoProcessor, Gemma3ForConditionalGeneration

from infer.prompt_gemma_fsss import prompt_gemma, save_answer


def parse_args(args):
    parser = argparse.ArgumentParser(description="CoT-RVS-Gemma FSSS offline test")
    parser.add_argument("--gemma_model_name", default="Gemma-3-12B-it", type=str)
    parser.add_argument("--model_path", default="./checkpoints/gemma_weights/gemma-3-12b-it", type=str)
    parser.add_argument("--image_path", default="./outputs/merged_episode/test_input.jpg", type=str)
    parser.add_argument("--k_shot", default=1, type=int)
    parser.add_argument("--output_dir", default="./vis_output", type=str)
    parser.add_argument("--save_name", default="test_sample_fsss", type=str)
    parser.add_argument("--task_note", default="FSSS offline merged image", type=str)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.makedirs(os.path.join(args.output_dir, f"{args.gemma_model_name}-responses"), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if not os.path.exists(args.image_path):
        raise RuntimeError(f"Image path not found: {args.image_path}")

    gemma_model_id = args.model_path
    model_name = args.gemma_model_name.strip().lower()

    if model_name == "gemma-3-12b-it":
        gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_model_id, device_map="auto"
        ).eval()
    elif model_name in {"gemma-4-e4b-it", "gemma-4-26b-a4b-it"}:
        gemma_model = Gemma4ForConditionalGeneration.from_pretrained(
            gemma_model_id, device_map="auto"
        ).eval()
    else:
        raise ValueError(
            f"Unsupported gemma model name: {args.gemma_model_name}. "
            "Expected gemma-3-12b-it, gemma-4-e4b-it, or gemma-4-26b-a4b-it."
        )

    gemma_processor = AutoProcessor.from_pretrained(gemma_model_id)

    # Full pipeline example (disabled for offline test use):
    # from merge_episode import create_mllm_test_image
    # support_path = "./few-shot_episode/support_img.jpg"
    # mask_path = "./few-shot_episode/support_mask.png"
    # query_path = "./few-shot_episode/query_img.jpg"
    # merged_path = create_mllm_test_image(
    #     support_path, mask_path, query_path, "./outputs/merged_episode", "test_input.jpg"
    # )
    # args.image_path = merged_path

    print("Using merged image:", args.image_path)
    print(f"Waiting for {args.gemma_model_name} response...")

    start_time = time.perf_counter()
    answer = prompt_gemma(
        model=gemma_model,
        processor=gemma_processor,
        image=args.image_path,
        k_shot=args.k_shot,
    )
    inference_time = time.perf_counter() - start_time
    print(f"Inference time: {inference_time:.2f} seconds")

    save_answer(
        args.task_note,
        answer,
        os.path.join(args.output_dir, f"{args.gemma_model_name}-responses", f"{args.save_name}.txt"),
        model_name=args.gemma_model_name,
        inference_time=inference_time,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
