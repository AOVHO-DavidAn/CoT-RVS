import argparse
import os
import random
import numpy as np
import torch

from fsss.datasets import CocoFSSDataset, FSS1000Dataset, Pascal5iDataset
from fsss.pipeline import FSSSPipeline

from fsss.data.dataset import FSSDataset
from fsss.common.logger import Logger, AverageMeter
from fsss.common.vis import Visualizer
from fsss.common.evaluation import Evaluator
from fsss.common import utils
from utils.util import save_masked_image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated FSSS pipeline")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="fss-1000",
        choices=["fss-1000", "pascal", "coco"],
    )
    parser.add_argument("--datapath", type=str, default="./datasets")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    # parser.add_argument('--fold', type=int, default=0)
    # parser.add_argument("--class_name", type=str, default=None)
    # parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--nshot", type=int, default=1)
    # parser.add_argument("--query_count", type=int, default=1)
    parser.add_argument('--img-size', type=int, default=1024)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='./outputs/logs')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel_height", type=int, default=512)
    parser.add_argument("--save_intermediates", action="store_true", default=True)
    parser.add_argument('--visualize', type=int, default=0)


    parser.add_argument("--gemma_model_name", type=str, default="Gemma-4-26B-A4B-it")
    parser.add_argument(
        "--gemma_model_path",
        type=str,
        default="./checkpoints/gemma_weights",
    )
    parser.add_argument(
        "--sam2_model",
        type=str,
        default="./checkpoints/sam_weights/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--sam2_model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--segzero_model",
        type=str,
        default="checkpoints/seg-zero_weights-cache/models--Ricky06662--Seg-Zero-7B/snapshots/7764ee846ff4d68e7c65ede1e63531302f63860a",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/fsss_pipeline")
    parser.add_argument("--coco_ann", type=str, default=None)

    return parser.parse_args()


def test(pipeline, dataloader, args=None):
    r""" Test CoT-RVS on the given dataloader, return mIoU and FB-IoU. """

    # Freeze randomness during testing for reproducibility
    # Follow HSNet
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        query_img, query_mask, support_imgs, support_masks = \
            batch['query_img'], batch['query_mask'], \
            batch['support_imgs'], batch['support_masks']

        # print(query_img.size(), query_mask.size(), support_imgs.size(), support_masks.size())

        # 1. 运行 FSSS pipeline（内部已完成 Tensor→PIL 转换）
        result = pipeline.run_episode(query_img, support_imgs, support_masks, args.nshot, idx)
        masks_info = result.masks_info
        object_masks = [info['mask'] for info in masks_info]
        pred_mask = utils.merge_masks(object_masks)

        if pred_mask is None:
            Logger.info(f"Episode {idx}: no valid masks produced, falling back to zero mask.")
            pred_mask = np.zeros((query_img.shape[-2], query_img.shape[-1]), dtype=np.uint8)
        
        if args.save_intermediates:
            # 保存每个 episode 的预测 mask 以及与 GT 的叠加可视化结果
            save_dir = os.path.join(args.output_dir, args.benchmark, f"episode_{idx}")
            os.makedirs(save_dir, exist_ok=True)
            save_masked_image(query_img, pred_mask, query_mask, save_dir, f"episode_{idx}_pred")
        
        pred_mask = torch.from_numpy(pred_mask).to(query_img.device)
        pred_mask = pred_mask.unsqueeze(0)  # [1, H, W]

        
        assert pred_mask.size() == batch['query_mask'].size(), \
            'pred {} ori {}'.format(pred_mask.size(), batch['query_mask'].size())

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        print(area_inter, area_union)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, _ = average_meter.compute_iou()

    return miou, fb_iou


# def build_dataset(args: argparse.Namespace):
#     if args.dataset == "fss-1000":
#         return FSS1000Dataset(os.path.join(args.dataset_root, "FSS-1000"))
#     if args.dataset == "pascal":
#         return Pascal5iDataset(
#             os.path.join(args.dataset_root, "PASCAL"),
#             fold=args.fold,
#             split=args.split,
#         )
#     if args.dataset == "coco":
#         split = args.split
#         if split in {"train", "train2017"}:
#             split = "train2017"
#         elif split in {"val", "val2017"}:
#             split = "val2017"
#         return CocoFSSDataset(
#             os.path.join(args.dataset_root, "coco"),
#             split=split,
#             ann_path=args.coco_ann,
#         )
#     raise ValueError(f"Unsupported dataset: {args.dataset}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    test_dataloader = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    gemma_model_path = f"{args.gemma_model_path}/{args.gemma_model_name}"

    # dataset = build_dataset(args)
    pipeline = FSSSPipeline(
        gemma_model_name=args.gemma_model_name,
        gemma_model_path=gemma_model_path,
        sam2_model=args.sam2_model,
        sam2_model_cfg=args.sam2_model_cfg,
        segzero_model=args.segzero_model,
        output_dir=args.output_dir,
        panel_height=args.panel_height,
        save_intermediates=args.save_intermediates,
        benchmark=args.benchmark,
    )
    
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)


    with torch.no_grad():
        test_miou, test_fb_iou = test(pipeline, test_dataloader, args=args)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
    # for idx in range(args.num_episodes):
    #     episode = dataset.sample_episode(
    #         k_shot=args.k_shot,
    #         query_count=args.query_count,
    #         rng=rng,
    #         class_name=args.class_name,
    #     )
    #     episode_id = f"{args.dataset}_{episode.class_name}_{idx}"
    #     result = pipeline.run_episode(episode, args.k_shot, episode_id)
    #     print(
    #         f"Episode {episode_id}: class={result.class_name}, "
    #         f"objects={len(result.target_objects)}"
    #     )


if __name__ == "__main__":
    main()
