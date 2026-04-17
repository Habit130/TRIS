import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from args import get_parser
from dataset.plantseg_dataset import PlantSegDataset
from dataset.transform import get_transform
from model.model_stage2 import TRIS
from utils.util import load_checkpoint, load_pretrained_checkpoint


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_metrics(tp, fp, fn, tn):
    iou_fg = _safe_div(tp, tp + fp + fn)
    iou_bg = _safe_div(tn, tn + fp + fn)
    recall_fg = _safe_div(tp, tp + fn)
    recall_bg = _safe_div(tn, tn + fp)
    dice_fg = _safe_div(2 * tp, 2 * tp + fp + fn)

    return {
        "IoU": 100.0 * iou_fg,
        "Dice": 100.0 * dice_fg,
        "Recall": 100.0 * recall_fg,
        "mIoU": 100.0 * ((iou_fg + iou_bg) / 2.0),
        "mACC": 100.0 * ((recall_fg + recall_bg) / 2.0),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def validate_plantseg(args, data_loader, model, local_rank=0, save_pred_masks=False):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    if save_pred_masks:
        os.makedirs(args.mask_output_dir, exist_ok=True)

    with torch.no_grad():
        for samples, targets in data_loader:
            img = samples["img"].cuda(local_rank, non_blocking=True)
            word_ids = samples["word_ids"].squeeze(1).cuda(local_rank, non_blocking=True)
            target = targets["target"].cuda(local_rank, non_blocking=True)
            orig_size = targets["orig_size"][0].tolist()

            output = model(img, word_ids)
            output = F.interpolate(
                output,
                (orig_size[0], orig_size[1]),
                align_corners=False,
                mode="bilinear",
            )
            pred = (torch.sigmoid(output) >= 0.5).to(dtype=torch.int64)
            target = (target > 0).to(dtype=torch.int64)

            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            total_tp += int(torch.sum((pred_flat == 1) & (target_flat == 1)).item())
            total_fp += int(torch.sum((pred_flat == 1) & (target_flat == 0)).item())
            total_fn += int(torch.sum((pred_flat == 0) & (target_flat == 1)).item())
            total_tn += int(torch.sum((pred_flat == 0) & (target_flat == 0)).item())

            if save_pred_masks:
                pred_mask = (pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)) * 255
                mask_name = os.path.basename(targets["mask_path_full"][0])
                from PIL import Image

                Image.fromarray(pred_mask, mode="L").save(os.path.join(args.mask_output_dir, mask_name))

    return _compute_metrics(total_tp, total_fp, total_fn, total_tn)


def build_dataloader(args, split):
    dataset = PlantSegDataset(
        root=args.plantseg_root,
        split=split,
        image_transforms=get_transform(args.size, train=False),
        max_tokens=args.max_query_len,
        caption_index=args.caption_index,
        eval_mode=True,
        size=args.size,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
    )


def main(args):
    model = TRIS(args).cuda()
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    if args.resume and args.pretrain is not None:
        load_checkpoint(args, model_without_ddp)
    elif args.pretrained_checkpoint is not None:
        load_pretrained_checkpoint(args.pretrained_checkpoint, model_without_ddp)
    else:
        raise ValueError("PlantSeg validation requires either --resume --pretrain or --pretrained_checkpoint")

    all_metrics = {}
    for split in args.test_split.split(","):
        data_loader = build_dataloader(args, split)
        should_save_masks = args.save_pred_masks and split == "test"
        metrics = validate_plantseg(
            args,
            data_loader,
            model,
            local_rank=0,
            save_pred_masks=should_save_masks,
        )
        all_metrics[split] = metrics
        print(f"[{split}] " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items() if key in {"IoU", "Dice", "Recall", "mIoU", "mACC"}))

    if args.metrics_output is not None:
        os.makedirs(os.path.dirname(args.metrics_output), exist_ok=True)
        summary = {
            "dataset": args.dataset,
            "test_split": args.test_split,
            "checkpoint": args.pretrain if args.pretrain is not None else args.pretrained_checkpoint,
            "metrics": all_metrics,
        }
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
