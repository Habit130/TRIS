import datetime
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ema_pytorch import EMA
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from args import get_parser
from dataset.ReferDataset import ReferDataset
from dataset.plantseg_dataset import PlantSegDataset
from dataset.transform import get_transform
from logger import create_logger
from model.model_stage2 import TRIS, criterion
from utils.official_assets import ensure_official_tris_checkpoint
from utils.poly_lr_decay import PolynomialLRDecay
from utils.util import (
    AverageMeter,
    load_checkpoint,
    load_pretrained_checkpoint,
    reduce_tensor,
    save_checkpoint,
)
from validate import validate as validate_refer
from validate_plantseg import validate_plantseg

writer = None


def build_train_dataset(args):
    if args.dataset == "plantseg":
        return PlantSegDataset(
            root=args.plantseg_root,
            split="train",
            image_transforms=get_transform(args.size, train=True),
            max_tokens=args.max_query_len,
            caption_index=args.caption_index,
            eval_mode=False,
            size=args.size,
        )

    return ReferDataset(
        refer_data_root=args.refer_data_root,
        dataset=args.dataset,
        split="train",
        splitBy=args.splitBy,
        image_transforms=get_transform(args.size, train=True),
        eval_mode=False,
        size=args.size,
        bert_tokenizer=args.bert_tokenizer,
        pseudo_path=args.pseudo_path,
    )


def build_eval_dataset(args, split):
    if args.dataset == "plantseg":
        return PlantSegDataset(
            root=args.plantseg_root,
            split=split,
            image_transforms=get_transform(args.size, train=False),
            max_tokens=args.max_query_len,
            caption_index=args.caption_index,
            eval_mode=True,
            size=args.size,
        )

    return ReferDataset(
        refer_data_root=args.refer_data_root,
        dataset=args.dataset,
        split=split,
        splitBy=args.splitBy,
        image_transforms=get_transform(args.size, train=False),
        eval_mode=True,
        size=args.size,
        bert_tokenizer=args.bert_tokenizer,
    )


def maybe_prepare_official_checkpoint(args):
    if args.dataset != "plantseg":
        return
    if args.resume or args.pretrained_checkpoint is not None:
        return
    args.pretrained_checkpoint = ensure_official_tris_checkpoint(
        "stage2_refcocog_umd.pth",
        args.official_weights_dir,
    )


def run_validation(args, data_loader, model, local_rank=0):
    if args.dataset == "plantseg":
        return validate_plantseg(args, data_loader, model, local_rank=local_rank, save_pred_masks=False)

    oIoU, mIoU, hit = validate_refer(args, data_loader, model, local_rank)
    return {
        "oIoU": float(oIoU),
        "mIoU": float(mIoU),
        "hit": float(hit),
    }


def log_validation_metrics(epoch, split, metrics):
    if writer is None:
        return

    if "mIoU" in metrics:
        writer.add_scalar(f"{split}/mIoU", metrics["mIoU"], epoch)
    if "oIoU" in metrics:
        writer.add_scalar(f"{split}/oIoU", metrics["oIoU"], epoch)
    if "hit" in metrics:
        writer.add_scalar(f"{split}/hit", metrics["hit"], epoch)
    if "IoU" in metrics:
        writer.add_scalar(f"{split}/IoU_fg", metrics["IoU"], epoch)
    if "Dice" in metrics:
        writer.add_scalar(f"{split}/Dice_fg", metrics["Dice"], epoch)
    if "Recall" in metrics:
        writer.add_scalar(f"{split}/Recall_fg", metrics["Recall"], epoch)
    if "mACC" in metrics:
        writer.add_scalar(f"{split}/mACC", metrics["mACC"], epoch)


def main(args):
    maybe_prepare_official_checkpoint(args)

    if args.distributed:
        local_rank = dist.get_rank()
    else:
        local_rank = 0

    model = TRIS(args)
    try:
        param_groups = model.trainable_parameters()
    except Exception:
        param_groups = None
        print("no param groups...")

    if args.distributed:
        model.cuda(local_rank)
    else:
        model.cuda()

    if args.model_ema:
        model_ema = EMA(model)
        print("ema initialized !")
    else:
        model_ema = None
        print("no ema")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)

    model_without_ddp = model.module
    if local_rank == 0:
        print()
        print(model_without_ddp)
        print()
    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {num_params/1e6:.2f}M")

    train_dataset = build_train_dataset(args)
    val_datasets = [build_eval_dataset(args, split) for split in args.test_split.split(",")]

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_samplers = [DistributedSampler(dataset, shuffle=False) for dataset in val_datasets]
    else:
        train_sampler = None
        val_samplers = [None for _ in val_datasets]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )
    val_loaders = [
        DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
        )
        for dataset, sampler in zip(val_datasets, val_samplers)
    ]

    if param_groups is not None:
        optimizer = AdamW(
            [
                {
                    "params": param_groups[0],
                    "lr": args.lr * args.lr_multi,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
            ],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(train_loader) * args.epoch)) ** 0.9,
        )
    else:
        optimizer = AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = PolynomialLRDecay(
            optimizer,
            max_decay_steps=args.max_decay_steps,
            end_learning_rate=args.end_lr,
            power=args.power,
        )

    print()
    print(optimizer)
    print(scheduler)
    print()

    if args.resume:
        load_checkpoint(args, model_without_ddp, optimizer, scheduler, logger)
        if args.eval:
            for split, val_loader in zip(args.test_split.split(","), val_loaders):
                metrics = run_validation(args, val_loader, model, local_rank)
                print(f"[{split}] {metrics}")
            return

    if args.pretrained_checkpoint is not None:
        print("loading ", args.pretrained_checkpoint)
        load_pretrained_checkpoint(args.pretrained_checkpoint, model_without_ddp)

    logger.info("Start training")

    train_time = 0
    start_time = time.time()
    primary_split = args.test_split.split(",")[0]
    best = {
        "val_acc": -1.0,
        "epoch": -1,
        "path": "",
        "metrics": {},
    }
    iteration = 0
    for epoch in range(args.start_epoch, args.epoch):
        st = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        iteration = train_one_epoch(train_loader, model, optimizer, epoch, local_rank, args, iteration, model_ema)
        scheduler.step()

        train_time += time.time() - st

        epoch_metrics = {}
        for split, val_loader in zip(args.test_split.split(","), val_loaders):
            metrics = run_validation(args, val_loader, model, local_rank)
            epoch_metrics[split] = metrics
            log_validation_metrics(epoch, split, metrics)
            print(f"[epoch {epoch}] [{split}] {metrics}")

        val_acc = float(epoch_metrics[primary_split]["mIoU"])
        if val_acc > best["val_acc"] and local_rank == 0:
            if os.path.exists(best["path"]):
                print("remove ", best["path"])
                os.remove(best["path"])
            save_path = save_checkpoint(
                epoch,
                model_without_ddp,
                optimizer,
                scheduler,
                logger,
                args,
                f"ckpt_320_epoch_{epoch}.pth",
            )
            best_alias = os.path.join(args.output, "best_model.pth")
            shutil.copyfile(save_path, best_alias)
            best["val_acc"] = val_acc
            best["epoch"] = epoch
            best["path"] = best_alias
            best["metrics"] = epoch_metrics
        print(best)

    if args.dataset != "plantseg":
        print()
        print()
        last_trainset = build_eval_dataset(args, "train")
        val_train_loader = DataLoader(
            last_trainset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            sampler=None,
            shuffle=False,
        )
        print("loading ", best["path"])
        load_pretrained_checkpoint(best["path"], model_without_ddp)
        train_metrics = run_validation(args, val_train_loader, model, local_rank)
        print("Validation on the train split: ", train_metrics)
        print()
        print(best)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(train_time))
    logger.info("Training + testing time {}".format(total_time_str))


def sigmoid_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_sigmoid = F.sigmoid(input_logits)
    target_sigmoid = F.sigmoid(target_logits)

    return F.mse_loss(input_sigmoid, target_sigmoid, reduction="mean")


def train_one_epoch(train_loader, model, optimizer, epoch, local_rank, args, iteration=0, model_ema=None):
    num_steps = len(train_loader)
    model.train()
    optimizer.zero_grad()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    max_iter = int(num_steps * args.epoch)

    if args.consistency_type == "mse":
        consistency_criterion = sigmoid_mse_loss
    elif args.consistency_type == "kl":
        consistency_criterion = F.kl_div
    else:
        raise NotImplementedError(f"Unknown consistency type: {args.consistency_type}")

    for idx, (samples, targets) in enumerate(train_loader):
        word_ids = samples["word_ids"].squeeze(1)
        img = samples["img"].cuda(local_rank, non_blocking=True)
        word_ids = word_ids.cuda(local_rank, non_blocking=True)
        pseudo = targets["pseudo_gt"].cuda(local_rank, non_blocking=True)

        output1, output2, output3, output4 = model(img, word_ids)
        if args.model_ema:
            with torch.no_grad():
                ema_output1, ema_output2, ema_output3, ema_output4 = model_ema(img, word_ids)
            ema_loss = 0
            ema_loss += consistency_criterion(output1, ema_output1)
            ema_loss += consistency_criterion(output2, ema_output2)
            ema_loss += consistency_criterion(output3, ema_output3)
            ema_loss += consistency_criterion(output4, ema_output4)
            l5 = ema_loss
        else:
            l5 = torch.tensor(0.0, device=img.device)

        l1 = criterion(output1, pseudo)
        l2 = criterion(output2, pseudo)
        l3 = criterion(output3, pseudo)
        l4 = criterion(output4, pseudo)

        loss = l1 + l2 + l3 + l4 + l5

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            try:
                model_ema.update(model)
            except Exception:
                model_ema.update()

        if args.distributed:
            reduce_tensor(args, l1)
            reduce_tensor(args, l2)
            reduce_tensor(args, l3)
            reduce_tensor(args, l4)
            reduce_tensor(args, l5)
            reduce_tensor(args, loss)

        if local_rank == 0 and writer is not None:
            lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("optim/lr", lr, iteration)
            writer.add_scalar("train/l1", l1.data.cpu().numpy(), iteration)
            writer.add_scalar("train/l2", l2.data.cpu().numpy(), iteration)
            writer.add_scalar("train/l3", l3.data.cpu().numpy(), iteration)
            writer.add_scalar("train/l4", l4.data.cpu().numpy(), iteration)
            writer.add_scalar("train/l5", l5.data.cpu().numpy(), iteration)
            writer.add_scalar("train/loss", loss.data.cpu().numpy(), iteration)

        loss_meter.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0 and local_rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            all_etas = batch_time.avg * (max_iter - iteration)
            print(
                f"Train:[{epoch:2d}/{args.epoch}][{idx:4d}/{num_steps}] | "
                f"eta: {datetime.timedelta(seconds=int(etas))} | lr {lr:.6f} || "
                f"loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"l1: {l1:.4f} | "
                f"l2: {l2:.4f} | "
                f"l3: {l3:.4f} | "
                f"l4: {l4:.4f} | "
                f"l5: {l5:.4f} | "
                f"time: {batch_time.val:.4f} ({batch_time.avg:.4f}) | "
                f"mem: {memory_used:.0f}MB || "
                f"all_eta: {datetime.timedelta(seconds=int(all_etas))}",
                flush=True,
            )
        iteration += 1
    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}", flush=True)
    return iteration


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parse = get_parser()
    args = parse.parse_args()
    if args.vis_out is not None and not os.path.exists(args.vis_out):
        os.mkdir(args.vis_out)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    print("=" * 100)
    print(args)
    print("=" * 100)

    if args.distributed:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.distributed.barrier()

    if args.distributed:
        logger = create_logger(dist_rank=dist.get_rank())
    else:
        logger = create_logger()

    if args.board_folder is not None:
        writer = SummaryWriter(args.board_folder)

    main(args)
