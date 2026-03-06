import argparse
import logging
import math
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, ramps
from val_2D import test_single_volume_ds


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="../data/ACDC", help="dataset path")
parser.add_argument(
    "--exp",
    type=str,
    default="ACDC/URPC_DynamicWeighted_Consistency",
    help="experiment name",
)
parser.add_argument("--model", type=str, default="unet_urpc", help="model name")
parser.add_argument("--max_iterations", type=int, default=30000, help="maximum iteration number")
parser.add_argument("--batch_size", type=int, default=24, help="batch size per gpu")
parser.add_argument("--deterministic", type=int, default=1, help="use deterministic training")
parser.add_argument("--base_lr", type=float, default=0.01, help="segmentation learning rate")
parser.add_argument("--patch_size", type=list, default=[256, 256], help="patch size")
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--num_classes", type=int, default=4, help="number of classes")

# labeled / unlabeled
parser.add_argument("--labeled_bs", type=int, default=12, help="labeled batch size")
parser.add_argument("--labeled_num", type=int, default=7, help="number of labeled patients")

# consistency
parser.add_argument("--consistency", type=float, default=0.1, help="maximum consistency weight")
parser.add_argument("--consistency_rampup", type=float, default=200.0, help="consistency rampup")

# confidence masking
parser.add_argument("--conf_thresh", type=float, default=0.7, help="final confidence threshold")
parser.add_argument("--conf_rampup", type=float, default=50.0, help="confidence threshold rampup")

# dynamic weighting
parser.add_argument("--sup_decay", type=float, default=0.3, help="how much supervised weight decays")
parser.add_argument("--feat_ramp_multiplier", type=float, default=2.0, help="feature ramp-up slower than output consistency")

# uncertainty penalty
parser.add_argument("--uncertainty_penalty", type=float, default=0.1, help="weight for variance penalty")

# optional feature consistency
parser.add_argument("--feat_weight_max", type=float, default=0.05, help="maximum feature consistency weight")
args = parser.parse_args()


def patients_to_slices(dataset, patients_num):
    if "ACDC" in dataset:
        ref_dict = {
            "3": 68,
            "7": 136,
            "14": 256,
            "21": 396,
            "28": 512,
            "35": 664,
            "140": 1312,
        }
    elif "Prostate" in dataset:
        ref_dict = {
            "2": 27,
            "4": 53,
            "8": 120,
            "12": 179,
            "16": 256,
            "21": 312,
            "42": 623,
        }
    else:
        raise ValueError("Unknown dataset for patients_to_slices")
    return ref_dict[str(patients_num)]


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_conf_threshold(epoch):
    start = 0.5
    end = float(args.conf_thresh)
    ramp = ramps.sigmoid_rampup(epoch, args.conf_rampup)
    return start + (end - start) * ramp


def get_supervised_weight(epoch, max_epoch):
    progress = min(epoch / max_epoch, 1.0)
    return 1.0 - args.sup_decay * progress


def get_output_consistency_scale(epoch):
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_feature_consistency_scale(epoch):
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup * args.feat_ramp_multiplier)


def normalize_feature(feat):
    return F.normalize(feat, p=2, dim=1)


def extract_outputs_and_feats(model_out):
    """
    Compatible with:
    1) old URPC model: returns 4 outputs
    2) extended model: returns 4 outputs + extra feature tensors
    """
    if not isinstance(model_out, (tuple, list)):
        raise ValueError("Model output must be tuple/list for URPC training")

    if len(model_out) < 4:
        raise ValueError("Model must return at least 4 outputs: main + 3 aux heads")

    outputs = model_out[:4]
    feats = list(model_out[4:]) if len(model_out) > 4 else []
    return outputs, feats


def compute_feature_loss(feats):
    """
    Optional feature consistency.
    If model returns >= 2 feature tensors, match adjacent feature maps.
    Works as a lightweight regularizer and does not break old models.
    """
    if feats is None or len(feats) < 2:
        return None

    norm_feats = [normalize_feature(f) for f in feats]
    feat_loss = 0.0
    pair_count = 0

    for i in range(len(norm_feats) - 1):
        f1 = norm_feats[i]
        f2 = norm_feats[i + 1]

        # align spatial size if needed
        if f1.shape[-2:] != f2.shape[-2:]:
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        # align channels if needed by slicing to min channel count
        c = min(f1.shape[1], f2.shape[1])
        feat_loss = feat_loss + F.mse_loss(f1[:, :c], f2[:, :c])
        pair_count += 1

    if pair_count == 0:
        return None
    return feat_loss / pair_count


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
    model.train()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    )
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print(f"Total slices: {total_slices}, labeled slices: {labeled_slice}")

    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3]).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction="none")

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info(f"{len(trainloader)} iterations per epoch")

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=80)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model_out = model(volume_batch)
            outputs_list, feats = extract_outputs_and_feats(model_out)
            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = outputs_list

            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

            # ---------------- supervised loss ----------------
            lb = args.labeled_bs
            label_l = label_batch[:lb]

            loss_ce = ce_loss(outputs[:lb], label_l.long())
            loss_ce_aux1 = ce_loss(outputs_aux1[:lb], label_l.long())
            loss_ce_aux2 = ce_loss(outputs_aux2[:lb], label_l.long())
            loss_ce_aux3 = ce_loss(outputs_aux3[:lb], label_l.long())

            loss_dice = dice_loss(outputs_soft[:lb], label_l.unsqueeze(1))
            loss_dice_aux1 = dice_loss(outputs_aux1_soft[:lb], label_l.unsqueeze(1))
            loss_dice_aux2 = dice_loss(outputs_aux2_soft[:lb], label_l.unsqueeze(1))
            loss_dice_aux3 = dice_loss(outputs_aux3_soft[:lb], label_l.unsqueeze(1))

            ce_sum = loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3
            dice_sum = loss_dice + loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3

            supervised_loss = (0.2 * ce_sum + 0.8 * dice_sum) / 4.0

            # ---------------- unlabeled pseudo target ----------------
            preds = (
                outputs_soft + outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft
            ) / 4.0
            preds = preds.detach()  # pseudo target as target, not source of unstable gradients

            # unlabeled part
            preds_u = preds[lb:]
            outputs_u = outputs_soft[lb:]
            outputs_aux1_u = outputs_aux1_soft[lb:]
            outputs_aux2_u = outputs_aux2_soft[lb:]
            outputs_aux3_u = outputs_aux3_soft[lb:]

            # ---------------- confidence mask ----------------
            with torch.no_grad():
                conf_map = torch.max(preds_u, dim=1, keepdim=True)[0]
                conf_thresh = get_current_conf_threshold(epoch_num)
                conf_mask = (conf_map > conf_thresh).float()
                mask_mean = conf_mask.mean()
                norm_factor = mask_mean + 1e-6

            # ---------------- variance-aware weighting ----------------
            variance_main = torch.sum(
                kl_distance(torch.log(outputs_u + 1e-8), preds_u),
                dim=1,
                keepdim=True,
            )
            exp_variance_main = torch.exp(-variance_main)

            variance_aux1 = torch.sum(
                kl_distance(torch.log(outputs_aux1_u + 1e-8), preds_u),
                dim=1,
                keepdim=True,
            )
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(
                kl_distance(torch.log(outputs_aux2_u + 1e-8), preds_u),
                dim=1,
                keepdim=True,
            )
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(
                kl_distance(torch.log(outputs_aux3_u + 1e-8), preds_u),
                dim=1,
                keepdim=True,
            )
            exp_variance_aux3 = torch.exp(-variance_aux3)

            consistency_dist_main = ((preds_u - outputs_u) ** 2) * conf_mask
            consistency_dist_aux1 = ((preds_u - outputs_aux1_u) ** 2) * conf_mask
            consistency_dist_aux2 = ((preds_u - outputs_aux2_u) ** 2) * conf_mask
            consistency_dist_aux3 = ((preds_u - outputs_aux3_u) ** 2) * conf_mask

            consistency_loss_main = (
                (torch.mean(consistency_dist_main * exp_variance_main) / norm_factor)
                / (torch.mean(exp_variance_main) + 1e-8)
                + args.uncertainty_penalty * torch.mean(variance_main)
            )

            consistency_loss_aux1 = (
                (torch.mean(consistency_dist_aux1 * exp_variance_aux1) / norm_factor)
                / (torch.mean(exp_variance_aux1) + 1e-8)
                + args.uncertainty_penalty * torch.mean(variance_aux1)
            )

            consistency_loss_aux2 = (
                (torch.mean(consistency_dist_aux2 * exp_variance_aux2) / norm_factor)
                / (torch.mean(exp_variance_aux2) + 1e-8)
                + args.uncertainty_penalty * torch.mean(variance_aux2)
            )

            consistency_loss_aux3 = (
                (torch.mean(consistency_dist_aux3 * exp_variance_aux3) / norm_factor)
                / (torch.mean(exp_variance_aux3) + 1e-8)
                + args.uncertainty_penalty * torch.mean(variance_aux3)
            )

            consistency_loss = (
                consistency_loss_main
                + consistency_loss_aux1
                + consistency_loss_aux2
                + consistency_loss_aux3
            ) / 4.0

            # ---------------- optional feature consistency ----------------
            feat_loss = compute_feature_loss(feats)
            if feat_loss is None:
                feat_loss = torch.tensor(0.0, device=volume_batch.device)

            # ---------------- dynamic weighting ----------------
            sup_weight = get_supervised_weight(epoch_num, max_epoch)
            out_scale = get_output_consistency_scale(epoch_num)
            feat_scale = get_feature_consistency_scale(epoch_num)

            consistency_weight = get_current_consistency_weight(epoch_num)
            feat_weight = args.feat_weight_max * feat_scale

            loss = (
                sup_weight * supervised_loss
                + out_scale * consistency_weight * consistency_loss
                + feat_weight * feat_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cosine lr
            lr_ = base_lr * (1.0 + math.cos(math.pi * iter_num / max_iterations)) / 2.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            # logs
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/supervised_loss", supervised_loss.item(), iter_num)
            writer.add_scalar("info/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("info/loss_dice", loss_dice.item(), iter_num)
            writer.add_scalar("info/consistency_loss", consistency_loss.item(), iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("info/sup_weight", sup_weight, iter_num)
            writer.add_scalar("info/out_scale", out_scale, iter_num)
            writer.add_scalar("info/feat_scale", feat_scale, iter_num)
            writer.add_scalar("info/feat_weight", feat_weight, iter_num)
            writer.add_scalar("info/feat_loss", feat_loss.item(), iter_num)
            writer.add_scalar("info/conf_thresh", conf_thresh, iter_num)
            writer.add_scalar("info/mask_ratio", mask_mean.item(), iter_num)
            writer.add_scalar("info/uncertainty_penalty", args.uncertainty_penalty, iter_num)

            logging.info(
                "iter %d | total %.4f | sup %.4f | cons %.4f | feat %.4f | ce %.4f | dice %.4f"
                % (
                    iter_num,
                    loss.item(),
                    supervised_loss.item(),
                    consistency_loss.item(),
                    feat_loss.item(),
                    loss_ce.item(),
                    loss_dice.item(),
                )
            )

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image("train/Image", image, iter_num)
                pred_show = torch.argmax(outputs_soft, dim=1, keepdim=True)
                writer.add_image("train/Prediction", pred_show[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image("train/GroundTruth", labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0

                for sampled_batch_val in valloader:
                    metric_i = test_single_volume_ds(
                        sampled_batch_val["image"],
                        sampled_batch_val["label"],
                        model,
                        classes=num_classes,
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        f"info/val_{class_i + 1}_dice",
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        f"info/val_{class_i + 1}_hd95",
                        metric_list[class_i, 1],
                        iter_num,
                    )

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar("info/val_mean_dice", performance, iter_num)
                writer.add_scalar("info/val_mean_hd95", mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(
                        snapshot_path,
                        f"iter_{iter_num}_dice_{round(best_performance, 4)}.pth",
                    )
                    save_best = os.path.join(snapshot_path, f"{args.model}_best_model.pth")
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    "iter %d | val_mean_dice %.4f | val_mean_hd95 %.4f"
                    % (iter_num, performance, mean_hd95)
                )
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, f"iter_{iter_num}.pth")
                torch.save(model.state_dict(), save_mode_path)
                logging.info(f"save model to {save_mode_path}")

            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")

    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
