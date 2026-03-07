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
    default="ACDC/URPC_ClassAware_Entropy",
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

# dynamic weighting
parser.add_argument("--sup_decay", type=float, default=0.3, help="how much supervised weight decays")

# uncertainty penalty
parser.add_argument("--uncertainty_penalty", type=float, default=0.1, help="weight for variance penalty")

# dual-view consistency
parser.add_argument("--dual_consistency_weight", type=float, default=0.05, help="weight for noisy-view consistency")
parser.add_argument("--noise_std", type=float, default=0.1, help="std of gaussian noise for second view")
parser.add_argument("--noise_clip", type=float, default=0.2, help="clip range for gaussian noise")

# entropy minimization
parser.add_argument("--entropy_weight", type=float, default=0.005, help="weight for entropy minimization on unlabeled data")

# class-aware thresholding
parser.add_argument("--bg_thresh", type=float, default=0.8, help="threshold for background")
parser.add_argument("--rv_thresh", type=float, default=0.7, help="threshold for RV")
parser.add_argument("--myo_thresh", type=float, default=0.6, help="threshold for MYO")
parser.add_argument("--lv_thresh", type=float, default=0.7, help="threshold for LV")
parser.add_argument("--class_thresh_rampup", type=float, default=50.0, help="rampup for class thresholds from relaxed to target")

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
        raise ValueError("Unknown dataset")
    return ref_dict[str(patients_num)]


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_supervised_weight(epoch, max_epoch):
    progress = min(epoch / max_epoch, 1.0)
    return 1.0 - args.sup_decay * progress


def get_output_consistency_scale(epoch):
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def get_current_class_thresholds(epoch):
    """
    Ramp thresholds from a relaxed value (0.5) to target thresholds.
    """
    start = 0.5
    ramp = ramps.sigmoid_rampup(epoch, args.class_thresh_rampup)

    target = torch.tensor(
        [args.bg_thresh, args.rv_thresh, args.myo_thresh, args.lv_thresh],
        dtype=torch.float32,
        device="cuda"
    )
    current = start + (target - start) * ramp
    return current


def unpack_outputs(model_out):
    if not isinstance(model_out, (tuple, list)) or len(model_out) != 4:
        raise ValueError("This train-file-only version expects model to return exactly 4 outputs.")
    return model_out[0], model_out[1], model_out[2], model_out[3]


def entropy_minimization_loss(prob):
    """
    prob: [B, C, H, W]
    """
    ent = -torch.sum(prob * torch.log(prob + 1e-8), dim=1)
    return torch.mean(ent)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)

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
    print("Total slices: {}, labeled slices: {}".format(total_slices, labeled_slice))

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

    model.train()
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
    )

    class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3]).cuda()
    ce_loss = CrossEntropyLoss(weight=class_weights)
    dice_loss = losses.DiceLoss(num_classes)
    kl_distance = nn.KLDivLoss(reduction="none")

    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=80)

    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            lb = args.labeled_bs

            # ---------------- main forward ----------------
            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = unpack_outputs(model(volume_batch))

            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

            # ---------------- second noisy-view forward ----------------
            noise = torch.clamp(
                torch.randn_like(volume_batch) * args.noise_std,
                -args.noise_clip,
                args.noise_clip,
            )
            volume_batch_noise = volume_batch + noise

            with torch.no_grad():
                outputs_t, outputs_aux1_t, outputs_aux2_t, outputs_aux3_t = unpack_outputs(model(volume_batch_noise))
                outputs_t_soft = torch.softmax(outputs_t, dim=1)
                outputs_aux1_t_soft = torch.softmax(outputs_aux1_t, dim=1)
                outputs_aux2_t_soft = torch.softmax(outputs_aux2_t, dim=1)
                outputs_aux3_t_soft = torch.softmax(outputs_aux3_t, dim=1)

            # ---------------- supervised loss ----------------
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

            # ---------------- ensemble pseudo target ----------------
            preds = (
                outputs_soft + outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft
            ) / 4.0
            preds = preds.detach()

            preds_u = preds[lb:]
            outputs_u = outputs_soft[lb:]
            outputs_aux1_u = outputs_aux1_soft[lb:]
            outputs_aux2_u = outputs_aux2_soft[lb:]
            outputs_aux3_u = outputs_aux3_soft[lb:]

            # ---------------- class-aware threshold mask ----------------
            with torch.no_grad():
                class_thresholds = get_current_class_thresholds(epoch_num)  # [4]
                pseudo_prob, pseudo_label = torch.max(preds_u, dim=1, keepdim=True)  # [B,1,H,W]
                threshold_map = class_thresholds[pseudo_label.squeeze(1)].unsqueeze(1)  # [B,1,H,W]
                conf_mask = (pseudo_prob > threshold_map).float()
                mask_mean = conf_mask.mean()
                norm_factor = mask_mean + 1e-6

            # ---------------- variance-aware output consistency ----------------
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

            # ---------------- dual-view consistency ----------------
            outputs_t_u = outputs_t_soft[lb:]
            outputs_aux1_t_u = outputs_aux1_t_soft[lb:]
            outputs_aux2_t_u = outputs_aux2_t_soft[lb:]
            outputs_aux3_t_u = outputs_aux3_t_soft[lb:]

            dual_consistency_main = torch.mean(((outputs_u - outputs_t_u) ** 2) * conf_mask) / norm_factor
            dual_consistency_aux1 = torch.mean(((outputs_aux1_u - outputs_aux1_t_u) ** 2) * conf_mask) / norm_factor
            dual_consistency_aux2 = torch.mean(((outputs_aux2_u - outputs_aux2_t_u) ** 2) * conf_mask) / norm_factor
            dual_consistency_aux3 = torch.mean(((outputs_aux3_u - outputs_aux3_t_u) ** 2) * conf_mask) / norm_factor

            dual_view_loss = (
                dual_consistency_main
                + dual_consistency_aux1
                + dual_consistency_aux2
                + dual_consistency_aux3
            ) / 4.0

            # ---------------- entropy minimization on unlabeled ----------------
            entropy_loss = entropy_minimization_loss(preds_u)

            # ---------------- dynamic weighting ----------------
            sup_weight = get_supervised_weight(epoch_num, max_epoch)
            out_scale = get_output_consistency_scale(epoch_num)
            consistency_weight = get_current_consistency_weight(epoch_num)

            loss = (
                sup_weight * supervised_loss
                + out_scale * consistency_weight * consistency_loss
                + args.dual_consistency_weight * out_scale * dual_view_loss
                + args.entropy_weight * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---------------- cosine lr ----------------
            lr_ = base_lr * (1.0 + math.cos(math.pi * iter_num / max_iterations)) / 2.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            iter_num += 1

            # ---------------- logs ----------------
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            writer.add_scalar("info/supervised_loss", supervised_loss.item(), iter_num)
            writer.add_scalar("info/consistency_loss", consistency_loss.item(), iter_num)
            writer.add_scalar("info/dual_view_loss", dual_view_loss.item(), iter_num)
            writer.add_scalar("info/entropy_loss", entropy_loss.item(), iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            writer.add_scalar("info/sup_weight", sup_weight, iter_num)
            writer.add_scalar("info/out_scale", out_scale, iter_num)
            writer.add_scalar("info/mask_ratio", mask_mean.item(), iter_num)
            writer.add_scalar("info/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("info/loss_dice", loss_dice.item(), iter_num)
            writer.add_scalar("info/bg_thresh", class_thresholds[0].item(), iter_num)
            writer.add_scalar("info/rv_thresh", class_thresholds[1].item(), iter_num)
            writer.add_scalar("info/myo_thresh", class_thresholds[2].item(), iter_num)
            writer.add_scalar("info/lv_thresh", class_thresholds[3].item(), iter_num)

            logging.info(
                "iter %d | total %.4f | sup %.4f | cons %.4f | dual %.4f | ent %.4f | ce %.4f | dice %.4f"
                % (
                    iter_num,
                    loss.item(),
                    supervised_loss.item(),
                    consistency_loss.item(),
                    dual_view_loss.item(),
                    entropy_loss.item(),
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
                        "info/val_{}_dice".format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num,
                    )
                    writer.add_scalar(
                        "info/val_{}_hd95".format(class_i + 1),
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
                        "iter_{}_dice_{}.pth".format(iter_num, round(best_performance, 4)),
                    )
                    save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    "iter %d | val_mean_dice %.4f | val_mean_hd95 %.4f"
                    % (iter_num, performance, mean_hd95)
                )
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path, "iter_{}.pth".format(iter_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

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
