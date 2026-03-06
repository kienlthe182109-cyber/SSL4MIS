import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from medpy import metric
from scipy.ndimage import zoom
from torch.utils.data import DataLoader

from networks.net_factory import net_factory
from dataloaders.dataset import BaseDataSets


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='dataset path')
parser.add_argument('--exp', type=str,
                    default='ACDC/Uncertainty_Rectified_Pyramid_Consistency')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='number of labeled patients')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size used during training')
FLAGS = parser.parse_args()


def calculate_metric_percase(pred, gt):
    """
    Safe metric calculation for binary masks.
    Handles empty-mask cases to avoid MedPy crashes.
    """
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    # both empty -> perfect match for this class
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0, 0.0

    # one empty, one non-empty -> worst case
    if pred_sum == 0 and gt_sum > 0:
        return 0.0, 100.0, 100.0

    if pred_sum > 0 and gt_sum == 0:
        return 0.0, 100.0, 100.0

    # normal case
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, hd95, asd


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    """
    image: torch tensor [1, D, H, W]
    label: torch tensor [1, D, H, W]
    """
    image = image.squeeze(0).cpu().numpy()
    label = label.squeeze(0).cpu().numpy()

    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]

        # Resize to training patch size to avoid U-Net shape mismatch
        if x != patch_size[0] or y != patch_size[1]:
            slice_resized = zoom(
                slice_img,
                (patch_size[0] / x, patch_size[1] / y),
                order=0
            )
        else:
            slice_resized = slice_img

        input_tensor = torch.from_numpy(slice_resized).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
            out = net(input_tensor)

            # handle multi-output models like URPC
            if isinstance(out, (tuple, list)):
                out = out[0]

            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            pred = out.cpu().numpy()

        # Resize prediction back to original slice size
        if x != patch_size[0] or y != patch_size[1]:
            pred = zoom(
                pred,
                (x / patch_size[0], y / patch_size[1]),
                order=0
            )

        prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    snapshot_path = "../model/{}_{}_labeled/{}/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model
    )

    num_classes = FLAGS.num_classes

    net = net_factory(
        net_type=FLAGS.model,
        in_chns=1,
        class_num=num_classes
    )

    save_mode_path = os.path.join(
        snapshot_path,
        '{}_best_model.pth'.format(FLAGS.model)
    )

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # SSL4MIS ACDC commonly uses "val" split for evaluation
    db_test = BaseDataSets(
        base_dir=FLAGS.root_path,
        split="val"
    )

    print("total {} samples".format(len(db_test)))

    if len(db_test) == 0:
        raise ValueError(
            "No samples found in dataset. Please check dataset path and split name."
        )

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    metric_list = 0.0

    for sampled_batch in tqdm(testloader):
        image = sampled_batch["image"].cuda()
        label = sampled_batch["label"].cuda()

        first_metric, second_metric, third_metric = test_single_volume(
            image,
            label,
            net,
            classes=num_classes,
            patch_size=FLAGS.patch_size
        )

        metric_list += np.array(
            (first_metric, second_metric, third_metric)
        )

    metric_list = metric_list / len(db_test)

    print("Mean metrics per class [Dice, HD95, ASD]:")
    print(metric_list)

    mean_dice = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_asd = np.mean(metric_list, axis=0)[2]

    print("Mean Dice: {:.4f}".format(mean_dice))
    print("Mean HD95: {:.4f}".format(mean_hd95))
    print("Mean ASD : {:.4f}".format(mean_asd))

    return metric_list


if __name__ == '__main__':
    metric = Inference(FLAGS)
