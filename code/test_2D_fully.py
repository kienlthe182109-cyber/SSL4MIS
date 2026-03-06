import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from medpy import metric
from networks.net_factory import net_factory
from dataloaders.dataset import BaseDataSets
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='dataset path')
parser.add_argument('--exp', type=str,
                    default='ACDC/Uncertainty_Rectified_Pyramid_Consistency')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7)
parser.add_argument('--batch_size', type=int, default=1)
FLAGS = parser.parse_args()


def calculate_metric_percase(pred, gt):
    """
    Safe metric calculation.
    Handles empty masks to avoid MedPy crash.
    """
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    # case 1: both empty
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0, 0.0

    # case 2: prediction empty but gt not
    if pred_sum == 0 and gt_sum > 0:
        return 0.0, 100.0, 100.0

    # case 3: gt empty but prediction not
    if pred_sum > 0 and gt_sum == 0:
        return 0.0, 100.0, 100.0

    # normal case
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, hd95, asd


def test_single_volume(image, label, net, classes):
    image = image.squeeze(0).cpu().numpy()
    label = label.squeeze(0).cpu().numpy()

    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
            out = net(input)

            # handle multi-output models
            if isinstance(out, (tuple, list)):
                out = out[0]

            out = torch.argmax(
                torch.softmax(out, dim=1),
                dim=1
            ).squeeze(0)

        prediction[ind] = out.cpu().numpy()

    first_metric = calculate_metric_percase(
        prediction == 1,
        label == 1
    )
    second_metric = calculate_metric_percase(
        prediction == 2,
        label == 2
    )
    third_metric = calculate_metric_percase(
        prediction == 3,
        label == 3
    )

    return first_metric, second_metric, third_metric


def Inference(FLAGS):

    snapshot_path = "../model/{}_{}_labeled/{}/".format(
        FLAGS.exp,
        FLAGS.labeled_num,
        FLAGS.model
    )

    num_classes = FLAGS.num_classes

    test_save_path = "../model/predictions/"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

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

    db_test = BaseDataSets(
        base_dir=FLAGS.root_path,
        split="test"
    )

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    metric_list = 0.0

    for sampled_batch in tqdm(testloader):
        image = sampled_batch["image"]
        label = sampled_batch["label"]

        image, label = image.cuda(), label.cuda()

        first_metric, second_metric, third_metric = test_single_volume(
            image,
            label,
            net,
            classes=num_classes
        )

        metric_list += np.array(
            (first_metric, second_metric, third_metric)
        )

    metric_list = metric_list / len(db_test)

    print("Mean metrics:")
    print(metric_list)

    return metric_list


if __name__ == '__main__':
    metric = Inference(FLAGS)
