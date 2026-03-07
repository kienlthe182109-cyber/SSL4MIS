import argparse
import os
import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory


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
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)

    pred[pred > 0] = 1
    gt[gt > 0] = 1

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    # both empty
    if pred_sum == 0 and gt_sum == 0:
        return 1.0, 0.0, 0.0

    # one empty, one non-empty
    if pred_sum == 0 or gt_sum == 0:
        return 0.0, 100.0, 100.0

    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd95, asd


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    """
    image: numpy array [D, H, W]
    label: numpy array [D, H, W]
    """
    prediction = np.zeros_like(label)

    for ind in range(image.shape[0]):
        slice_img = image[ind, :, :]
        x, y = slice_img.shape[0], slice_img.shape[1]

        # resize slice to patch size used in training
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

        # resize prediction back to original size
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

    net = net_factory(
        net_type=FLAGS.model,
        in_chns=1,
        class_num=FLAGS.num_classes
    )

    save_mode_path = os.path.join(
        snapshot_path,
        '{}_best_model.pth'.format(FLAGS.model)
    )

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    # ---------------- full dataset from test.list ----------------
    test_list_path = os.path.join(FLAGS.root_path, "test.list")
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(
            "Không tìm thấy test.list tại: {}. "
            "Hãy kiểm tra lại thư mục dataset ACDC.".format(test_list_path)
        )

    with open(test_list_path, "r") as f:
        image_list = [item.strip() for item in f.readlines() if item.strip()]

    print("total {} samples".format(len(image_list)))

    if len(image_list) == 0:
        raise ValueError("test.list đang rỗng, không có case nào để test.")

    metric_list = 0.0

    for case in tqdm(image_list):
        h5_path = os.path.join(FLAGS.root_path, "data", "{}.h5".format(case))
        if not os.path.exists(h5_path):
            raise FileNotFoundError("Không tìm thấy file: {}".format(h5_path))

        h5f = h5py.File(h5_path, "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        h5f.close()

        first_metric, second_metric, third_metric = test_single_volume(
            image,
            label,
            net,
            classes=FLAGS.num_classes,
            patch_size=FLAGS.patch_size
        )

        metric_list += np.array((first_metric, second_metric, third_metric))

    metric_list = metric_list / len(image_list)

    print("Mean metric per class [Dice, HD95, ASD]:")
    print(metric_list)

    print("Mean Dice: {:.4f}".format(np.mean(metric_list[:, 0])))
    print("Mean HD95: {:.4f}".format(np.mean(metric_list[:, 1])))
    print("Mean ASD : {:.4f}".format(np.mean(metric_list[:, 2])))

    return metric_list


if __name__ == '__main__':
    metric = Inference(FLAGS)
