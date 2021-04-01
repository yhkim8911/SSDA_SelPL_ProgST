from __future__ import print_function

import argparse
import os
import torch
from model.resnet import resnet34, resnet50, resnet101
from torch.autograd import Variable
from tqdm import tqdm
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.return_dataset import return_dataset_feature
import torch.nn.functional as F
import torchvision
import cv2
from skimage import img_as_float
import numpy
import math

# setups
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda/baseline',
                    help='dir to save checkpoint')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         'S+T is training only on labeled examples')
parser.add_argument('--output', type=str, default='./output.txt',
                    help='path to store result file')

parser.add_argument('--net', type=str, default='resnet34', metavar='B',
                    help='which network ')
parser.add_argument('--source', type=str, default='real', metavar='B',
                    help='board dir')
parser.add_argument('--target', type=str, default='clipart', metavar='B',
                    help='board dir')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi'],
                    help='the name of dataset, multi is large scale dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--pl_ratio', type=int, default=20,
                    help='ratio of pseudo labels (in percentage)')
parser.add_argument('--device', type=str, default='0',
                    help='GPU ID')

args = parser.parse_args()
print('dataset %s source %s target %s network %s' %
      (args.dataset, args.source, args.target, args.net))
# target_loader_unl, class_list = return_dataset_test(args)
target_loader_labeled, target_loader_unl, class_list = return_dataset_feature(args)
use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet50':
    G = resnet50()
    inc = 2048
elif args.net == 'resnet101':
    G = resnet101()
    inc = 2048
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
elif args.net == "densenet":
    G = torchvision.models.densenet121(pretrained=True)
    G.classifier = torch.nn.Identity()
    inc = 1024
elif args.net == "mobilenet":
    G = torchvision.models.mobilenet_v2(pretrained=True)
    inc = 1280
    G.classifier = torch.nn.Identity()
else:
    raise ValueError('Model cannot be recognized.')

if "resnet" in args.net or "densenet" in args.net or "mobilenet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)

device = torch.device("cuda:" + args.device)

G = G.to(device)
F1 = F1.to(device)

G.load_state_dict(torch.load(os.path.join(args.checkpath,
                                          "G_iter_model_{}_{}_{}_to_{}_num_{}.pth.tar".
                                          format(args.method, args.net, args.source, args.target, args.num))))
F1.load_state_dict(torch.load(os.path.join(args.checkpath,
                                           "F1_iter_model_{}_{}_{}_to_{}_num_{}.pth.tar".
                                           format(args.method, args.net, args.source, args.target, args.num))))


im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.to(device)
gt_labels_t = gt_labels_t.to(device)

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)
if not os.path.exists(args.checkpath):
    os.mkdir(args.checkpath)


def eval_feature_dist_and_conduct_soft_pl(args):
    G.eval()
    F1.eval()
    size = 0
    feature_dir = os.path.join("features", args.net, args.source + "_to_" + args.target)
    filepath_labeled = os.path.join(feature_dir,
                                    "labeled_" + args.source + "_to_" + args.target + "_" + str(args.num) + ".pt")
    filepath_unl = os.path.join(feature_dir, "unl_" + args.source + "_to_" + args.target + "_" + str(args.num) + ".pt")
    feat_list_labeled = torch.load(filepath_labeled)  # [feat, pred, gt]
    feat_list_unl = torch.load(filepath_unl)

    filepath_pl = os.path.join("./data/txt", args.dataset, "pseudo_label", args.net)
    if not os.path.exists(filepath_pl):
        os.makedirs(filepath_pl)
    filepath_pl = os.path.join(filepath_pl, "pseudo_label_"
                               + args.source + "_" + args.target + "_"
                               + str(args.num) + "_ratio_" + str(args.pl_ratio) + ".pt")
    num_tot_sample = 0.
    num_tot_correct = 0.
    num_p_sample = 0.
    num_p_correct = 0.
    path_with_soft_pl = []

    for label_idx in range(len(class_list)):
        feat_centroid = []
        for shot_idx in range(args.num):
            feat_centroid_idx = label_idx * args.num + shot_idx
            feat_centroid.append(feat_list_labeled[feat_centroid_idx][0])

        dist_list = []
        num_sample = 0
        for feat_unl in feat_list_unl:
            if feat_unl[1] == label_idx:
                diff = 0.
                for shot_idx in range(args.num):
                    diff += torch.norm(feat_unl[0] - feat_centroid[shot_idx], 1).numpy()
                is_correct = feat_unl[1] == feat_unl[2]
                dist_list.append([diff, feat_unl[1], feat_unl[2], feat_unl[3],
                                  feat_unl[4], feat_unl[5]])  # diff, pred, gt, conf, path, similarities
                num_sample += 1
                num_tot_sample += 1
                if is_correct:
                    num_tot_correct += 1
        dist_list_sorted = sorted(dist_list, key=lambda x: x[0])
        num_pl_per_class = round(len(target_loader_unl.dataset) / len(class_list) * args.pl_ratio / 100.0)
        pl_idx = 0
        while pl_idx < min(num_pl_per_class, num_sample):
            _, p_label, gt_label, conf, path, sim = dist_list_sorted[pl_idx]
            path_with_soft_pl.append([path, sim])
            if p_label == gt_label:
                num_p_correct += 1
            num_p_sample += 1
            pl_idx += 1

    torch.save(path_with_soft_pl, filepath_pl)
    print("Total corrects: {}, Total samples: {}".format(num_tot_correct, num_tot_sample))
    print("Total accuracy of pseudo labels: {}".format(num_tot_correct / num_tot_sample))
    print()
    print("Total pseudo corrects: {}, Total pseudo samples: {}".format(num_p_correct, num_p_sample))
    print("Total accuracy of filtered pseudo labels: {}".format(num_p_correct / num_p_sample))
    print("\n")


if __name__ == "__main__":
    eval_feature_dist_and_conduct_soft_pl(args)
