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
parser.add_argument('--device', type=str, default='0',
                    help='GPU ID')
parser.add_argument('--T-sl', type=float, default=1.0, metavar='T',
                    help='temperature for soft label (default: 1.0)')

args = parser.parse_args()
print('dataset %s source %s target %s network %s' %
      (args.dataset, args.source, args.target, args.net))
# target_loader_unl, class_list = return_dataset_test(args)
target_loader_labeled, target_loader_unl, class_list = return_dataset_feature(args)
use_gpu = torch.cuda.is_available()

if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == 'resnet101':
    G = resnet101()
    inc = 2048
elif args.net == 'resnet50':
    G = resnet50()
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
                                          "G_iter_model_{}_{}_{}_"
                                          "to_{}_num_{}.pth.tar".
                                          format(args.method, args.net, args.source, args.target, args.num))))
F1.load_state_dict(torch.load(os.path.join(args.checkpath,
                                           "F1_iter_model_{}_{}_{}_"
                                           "to_{}_num_{}.pth.tar".
                                           format(args.method, args.net, args.source, args.target, args.num))))

im_data_t = torch.FloatTensor(1)
gt_labels_t = torch.LongTensor(1)

im_data_t = im_data_t.to(device)
gt_labels_t = gt_labels_t.to(device)

im_data_t = Variable(im_data_t)
gt_labels_t = Variable(gt_labels_t)
if not os.path.exists(args.checkpath):
    os.mkdir(args.checkpath)


def eval_labeled(args, loader, output_file="output.txt"):
    G.eval()
    F1.eval()
    size = 0
    feat_list = []
    feature_dir = os.path.join("features", args.net, args.source + "_to_" + args.target)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    filepath = os.path.join(feature_dir, "labeled_" + args.source + "_to_" + args.target + "_" + str(args.num) + ".pt")
    if not os.path.exists("record_elements"):
        os.makedirs("record_elements")
    output_file = os.path.join("record_elements", output_file)
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(loader)):
                im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
                paths = data_t[2]
                feat = G(im_data_t)
                output1 = F1(feat)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for i, path in enumerate(paths):
                    f.write("%s %d\n" % (path, pred1[i]))
                    feat_list.append([feat[i].cpu(), pred1[i].cpu(), gt_labels_t[i].cpu()])
        torch.save(feat_list, filepath)


def eval_unl(args, loader, output_file="output.txt"):
    G.eval()
    F1.eval()
    size = 0
    feat_list = []
    feature_dir = os.path.join("features", args.net, args.source + "_to_" + args.target)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    filepath = os.path.join(feature_dir, "unl_" + args.source + "_to_" + args.target + "_" + str(args.num) + ".pt")
    output_file = os.path.join("record_elements", output_file)
    with open(output_file, "w") as f:
        with torch.no_grad():
            for batch_idx, data_t in tqdm(enumerate(loader)):
                im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
                paths = data_t[2]
                feat = G(im_data_t)
                output1 = F1(feat)
                # sim = F.softmax(output1 / args.T_sl, dim=1)
                sim = F.softmax(output1, dim=1)
                size += im_data_t.size(0)
                pred1 = output1.data.max(1)[1]
                for i, path in enumerate(paths):
                    f.write("%s %d\n" % (path, pred1[i]))
                    feat_list.append([feat[i].cpu(), pred1[i].cpu(), gt_labels_t[i].cpu(),
                                      sim[i][pred1[i]].cpu(), path, sim[i].cpu()])
    torch.save(feat_list, filepath)


eval_labeled(args, target_loader_labeled, output_file="%s_%s_%s_to_%s.txt" % (args.method, args.net,
                                                                           args.source, args.target))
eval_unl(args, target_loader_unl, output_file="%s_%s_%s_to_%s.txt" % (args.method, args.net,
                                                                   args.source, args.target))

