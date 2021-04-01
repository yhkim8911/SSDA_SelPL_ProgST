from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34, resnet101
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset_soft_pl
from utils.loss import entropy, adentropy
import torch.nn.functional as F
import torchvision

# setups
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME',
                    choices=['S+T', 'ENT', 'MME'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath-load', type=str, default='./save_model_ssda/baseline',
                    help='dir to save checkpoint')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda/selPL_progST',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--val_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before validation')
parser.add_argument('--save_interval', type=int, default=2000, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='clipart',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=10, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 10 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--device', type=str, default='0',
                    help='GPU ID')
parser.add_argument('--pl_ratio', type=int, default=20,
                    help='ratio of pseudo labels (percentage)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum of updating pseudo labels')

args = parser.parse_args()
print('Dataset %s, Source %s, Target %s, Labeled num perclass %s, Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, target_loader_val, \
    target_loader_test, target_loader_pl, target_loader_pl_update, class_list = return_dataset_soft_pl(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s/%s' % (args.dataset, args.method, 'selPL_progST')
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s_ratio_%s' %
                           (args.method, args.net, args.source, args.target, args.num, args.pl_ratio))

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
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

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net or "densenet" in args.net or "mobilenet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc, temp=args.T)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
weights_init(F1)
lr = args.lr
device = torch.device("cuda:" + args.device)

G = G.to(device)
F1 = F1.to(device)

G.load_state_dict(torch.load(os.path.join(args.checkpath_load,
                                          "G_iter_model_{}_{}_{}_"
                                          "to_{}_num_{}.pth.tar".
                                          format(args.method, args.net, args.source, args.target, args.num))))
F1.load_state_dict(torch.load(os.path.join(args.checkpath_load,
                                           "F1_iter_model_{}_{}_{}_"
                                           "to_{}_num_{}.pth.tar".
                                           format(args.method, args.net, args.source, args.target, args.num))))

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_t_pl = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
p_labels_t = torch.FloatTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)


im_data_s = im_data_s.to(device)
im_data_t = im_data_t.to(device)
im_data_tu = im_data_tu.to(device)
im_data_t_pl = im_data_t_pl.to(device)
gt_labels_s = gt_labels_s.to(device)
gt_labels_t = gt_labels_t.to(device)
p_labels_t = p_labels_t.to(device)
sample_labels_t = sample_labels_t.to(device)
sample_labels_s = sample_labels_s.to(device)


im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_t_pl = Variable(im_data_t_pl)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
p_labels_t = Variable(p_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)


if not os.path.exists(args.checkpath):
    os.mkdir(args.checkpath)


def train():
    G.train()
    F1.train()
    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_sl = criterion_soft
    criterion_sl_rev = criterion_soft_rev

    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    data_iter_t_pl = iter(target_loader_pl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    len_train_target_pl = len(target_loader_pl)
    best_acc = 0.
    best_acc_test = 0.
    counter = 0

    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        if step % len_train_target_pl == 0:
            """
            # update pseudo labels
            data_iter_t_pl_update = iter(target_loader_pl_update)
            path_with_soft_pl = []
            filepath_pl = os.path.join("./data/txt", args.dataset, "pseudo_label", args.net)
            filepath_pl = os.path.join(filepath_pl, "pseudo_label_"
                                       + args.source + "_" + args.target + "_"
                                       + str(args.num) + "_ratio_" + str(args.pl_ratio) + ".pt")
            G.eval()
            F1.eval()
            with torch.no_grad():
                for data_t_pl in data_iter_t_pl_update:
                    # data_t_pl = next(data_iter_t_pl_update)
                    im_data_t_pl.resize_(data_t_pl[0].size()).copy_(data_t_pl[0])
                    p_labels_t.resize_(data_t_pl[1].size()).copy_(data_t_pl[1])
                    output_sl_update = G(im_data_t_pl)
                    output_sl_update = F1(output_sl_update)

                    data_t_pl[1] = args.momentum * data_t_pl[1] + (1.0 - args.momentum) * output_sl_update.data.cpu()
                    data_t_pl[0] = data_t_pl[0].data.cpu()
                    data_t_pl[1] = data_t_pl[1].data.cpu()
                    path_with_soft_pl.append([data_t_pl[2][0], data_t_pl[1][0]])
                    """

            data_iter_t_pl = iter(target_loader_pl)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        data_t_pl = next(data_iter_t_pl)
        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_t_pl.resize_(data_t_pl[0].size()).copy_(data_t_pl[0])
        p_labels_t.resize_(data_t_pl[1].size()).copy_(data_t_pl[1])
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        zero_grad_all()
        # data = torch.cat((im_data_s, im_data_t), 0)
        # target = torch.cat((gt_labels_s, gt_labels_t), 0)
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)
        out1 = F1(output)

        # Soft label
        data_sl = im_data_t_pl
        target_sl = p_labels_t
        output_sl = G(data_sl)
        out1_sl = F1(output_sl)
        loss = criterion(out1, target) + criterion_sl(out1_sl, target_sl)
        # loss = criterion_sl(out1_sl, target_sl)
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        if not args.method == 'S+T':
            output = G(im_data_tu)
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S: {}, T: {}, Train Ep: {} lr: {} \t ' \
                        'Loss Classification: {:.6f} Loss T: {:.6f} ' \
                        'Method: {}\n'.format(args.source, args.target,
                                              step, lr, loss.data,
                                              -loss_t.data, args.method)
        else:
            log_train = 'S: {}, T: {}, Train Ep: {} lr: {} \t ' \
                        'Loss Classification: {:.6f} Method: {}\n'. \
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.val_interval == 0 and step > 0:
            # -----------updating pseudo labels----------- #
            if step >= 5000:
                data_iter_t_pl_update = iter(target_loader_pl_update)
                path_with_soft_pl = []
                filepath_pl = os.path.join("./data/txt", args.dataset, "pseudo_label", args.net)
                filepath_pl = os.path.join(filepath_pl, "pseudo_label_"
                                           + args.source + "_" + args.target + "_"
                                           + str(args.num) + "_ratio_" + str(args.pl_ratio) + ".pt")
                G.eval()
                F1.eval()
                with torch.no_grad():
                    for data_t_pl in data_iter_t_pl_update:
                        # data_t_pl = next(data_iter_t_pl_update)
                        im_data_t_pl.resize_(data_t_pl[0].size()).copy_(data_t_pl[0])
                        p_labels_t.resize_(data_t_pl[1].size()).copy_(data_t_pl[1])
                        output_sl_update = G(im_data_t_pl)
                        output_sl_update = F1(output_sl_update)
                        output_sl_update = F.softmax(output_sl_update, dim=1)

                        data_t_pl[1] = args.momentum * data_t_pl[1] + (
                                    1.0 - args.momentum) * output_sl_update.data.cpu()
                        data_t_pl[0] = data_t_pl[0].data.cpu()
                        data_t_pl[1] = data_t_pl[1].data.cpu()
                        path_with_soft_pl.append([data_t_pl[2][0], data_t_pl[1][0]])
                # -----------end of updating pseudo labels----------- #
                torch.save(path_with_soft_pl, filepath_pl)

            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            G.train()
            F1.train()
            if acc_val > best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_{}_"
                                        "to_{}_best.pth.tar".
                                        format(args.method, args.net, args.source, args.target)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_{}_"
                                        "to_{}_best.pth.tar".
                                        format(args.method, args.net, args.source, args.target)))
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    torch.save(G.state_dict(), os.path.join(args.checkpath, "G_iter_model_{}_{}_{}_to_{}_last.pth.tar".format(args.method, args.net, args.source, args.target)))
                    torch.save(F1.state_dict(), os.path.join(args.checkpath, "F1_iter_model_{}_{}_{}_to_{}_last.pth.tar".format(args.method, args.net, args.source, args.target)))
                    break
            print('best acc test: %f, acc test: %f' % (best_acc_test, acc_test))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step: %d, best: %f, current: %f \n' % (step, best_acc_test, acc_test))
            G.train()
            F1.train()
            if args.save_check and step % args.save_interval == 0:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().to(device)
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.2f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


def criterion_soft(outputs, soft_targets):
    # We introduce a prior probability distribution p, which is a distribution of classes among all training data.
    # p = torch.ones(126).to(device) / 126

    # probs = F.softmax(outputs, dim=1)
    # avg_probs = torch.mean(probs, dim=0)

    L_c = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * soft_targets, dim=1))
    # L_p = -torch.sum(torch.log(avg_probs) * p)
    # L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

    # loss = L_c + 1.0 * L_p + 0.5 * L_e
    loss = L_c
    return loss


def criterion_soft_rev(outputs, soft_targets):
    # We introduce a prior probability distribution p, which is a distribution of classes among all training data.
    # p = torch.ones(126).to(device) / 126

    # probs = F.softmax(outputs, dim=1)
    # avg_probs = torch.mean(probs, dim=0)

    L_c = -torch.mean(torch.sum(torch.log(soft_targets) * F.softmax(outputs, dim=1), dim=1))
    # L_p = -torch.sum(torch.log(avg_probs) * p)
    # L_e = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * probs, dim=1))

    # loss = L_c + 1.0 * L_p + 0.5 * L_e
    loss = L_c
    return loss


if __name__ == "__main__":
    train()
