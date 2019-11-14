from __future__ import print_function

import os
import argparse
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from keras.datasets import cifar100

from models.PreActResNet import *

from COT import *
from HCOT import *
from utils import *

# The number of coarse categories for cifar-100
coarse_categories = 20

# The number of fine categories for cifar-100
fine_categories = 100

(X, y_c), (x_test, y_c_test) = cifar100.load_data(label_mode='coarse')
(X, y), (x_test, y_test) = cifar100.load_data(label_mode='fine')

fine2coarse = np.zeros((fine_categories,coarse_categories))
for i in range(coarse_categories):
    index = np.where(y_c_test[:,0] == i)[0]
    fine_cat = np.unique([y_test[j,0] for j in index])
    for j in fine_cat:
        fine2coarse[j,i] = 1

fine2coarse = np.argmax(fine2coarse, axis=1)

# print(fine2coarse)

parser = argparse.ArgumentParser(
    description='PyTorch Hierarchical Complement Objective Training (HCOT)')


parser.add_argument('--HCOT', action='store_true',
                    help='Using Hierarchical Complement Objective Training (HCOT)')
parser.add_argument('--COT', action='store_true',
                    help='Using Complement Objective Training (COT)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--seed', default=11111, type=int, help='rng seed')
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size', '-b', default=128,
                    type=int, help='mini-batch size (default: 128)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')

args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
batch_size = args.batch_size
base_learning_rate = args.lr
complement_learning_rate = args.lr

if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu
    complement_learning_rate *= n_gpu

# Data (Default: CIFAR100)

print('==> Preparing CIFAR100 data.. (Default)')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' +
                            args.sess + '_' + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model.. (Default : PreActResNet-18)')
    start_epoch = 0
    net = PreActResNet18_CIFAR100()

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + \
    '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate,
                      momentum=0.9, weight_decay=args.decay)

# for COT
complement_criterion = ComplementEntropy()

# for HCOT
inner_complement_criterion = InnerComplementEntropy(fine2coarse)
outer_complement_criterion = OuterComplementEntropy(fine2coarse)

complement_optimizer = optim.SGD(net.parameters(
), lr=complement_learning_rate, momentum=0.9, weight_decay=args.decay)


# Training


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Baseline Implementation
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


        # COT Implementation
        if args.COT:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            loss = complement_criterion(outputs, targets)
            complement_optimizer.zero_grad()
            loss.backward()
            complement_optimizer.step()   

        # HCOT Implementation
        if args.HCOT:
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

            loss = inner_complement_criterion(outputs, targets) + outer_complement_criterion(outputs, targets)
            complement_optimizer.zero_grad()
            loss.backward()
            complement_optimizer.step()

    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        # checkpoint(acc, epoch)
    return (test_loss / batch_idx, 100. * correct / total, best_acc)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def complement_adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = complement_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (complement_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(
            ['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    complement_adjust_learning_rate(complement_optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc, best_acc = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

    if epoch == args.epochs - 1:
        print("best_acc : {}".format(best_acc))
