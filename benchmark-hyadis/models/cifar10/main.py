'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

from models import *

from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter

s = time.time()

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.08, type=float, help='learning rate')
parser.add_argument('--epochs', default=90, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="/workspace/data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024),
                                 gradient_accumulation=True)

validset = torchvision.datasets.CIFAR10(root="/workspace/data", train=False, download=False, transform=transform_test)
validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = eval(args.model)()
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ExponentialLR(optimizer, 0.0133 ** (1.0 / args.epochs))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    begin_train_time = time.time()
    net.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data")
        net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model")

    use_time = time.time() - begin_train_time

def valid(epoch):
    begin_valid_time = time.time()
    net.eval()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
    
    use_time = time.time() - begin_valid_time
    


with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in range(args.epochs):
        train(epoch)
        valid(epoch)
        lr_scheduler.step()
    writer.export_scalars_to_json(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp") + 'tensorboard.json')
