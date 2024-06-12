'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import timm

import torchvision
import torchvision.transforms as transforms
import random
import os
import argparse

from models import *
from utils import progress_bar
import ttach as tta


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # gpu 1개 이상일 때

    cudnn.benchmark = False
    cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy

#seed_all(42)

transforms_TTA = tta.Compose(
    [
        tta.HorizontalFlip(),
        #tta.Rotate90(angles=[0, 180]),
        #tta.Scale(scales=[1, 2, 4]),
        #tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)

print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.Resize([128, 128]), # 192 Resize 추가
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#criterion = nn.CrossEntropyLoss()

net_names = ["resnet18","vgg19","resnet50","resnet101","mobilenetv2_050","resnext26ts","dla34","densenet121","dpn92","efficientnet_b0","dla169"]
optimizer_list = ["SGD", "AdamW"]
scheduler_list = ["CosineAnnealingLR", "CosineAnnealingWarmRestarts"]

### mission1. 필요한 모델 불러오기. 불러와서 리스트에 따로 담아두기.
net_list = []

# 성능이 90 이상인 모델 weight만 불러오기
for model_name in os.listdir('./checkpoint_net'):   # model_name는 'str'
    acc = int(model_name.split('_')[-1].replace('.pth', ''))
    model = model_name.split('_')[0]
    seed_all(42)
    if acc > 9120:
        net = timm.create_model(model, pretrained=True, num_classes=10)
        net_state_dict = torch.load(f'./checkpoint_net/{model_name}', map_location=device)
        net.load_state_dict(net_state_dict, strict=False)
        net.eval()
        net = net.to(device)
        net_list.append(net)
print(len(net_list))
### mission2. 필요한 모델을 모아둔 리스트에서 (원하시는대로/원하시는만큼) 모델 불러와서 앙상블 + TTA 적용시키기
""" transforms_TTA = tta.Compose(
    [
        tta.HorizontalFlip(),
        #tta.Rotate90(angles=[0, 180]),
        #tta.Scale(scales=[1, 2, 4]),
        #tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
) """

test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # mission2-1. 필요한 모델을 불러오고 이에 tta를 적용한다.
        outputs = torch.zeros(100, 10).to(device) 
        for net_name in net_list:
            tta_model = tta.ClassificationTTAWrapper(net_name, transforms_TTA)
            tta_outputs = tta_model(inputs)
            outputs += tta_outputs

        #loss = criterion(outputs, targets)

        #test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

# 현재 정확도 계산
acc = 100. * correct / total
print(acc)
