from torchvision import transforms
import torch
from utilities import Sobel, TwoViews
from SimCLR_model import SimCLR
device = ("cuda:0" if torch.cuda.is_available else "cpu")

transform_train = transforms.Compose([
    transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, hue=0)], p=0.5), # 以50%的概率colorJitter一张图像
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), # 以50的概率Sobel滤波一张图像(提取边缘)
    transforms.RandomApply([Sobel()], p=0.5), # 以50的概率Sobel滤波一张图像(提取边缘)
    transforms.ToTensor(),
])

# train_dataset = TwoViewMNIST(root="./dataset", train=True, transform=transform_train, download=True)
from torchvision.datasets import MNIST# CIFAR10
train_dataset = MNIST(root="./dataset", train=True, transform=TwoViews(transform_train,2), download=True)
Ntr = 6000
torch.manual_seed(0)
chosen_tr = torch.randperm(train_dataset.data.size(dim=0))[:Ntr]

train_dataset.data = train_dataset.data[chosen_tr,:,:] # [Ntr, 28, 28] float32  /255  why加上255就解压trainloader时乱码？？？
train_dataset.targets = train_dataset.targets[chosen_tr]
train_dataset.targets.requires_grad = False
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=300, shuffle=False)

simclr = SimCLR(tau=0.1, out_feas=128, gray=True, device=device)
# pre_trained_dir = "./models/simclr_unsup_tau0.1_Adam_lr0.0003_decay0.0002_Ntr6000_eph10.pth"
# simclr.load_state_dict(torch.load(pre_trained_dir,map_location=device))
simclr.fit(trainloader, max_epoch=200,lr=3e-4,start_epoch=0) # int(pre_trained_dir[-6:-4])

