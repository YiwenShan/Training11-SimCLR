from torchvision import datasets, transforms
import torch
from utilities import Sobel
from SimCLR_model import SimCLR_DownStream
device = ("cuda:0" if torch.cuda.is_available else "cpu")

train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
Ntr = 60000
torch.manual_seed(0)
chosen_tr = torch.randperm(train_dataset.data.size(dim=0))[:Ntr]

train_dataset.data = train_dataset.data[chosen_tr,:,:] # [Ntr, 28, 28] float32  /255
train_dataset.targets = train_dataset.targets[chosen_tr]
train_dataset.targets.requires_grad = False
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=256, shuffle=True)

simclr = SimCLR_DownStream(out_feas=10, device=device)
simclr.load_state_dict(torch.load("./models/train_noSobel/simclr_unsup_tau0.1_Adam_lr0.0003_decay0.0002_Ntr6000_eph190.pth",map_location=device),strict=False)
simclr.fit(trainloader,max_epoch=200,lr=1e-3)
