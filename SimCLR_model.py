from torch import nn
from torchvision import models
# from ResNet import BasicBlock,ResNet_bkbone
# from utilities import LARS
import torch
import matplotlib.pyplot as plt

class SimCLR(nn.Module):
    def __init__(self, out_feas=128, tau=0.1, gray=False, device="cuda:0"):
        super(self.__class__, self).__init__()

        # ResNet18  最后一层是.fc: Linear(512,1000, bias=True)
        # self.backbone = ResNet_bkbone(block=BasicBlock, layers=[2,2,2,2],grayscale=True).to(device)
        self.backbone = models.resnet50(pretrained=False).to(device)
        if gray: self.backbone.conv1 = torch.nn.Conv2d(1,64,kernel_size=7,stride=2, padding=3, bias=False).to(device)

        # 替换resnet最后的全连接层
        mlp_in_feas = self.backbone.fc.in_features # resnet18的bkbong输出  resnet50是2048
        self.backbone.fc = nn.Sequential(nn.Linear(mlp_in_feas, mlp_in_feas,bias=False),
                                        nn.ReLU(),
                                        nn.Linear(mlp_in_feas, out_feas)).to(device)

        self.loss_fn = nn.CrossEntropyLoss().to(device)
        self.views = 2
        self.tau = tau

    def forward(self, X): # X:[B,1,28,28](MNIST)
        z = self.backbone(X) # h:[B,512]
        # z = self.mlp(h) # z:[B,out_feas]
        return z

    def infoNCE(self, features): # features:[2bs,d]
        device = features.device
        bs = features.size(dim=0)//2

        labels = torch.cat([torch.arange(bs) for i in range(self.views)], dim=0) # [2bs]
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # [2*bs,2*bs]
        labels = labels.to(device)

        features = torch.nn.functional.normalize(features, dim=1) # 归一化后, torch.sum(features*features, dim=1) = 全1
        sim = torch.matmul(features, features.T) # [2*bs,2*bs] [i,j]: (xi.T * xj)/(||xi||2 * ||xj||2)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device) # [2bs,2bs] bool单位阵
        labels = labels[~mask].view(labels.shape[0], -1) # [2bs, 2bs-1] 等价于: labels挖去对角线元素, 右上角矩阵向左移1格
        sim = sim[~mask].view(sim.shape[0], -1) # [2bs, 2bs-1]

        # select and combine multiple positives
        positives = sim[labels.bool()].view(labels.shape[0], -1) # [2bs,1]
        negatives = sim[~labels.bool()].view(sim.shape[0], -1) # [2bs,2bs-2]
        logits = torch.cat([positives, negatives], dim=1) # [2bs,2bs-1] 把simi_mat中labels==True的位置(共2bs个)移到第一列
        logits = logits / self.tau

        targets = torch.zeros(logits.shape[0], dtype=torch.long).to(device) # [2bs]
        return logits, targets # labels全0

    def fit(self, trainloader, lr=1e-3, momentum=0.9, weight_decay=2e-4, max_epoch=500,start_epoch=0, device="cuda:0"):
        print("...Training...")
        print("lr={:e}  mom={:.2f}  wei_decay={:e}  epoches={:d}  bs={:d}"\
              .format(lr,momentum,weight_decay,max_epoch,trainloader.batch_size))
        # optimizer = LARS(self.parameters(), lr,momentum,weight_decay,eta,max_epoch)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=weight_decay) # ,momentum=momentum
        for epoch in range(1+start_epoch,max_epoch+1):
            self.train()
            Tr_loss = 0
            for X,_ in trainloader:
                X = torch.cat(X, dim=0).to(device) # list:[[bs,C,H,W],[bs,C,H,W]] -> Tensor:[2bs,C,H,W]
            # for (imgL,imgR,labels) in trainloader:
            #     X = torch.cat((imgL, imgR), dim=0).to(device)
                z = self.forward(X) # [2bs,d]
                logits,targets = self.infoNCE(z) #logits:[2bs,2bs-1]  targets:[2bs]
                loss = self.loss_fn(logits, targets)
                Tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("epoch {:3d}  InfoNCE Loss={:e}".format(epoch, Tr_loss))

            if (epoch) % 10==0:
                torch.save(self.state_dict(), "./models/simclr_unsup_tau"+str(self.tau)+"_"+str(optimizer.__class__)[-6:-2]+"_lr"+str(lr)\
                        +"_decay"+str(weight_decay)+"_Ntr"+str(trainloader.dataset.data.size(0))+"_eph"+str(epoch)+".pth")
        torch.save(self.state_dict(), "./models/simclr_unsup_tau"+str(self.tau)+"_"+str(optimizer.__class__)[-6:-2]+"_lr"+str(lr)\
                +"_decay"+str(weight_decay)+"_Ntr"+str(trainloader.dataset.data.size(0))+"_eph"+str(epoch)+".pth")


class SimCLR_DownStream(nn.Module):
    def __init__(self, out_feas=128, gray=True, device="cuda:0"):
        super(self.__class__, self).__init__()

        # self.backbone = SimCLR().backbone.to(device)
        self.backbone = models.resnet50(pretrained=False).to(device)
        mlp_in_feas = self.backbone.fc.in_features
        if gray: self.backbone.conv1 = torch.nn.Conv2d(1,64,kernel_size=7,stride=2, padding=3, bias=False).to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False # 冻结网络f, 训练过程中不更新f的参数

        self.backbone.fc = nn.Linear(in_features=mlp_in_feas, out_features=out_feas, bias=True).to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(device)

    def forward(self, X): # X:[B,1,28,28](MNIST)
        z = self.backbone(X) # h:[B,512]
        # z = self.mlp(h) # z:[B,out_feas];
        return z

    def fit(self, trainloader, lr=1e-3, weight_decay=2e-4, max_epoch=500, device="cuda:0"):
        print("...Training...\nlr={:.4f}  wei_decay={:.4f}  epoches={:d}  bs={:d}"\
              .format(lr,weight_decay,max_epoch,trainloader.batch_size))

        optimizer = torch.optim.Adam(self.backbone.fc.parameters(), lr=lr,weight_decay=weight_decay) # 
        for epoch in range(1,max_epoch+1):
            self.train()
            Tr_loss = 0
            for X,labels in trainloader:
                X, labels = X.to(device), labels.to(device)
                z = self.forward(X) # [bs,cls]
                loss = self.loss_fn(z, labels) # cross entropy
                Tr_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch%10==0:
                torch.save(self.state_dict(), "./models/down_stream/"+str(optimizer.__class__)[-6:-2]\
                           +"_lr"+str(lr)+"_decay"+str(weight_decay)+"_eph"+str(epoch)+".pth")
                self.eval()
                with torch.no_grad():
                    total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
                    for data, labels in trainloader:
                        data, labels = data.to(device), labels.to(device)
                        total_num += data.size(dim=0)

                        pred_cls = self.forward(data)
                        _, pred_cls = torch.topk(pred_cls, k=5, dim=-1, largest=True)
                        top1_acc = torch.sum(( pred_cls[:,0] == labels ).float()).item() # item():tensor->float
                        top5_acc = torch.sum(( pred_cls[:,0:5] == labels.unsqueeze(-1).repeat(1,5) ).any(dim=-1)).item()
                        total_correct_1 += top1_acc
                        total_correct_5 += top5_acc
                    print("epoch{:3d}: InfoNCE Loss={:e} ".format(epoch, Tr_loss),
                          "Top1 acc={:.4f}%".format(total_correct_1/total_num*100),
                          "Top5 acc={:.4f}%".format(total_correct_5/total_num*100))


