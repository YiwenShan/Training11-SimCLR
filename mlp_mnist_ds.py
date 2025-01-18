import torch

class MLP_ds(torch.nn.Module):
    def __init__(self, in_feas, out_feas, device="cuda:0"):
        super(self.__class__, self).__init__()
        self.hid_num = 2048
        self.encoder = torch.nn.Linear(in_features=in_feas, out_features=self.hid_num, bias=True).to(device)
        self.fc = torch.nn.Linear(in_features=self.hid_num,out_features=out_feas, bias=True).to(device)
        # self.net = torch.nn.Sequential(
        #     torch.nn.Linear(in_features=in_feas, out_features=self.hid_num, bias=True),
        #     torch.nn.Linear(in_features=self.hid_num,out_features=out_feas, bias=True),
        # ).to(device)
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, X): # X:[B,1,H,W]
        x = torch.nn.Flatten()(X) # [B,1,H,W] -> [B,d]
        return self.fc(self.encoder(x))
    
    def fit(self, trainloader, lr=1e-3, weight_decay=2e-4, epoches=500, device="cuda:0"):
        print("...Training...")
        print("bs={:d}  lr={:.3f}  decay={:.4f}".format(trainloader.batch_size, lr, weight_decay))

        opt = torch.optim.Adam(self.fc.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(1,epoches+1):
            self.train()
            Tr_loss = 0
            for X, labels in trainloader:
                X,labels = X.to(device), labels.to(device) # X:[B,1,H,W]  labels:[B]
                y = self.forward(X) # pred_val:[B,cls]
                loss = torch.nn.CrossEntropyLoss()(y, labels)
                Tr_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            if epoch%10==0:
                # torch.save(self.state_dict(), "./models/mlp/smp"+str(trainloader.dataset.data.shape[0])\
                #            +"_"+str(opt.__class__)[-6:-2]+"_bs"+str(trainloader.batch_size)+"_lr"+str(lr)\
                #            +"_decay"+str(weight_decay)+"_eph"+str(epoch)+".pth")
                self.eval()
                with torch.no_grad():
                    total_correct_1, total_correct_5, total_num = 0.0, 0.0, 0
                    for X, labels in trainloader:
                        X, labels = X.to(device), labels.to(device)
                        total_num += X.size(dim=0)

                        pred_val = self.forward(X)
                        _, pred_cls = torch.topk(pred_val, k=5, dim=-1, largest=True)
                        top1_acc = torch.sum(( pred_cls[:,0] == labels ).float()).item() # item():tensor->float
                        top5_acc = torch.sum(( pred_cls[:,0:5] == labels.unsqueeze(-1).repeat(1,5) ).any(dim=-1)).item()
                        total_correct_1 += top1_acc
                        total_correct_5 += top5_acc
                    print("epoch{:3d}: InfoNCE Loss={:e} ".format(epoch, Tr_loss),
                          "Top1 acc={:.4f}%".format(total_correct_1/total_num*100),
                          "Top5 acc={:.4f}%".format(total_correct_5/total_num*100))



from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
train_dataset = MNIST(root="./dataset", train=True, transform=ToTensor(), download=True)
Ntr = 60000
torch.manual_seed(0)
chosen_tr = torch.randperm(train_dataset.data.size(dim=0))[:Ntr]

train_dataset.data = train_dataset.data[chosen_tr,:,:] # [Ntr, 28, 28] float32
train_dataset.targets = train_dataset.targets[chosen_tr]
train_dataset.targets.requires_grad = False
trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=256, shuffle=True)

mlp = MLP_ds(in_feas=784, out_feas=128)
mlp.load_state_dict(torch.load("./models/mlp/N6000_Adam_bs300_lr0.0003_decay0.0002_eph200.pth",map_location="cuda:0"), strict=False)
mlp.fc = torch.nn.Linear(mlp.hid_num, 10, bias=True).to("cuda:0")
mlp.fit(trainloader,lr=1e-3, weight_decay=2e-4, epoches=200, device="cuda:0")

