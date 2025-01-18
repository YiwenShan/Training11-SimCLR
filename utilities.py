import torch
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image

class TwoViewMNIST(MNIST):
    def __getitem__(self, item):
        img,target=self.data[item],self.targets[item]
        # img = Image.fromarray(img)

        if self.transform is not None: # MNIST 的transform
            imgL = self.transform(transforms.ToPILImage()(img))
            imgR = self.transform(transforms.ToPILImage()(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target


class Sobel(object):
    # def __init__(self):

    def __call__(self, X):
        # X: PIL Image 
        if not torch.is_tensor(X): X = transforms.ToTensor()(X)
        dims = len(X.shape)
        if dims==2: 
            H,W = X.shape
            B,C = 1,1
            X = X.unsqueeze(0).unsqueeze(0)
        elif dims==3:
            C,H,W = X.shape
            B = 1
            X = X.unsqueeze(0)
        elif dims==4: B,C,H,W = X.shape
        else: raise ValueError("Invalid structure of dataset.")

        fx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        fx = fx.expand(B,C,3,3)
        fy = torch.tensor([[-1., -2. , -1.], [0., 0., 0.], [1., 2. , 1.]]) # [3,3]
        fy = fy.expand(B,C,3,3) # [B,C,3,3]

        Gx = torch.nn.functional.conv2d(X, fx, bias=None, stride=1,padding=1) # [B,C,H,W]
        Gy = torch.nn.functional.conv2d(X, fy, bias=None, stride=1,padding=1) # [B,C,H,W]
        magnitude = torch.sqrt(Gx*Gx + Gy*Gy) # [B,C,H,W]
        mag_v = magnitude.reshape((B,C,-1)) # [B,C, H*W]
        m,_ = torch.min(mag_v, dim=2) # [B,C]
        M,_ = torch.max(mag_v, dim=2) # [B,C]

        for bi in range(B):
            for ci in range(C):
                mag_v[bi,ci,:] = (mag_v[bi,ci,:]-m[bi,ci])/(M[bi,ci]-m[bi,ci] + 1e-16) # 不加1e-16会出nan
        magnitude = mag_v.reshape((B,C,H,W)) # [B,C,H,W]
        return transforms.ToPILImage()(magnitude.squeeze(0))
# import cv2
# cv2.imshow("Sobel", magnitude.squeeze().numpy())
# cv2.waitKey(0)

class TwoViews(object):
    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views
    def __call__(self, x):
        views = []
        for i in range(self.n_views):
            views.append(self.transform(x))
        return views
        # return [self.transform(x) for i in range(self.n_views)]


class LARS(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=.9, weight_decay=.0005, eta=0.001, max_epoch=200):
        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)
    
    def step(self):
        '''
        closure (callable): A closure that reevaluates the model and
        returns the loss. Optional for most optimizers.
        '''

        loss = None
        # if closure is not None: loss = closure() # ?
        self.epoch += 1

# opt.param_groups是list, 有1个dict元素, 即opt.param_groups[0]是dict
# opt.param_groups[0]['params']是list, 其余有opt.param_groups[0]['lr'],['eta']等
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']

            for p in group['params']:
                if p.grad is None: continue

                grad_norm = torch.norm(p.grad.data)
                weight_norm = torch.norm(p.data)
                decay = (1 - float(self.epoch) / max_epoch) ** 2
                global_lr = lr * decay
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                actual_lr = local_lr * global_lr

                param_state = self.state[p] # 哪有.state?
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                else: buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(actual_lr, p.grad.data + weight_decay * p.data)

                p.data.add_(-buf)
        return loss
