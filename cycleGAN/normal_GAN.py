from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
import torchvision
import glob
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn as nn
import matplotlib as plt
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from multiprocessing import freeze_support



print(torch.__version__)
print(torch.cuda.is_available())


workspace_dir = './'

# 预处理数据集
class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

        print("num_samples", self.num_samples) # ans:0

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        # print(fname)
        img = cv2.imread(fname)
        img = self.BGR2RGB(img) #because "torchvision.utils.save_image" use RGB
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 读取数据集
def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))

    # resize the image to (64, 64)
    # linearly map [0, 1] to [-1, 1]
    transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.Resize((64, 64)), # 图片大小，可以改！
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    dataset = FaceDataset(fnames, transform)
    return dataset

# get_dataset(workspace_dir)

# 生成随机数
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 随机化初始参数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())
        self.apply(weights_init)
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y

class Discriminator(nn.Module):
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid())
        self.apply(weights_init)
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


# 训练过程


if __name__ == '__main__':
# if(1):

    # hyperparameters
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 50
    save_dir = os.path.join(workspace_dir, 'logs')
    os.makedirs(save_dir, exist_ok=True)

    # model
    G = Generator(in_dim=z_dim).to("cuda")
    D = Discriminator(3).to("cuda")
    G.train()
    D.train()

    print("G param:",sum(x.numel() for x in G.parameters()))
    print("D param:",sum(x.numel() for x in D.parameters()))

    # loss criterion
    criterion = nn.BCELoss()

    # optimizer
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))


    same_seeds(0)
    # dataloader (You might need to edit the dataset path if you use extra dataset.)
    dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # show picture: test
    # import matplotlib.pyplot as plt
    # imagee = dataset[1].numpy()
    # imagee = (imagee + 1) / 2
    #
    # plt.imshow(imagee.transpose(1,2,0))
    # plt.show()


    # start training!!!
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    if(1):
        for e, epoch in enumerate(range(n_epoch)):

            counter = 0 # counter
            for i, data in enumerate(dataloader):
                counter += 1

                imgs = data
                imgs = imgs.to("cuda")

                bs = imgs.size(0) #batch size

                """ Train D """
                z = Variable(torch.randn(bs, z_dim)).to("cuda")
                r_imgs = Variable(imgs).to("cuda")
                f_imgs = G(z)

                # label
                r_label = torch.ones((bs)).to("cuda")
                f_label = torch.zeros((bs)).to("cuda")

                # dis
                r_logit = D(r_imgs.detach())
                f_logit = D(f_imgs.detach())

                # compute loss
                r_loss = criterion(r_logit, r_label)
                f_loss = criterion(f_logit, f_label)
                loss_D = (r_loss + f_loss) / 2

                # update model
                D.zero_grad()
                loss_D.backward()
                opt_D.step()

                """ train G """
                # repeat n times
                n = 3
                for t in range(n):
                    # leaf
                    z = Variable(torch.randn(bs, z_dim)).to("cuda")
                    f_imgs = G(z)

                    # dis
                    f_logit = D(f_imgs)

                    # compute loss
                    loss_G = criterion(f_logit, r_label)

                    # update model
                    G.zero_grad()
                    loss_G.backward()
                    opt_G.step()

                # log

                print(
                    f'\rEpoch [{epoch + 1}/{n_epoch}] {i + 1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}',
                    end='')

                # save some imgs
                if(i % 10 == 9):
                    z_sample = Variable(torch.randn(100, z_dim)).to("cuda")
                    cnt_img = (G(z_sample).data[1] + 1) / 2.0
                    filename = os.path.join(save_dir, f'{epoch:02d}{i:05d}.jpg')
                    torchvision.utils.save_image(cnt_img, filename)



            G.eval()

            f_imgs_sample = (G(z_sample).data + 1) / 2.0
            filename = os.path.join(save_dir, f'Epoch_{epoch + 1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            print(f' | Save some samples to {filename}.')

            G.train()

            if (e + 1) % 5 == 0:
                torch.save(G.state_dict(), os.path.join(workspace_dir, f'dcgan_g.pth'))
                torch.save(D.state_dict(), os.path.join(workspace_dir, f'dcgan_d.pth'))

    #print(prof)
