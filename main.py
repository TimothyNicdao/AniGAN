import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils
# import keyboard  # using module keyboard
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


nz = 100 # size of z latent vector
ngf = 64 # size of feature maps
nc = 3
ngpu = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
noise = torch.randn(1, nz, 1, 1, device=device)

#Define generator model
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
    
# Create the generator
netG = Generator(ngpu).to(device)
netG = torch.load('generator.pt')
print('hi')
cpuDev = torch.device('cpu')

while True:  # making a loop
    with torch.no_grad():
        noise = torch.randn(3,nz, 1, 1, device=device)
        # image = torch.squeeze(netG(noise))
        image = netG(noise)
        # image = image.permute(1,2,0)
        vutils.save_image(image, 'myImages.jpg')
        # image = np.array(image.to(cpuDev))
        # print(image.shape)
        # image = Image.fromarray(image,'RGB')
        # image.resize((512,512))
        # print(image.shape)
        # image.show()
        # plt.imshow(image)
        input("Press Enter to continue...")