import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from experiments.grasp_stability.albi.models.models_utils import init_weights, activation_func
from experiments.grasp_stability.albi.models.layers import Flatten, conv2d, deconv


class ConvDecoder(nn.Module):
    def __init__(self, args, zsize=48):
        super(ConvDecoder, self).__init__()

        self.batch_size = args.batch_size

        self.dfc3 = nn.Linear(zsize, 4096)
        self.bn3 = nn.BatchNorm2d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm2d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = nn.BatchNorm2d(256 * 6 * 6)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding=2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride=4, padding=4)

    def forward(self, x):  # ,i1,i2,i3):

        x = self.dfc3(x)
        # x = F.relu(x)
        x = F.relu(self.bn3(x))

        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # x = F.relu(x)
        # print(x.size())
        x = x.view(self.batch_size, 256, 6, 6)
        # print (x.size())
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv5(x)
        # print x.size()
        x = F.relu(x)
        # print x.size()
        x = F.relu(self.dconv4(x))
        # print x.size()
        x = F.relu(self.dconv3(x))
        # print x.size()
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv2(x)
        # print x.size()
        x = F.relu(x)
        x = self.upsample1(x)
        # print x.size()
        x = self.dconv1(x)
        # print x.size()
        x = F.sigmoid(x)
        # print x
        return x


class ImageDecoder(nn.Module):
    def __init__(self, fields, z_dim, deterministic, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.fields = fields
        self.z_dim = z_dim
        self.deterministic = deterministic

        self.lin1 = nn.Linear(self.z_dim, self.z_dim*4*4)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(self.z_dim, 4, 4))

        self.img_deconv6 = deconv(self.z_dim, 128, kernel_size=3, stride=2)
        self.img_deconv5 = deconv(128, 64, kernel_size=3, stride=2)
        self.img_deconv4 = deconv(64, 64, kernel_size=3, stride=2)
        self.img_deconv3 = deconv(64, 32, kernel_size=5, stride=2)
        self.img_deconv2 = deconv(32, 16, kernel_size=5, stride=2)
        self.img_deconv1 = deconv(16, 3, kernel_size=7, stride=2)

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):

        out_img_lin = self.act1(self.lin1(image))
        out_img_unflat = self.unflatten(out_img_lin)

        # image encoding layers
        out_img_deconv6 = self.img_deconv6(out_img_unflat)
        out_img_deconv5 = self.img_deconv5(out_img_deconv6)
        out_img_deconv4 = self.img_deconv4(out_img_deconv5)
        out_img_deconv3 = self.img_deconv3(out_img_deconv4)
        out_img_deconv2 = self.img_deconv2(out_img_deconv3)
        out_img_deconv1 = self.img_deconv1(out_img_deconv2)

        # img_out_convs = (
        #     out_img_conv1,
        #     out_img_conv2,
        #     out_img_conv3,
        #     out_img_conv4,
        #     out_img_conv5,
        #     out_img_conv6,
        # )

        # image embedding parameters
        # flattened = self.flatten(out_img_deconv1)
        # img_out = self.img_encoder(flattened)
        img_out = torch.sigmoid(out_img_deconv1)

        return img_out #, img_out_convs