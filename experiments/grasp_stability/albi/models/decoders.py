import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from experiments.grasp_stability.albi.models.models_utils import init_weights, activation_func
from experiments.grasp_stability.albi.models.layers import Flatten, conv2d, deconv



class ImageDecoder(nn.Module):
    def __init__(self, fields, z_dim, deterministic, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.fields = fields
        self.z_dim = z_dim
        self.deterministic = deterministic
        mult = 1*(1-self.deterministic) + len(self.fields)*(self.deterministic)

        self.lin1 = nn.Linear(self.z_dim*mult, self.z_dim*4*4)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(self.z_dim, 4, 4))

        self.img_deconv6 = deconv(self.z_dim, 128, kernel_size=3, stride=2)
        self.img_deconv5 = deconv(128, 64, kernel_size=3, stride=2)
        self.img_deconv4 = deconv(64, 64, kernel_size=5, stride=2)
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
        img_out = torch.sigmoid(out_img_deconv1)

        return img_out #, img_out_convs