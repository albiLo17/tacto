import torch.nn as nn
import torch
import math
from experiments.grasp_stability.albi.models.models_utils import init_weights, activation_func
from experiments.grasp_stability.albi.models.layers import Flatten, conv2d, ResidualBlock



class ImageEncoder(nn.Module):
    def __init__(self, z_dim, deterministic, initailize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim
        self.deterministic = deterministic

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)

        if self.deterministic == 1:
            self.img_encoder = nn.Linear(4 * 4 * self.z_dim, self.z_dim)
        else:
            self.img_encoder = nn.Linear(4 * 4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers
        # image = torch.sigmoid(image)
        out_img_conv1 = self.img_conv1(image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)

        # img_out_convs = (
        #     out_img_conv1,
        #     out_img_conv2,
        #     out_img_conv3,
        #     out_img_conv4,
        #     out_img_conv5,
        #     out_img_conv6,
        # )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened)

        return img_out #, img_out_convs

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))