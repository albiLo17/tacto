import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from experiments.grasp_stability.albi.models.encoders import ImageEncoder
from experiments.grasp_stability.albi.models.decoders import ConvDecoder
from experiments.grasp_stability.albi.models.layers import Classifier, Flatten



class Multimodal(nn.Module):
    def __init__(self, args, fields):
        super(Multimodal, self).__init__()
        self.fields = fields
        self.z_dim = args.z_dim
        self.deterministic = args.deterministic

        # Encoders for each modality
        for k in self.fields:
            if self.deterministic == 1:
                net = ImageEncoder(z_dim=self.z_dim, deterministic=self.deterministic)
            net_name = "net_{}".format(k)

            # Add for training
            self.add_module(net_name, net)

        self.flatten = Flatten()

        # Classifier (assuming that datafusion is concatenation and deterministic)
        self.classifier = Classifier(infeatures=args.z_dim*len(self.fields))

    def forward(self, x):
        features = []

        # Get stored modules/networks
        nets = self.__dict__["_modules"]

        for k in self.fields:
            # Get the network by name
            net_name = "net_{}".format(k)
            net = nets[net_name]

            # Forward
            embedding = net(x[k])

            features.append(embedding)

            # Concatenate embeddings
        emb_fusion = torch.cat(features, dim=1)

        output = self.classifier(emb_fusion)

        return output






# class AE(nn.Module):
#     def __init__(self, args, fields):
#         super(AE, self).__init__()
#         self.fields = fields
#
#         self.encoder = ResNetEncoder()
#         #TODO: is something missin in between?
#         self.decoder = ConvDecoder(args, zsize=48)
#
#
#     def forward(self, x):
#         for k in self.fields:
#             x = self.encoder(x[k])
#             x = self.decoder(x)
#
#         return x




