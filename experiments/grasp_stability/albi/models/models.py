import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from experiments.grasp_stability.albi.models.encoders import ImageEncoder
from experiments.grasp_stability.albi.models.decoders import ImageDecoder
from experiments.grasp_stability.albi.models.layers import Classifier, Flatten
from experiments.grasp_stability.albi.models.models_utils import gaussian_parameters, reparameterize



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






class AE(nn.Module):
    def __init__(self, args, fields):
        super(AE, self).__init__()
        self.fields = fields
        self.z_dim = args.z_dim
        self.deterministic = args.deterministic

        # Encoders for each modality
        for k in self.fields:
            net = ImageEncoder(z_dim=self.z_dim, deterministic=self.deterministic)
            net_name = "net_{}".format(k)

            # Add for training
            self.add_module(net_name, net)

        # if self.deterministic == 0:
        #     # TODO: needed? Only if you want to add a learned prior?
        #     # zero centered, 1 std normal distribution
        #     self.z_prior_m = torch.nn.Parameter(
        #         torch.zeros(1, self.z_dim), requires_grad=False
        #     )
        #     self.z_prior_v = torch.nn.Parameter(
        #         torch.ones(1, self.z_dim), requires_grad=False
        #     )
        #     self.z_prior = (self.z_prior_m, self.z_prior_v)


        self.decoder = ImageDecoder(self.fields, z_dim=self.z_dim, deterministic=self.deterministic)


    def forward(self, x):
        # Get stored modules/networks
        nets = self.__dict__["_modules"]

        for k in self.fields:
            # Get the network by name
            net_name = "net_{}".format(k)
            net = nets[net_name]

            # Forward
            encoded = net(x[k])
            if self.deterministic == 0:
                mu, logvar = gaussian_parameters(encoded)
                encoded = reparameterize(mu, logvar)        #TODO: check its device

            decoded = self.decoder(encoded)

        if self.deterministic == 1:
            return decoded

        return [decoded, mu, logvar]




