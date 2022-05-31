import torchvision
import torch
from torch import nn
import torch.nn.functional as F

from experiments.grasp_stability.albi.models.encoders import ImageEncoder
from experiments.grasp_stability.albi.models.decoders import ImageDecoder
from experiments.grasp_stability.albi.models.layers import Classifier, Flatten
from experiments.grasp_stability.albi.models.models_utils import gaussian_parameters, reparameterize, get_dict, product_of_experts
from experiments.grasp_stability.albi.models.models_utils import encode, decode


class Multimodal(nn.Module):
    def __init__(self, args, fields, paths=None):
        super(Multimodal, self).__init__()
        self.fields = fields
        self.z_dim = args.z_dim

        self.deterministic = args.deterministic
        self.e2e = args.e2e

        # Encoders for each modality
        for k in self.fields:
            net = ImageEncoder(z_dim=self.z_dim, deterministic=self.deterministic)
            net_name = "net_{}".format(k)

            # Load pretrained representations
            if self.e2e == 0:
                net.load_state_dict(get_dict(key_transformation=net_name, path=paths[k]), strict=False)
                net.requires_grad = False

            # Add for training
            self.add_module(net_name, net)

        self.flatten = Flatten()

        # Classifier
        if self.deterministic == 1:
            # Concatenation
            self.classifier = Classifier(infeatures=args.z_dim*len(self.fields))
        else:
            # PoE
            self.classifier = Classifier(infeatures=args.z_dim)

    def forward(self, x):

        # Get stored modules/networks
        nets = self.__dict__["_modules"]

        features, m_vec, var_vec = encode(self.fields, nets, x, self.deterministic)

        # Concatenate embeddings
        if self.deterministic == 1:
            emb_fusion = torch.cat(features, dim=1)
        else:
            mu_fusion, logvar_fusion = product_of_experts(m_vec, var_vec)
            emb_fusion = reparameterize(mu_fusion, logvar_fusion)

        # Classify
        output = self.classifier(emb_fusion)

        return output

    def load_layers(self, path):
        print()






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

            # Add encoding for training
            self.add_module(net_name, net)

        # Decoders for each Modality
        for k in self.fields:
            net = ImageDecoder(self.fields, z_dim=self.z_dim, deterministic=self.deterministic)
            net_name = "dec_{}".format(k)

            # Add encoding for training
            self.add_module(net_name, net)


    def forward(self, x):
        # Get stored modules/networks
        nets = self.__dict__["_modules"]

        # Encode
        features, m_vec, var_vec = encode(self.fields, nets, x, self.deterministic)

        # Concatenate embeddings
        if self.deterministic == 1:
            emb_fusion = torch.cat(features, dim=1)
        else:
            mu_fusion, logvar_fusion = product_of_experts(m_vec, var_vec)
            emb_fusion = reparameterize(mu_fusion, logvar_fusion)

        # Encode
        decoded = decode(self.fields, nets, emb_fusion)

        if self.deterministic == 1:
            return decoded

        return [decoded, mu_fusion, logvar_fusion]      # TODO: is the fused embedding to be forced?




