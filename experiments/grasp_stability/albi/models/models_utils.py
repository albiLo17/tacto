import torch.nn as nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from collections import OrderedDict


def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()




def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v



def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ELBO(nn.Module):
    def __init__(self, args, fields=None):
        super(ELBO, self).__init__()
        self.fields = fields
        self.batch_size = args.batch_size

    def forward(self, output, ground_truth):

        reconstruction, mu, logvar = output

        # Reconstruction loss
        # rec_crit = nn.MSELoss()
        rec_crit = nn.BCELoss(reduction='sum')
        if self.fields != None:
            if len(self.fields) == 1:
                rec_loss = rec_crit(reconstruction, ground_truth)
            else:
                rec_loss = {}
                for k in self.fields:
                    rec_loss[k] = rec_crit(reconstruction[k], ground_truth[k])

        # Regularization term
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= self.batch_size
        # print()

        return [rec_loss, kl_loss]
        # return rec_loss+kl_loss

class MSE(nn.Module):
    def __init__(self, args, fields=None):
        super(MSE, self).__init__()
        self.fields = fields
        self.batch_size = args.batch_size

    def forward(self, output, ground_truth):

        reconstruction = output

        # Reconstruction loss
        # rec_crit = nn.MSELoss()
        rec_crit = nn.MSELoss()
        if self.fields != None:
            rec_loss = {}
            for k in self.fields:
                rec_loss[k] = rec_crit(reconstruction[k], ground_truth[k])


        return rec_loss


def get_dict(key_transformation, path):
    model = torch.load(path)
    state_dict = model.state_dict()

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace(f'{key_transformation}.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def product_of_experts(m_vect, v_vect):

    T_vect = 1.0 / v_vect

    mu = (m_vect * T_vect).sum(2) * (1 / T_vect.sum(2))
    var = 1 / T_vect.sum(2)

    return mu, torch.log(var)


def encode(fields, nets, x, deterministic):
    features = []
    m_vec = None
    var_vec = None

    for k in fields:
        # Get the network by name
        net_name = "net_{}".format(k)
        net = nets[net_name]

        # Forward
        embedding = net(x[k])

        if deterministic == 1:
            features.append(embedding)
        else:
            mu, logvar = gaussian_parameters(embedding)
            m_vec = torch.cat([m_vec, mu.unsqueeze(2)], dim=2) if m_vec != None else mu.unsqueeze(2)
            var_vec = torch.cat([var_vec, logvar.exp().unsqueeze(2)], dim=2) if var_vec != None else logvar.exp().unsqueeze(2)

    return features, m_vec, var_vec


def decode(fields, nets, emb_fusion):
    decoded = {}
    # Decode
    for k in fields:
        # Get the network by name
        net_name = "dec_{}".format(k)
        net = nets[net_name]

        # decode
        decoded[k] = net(emb_fusion)

    return decoded