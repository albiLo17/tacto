import torchvision
import torch
from torch import nn
import torch.nn.functional as F


## TACTO IMPLEMENTATION ###
class Model(nn.Module):
    def __init__(self, fields):
        super(Model, self).__init__()

        self.fields = fields

        for k in self.fields:
            # Load base network
            net = self.get_base_net()
            net_name = "net_{}".format(k)

            # Add for training
            self.add_module(net_name, net)

        self.nb_feature = 512
        self.fc1 = nn.Linear(self.nb_feature * len(fields), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def get_base_net(self):
        # Load pre-trained resnet-18
        net = torchvision.models.resnet18(pretrained=True)

        # Remove the last fc layer, and rebuild
        modules = list(net.children())[:-1]
        net = nn.Sequential(*modules)

        return net

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
            embedding = embedding.view(-1, self.nb_feature)

            features.append(embedding)

        # Concatenate embeddings
        emb_fusion = torch.cat(features, dim=1)

        # Add fc layer for final prediction
        output = self.fc1(emb_fusion)
        output = self.fc2(F.relu(output))
        output = self.fc3(F.relu(output))

        return output

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))
