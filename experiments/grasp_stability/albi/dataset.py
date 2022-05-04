import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

import glob
import os
import deepdish as dd

from utils import AddGaussianNoise


class GraspingDataset(Dataset):
    def __init__(self, fileNames, fields=[], transform=None, transformDepth=None):
        self.transform = transform
        self.transformDepth = transformDepth
        self.fileNames = fileNames
        self.fields = fields + ["label"]
        self.numGroup = 100  # data points per file

        self.dataList = None
        self.dataFileID = -1

    def __len__(self):
        return len(self.fileNames * self.numGroup)

    def load_data(self, idx):
        dirName = self.fileNames[idx]
        data = {}

        for k in self.fields:
            fn = dirName.split("/")[-1]

            fnk = "{}_{}.h5".format(fn, k)

            filenamek = os.path.join(dirName, fnk)
            d = dd.io.load(filenamek)

            data[k] = d

        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fileID = idx // self.numGroup
        if fileID != self.dataFileID:
            self.dataList = self.load_data(fileID)
            self.dataFileID = fileID

        sample = {}

        data = self.dataList

        for k in self.fields:
            d = data[k][idx % self.numGroup]

            if k in ["tactileColorL", "tactileColorR", "visionColor"]:
                d = d[:, :, :3]
                # print(k, d.min(), d.max())

            if k in ["tactileDepthL", "tactileDepthR", "visionDepth"]:
                d = np.dstack([d, d, d])

            if k in ["tactileDepthL", "tactileDepthR"]:
                d = d / 0.002 * 255
                d = np.clip(d, 0, 255).astype(np.uint8)
                # print("depth min", d.min(), "max", d.max())

            if k in ["visionDepth"]:
                d = (d * 255).astype(np.uint8)

            if k in [
                "tactileColorL",
                "tactileColorR",
                "visionColor",
                "visionDepth",
            ]:
                if self.transform:
                    d = self.transform(d)

            if k in [
                "tactileDepthL",
                "tactileDepthR",
            ]:
                # print("before", d.min(), d.max(), d.mean(), d.std())
                d = self.transformDepth(d)
                # d = (d + 2) / 0.05
                # print("after", d.min(), d.max(), d.mean(), d.std())

            sample[k] = d

        return sample


def get_dataloader(args, K, i, modality):
    # K-fold, test the i-th fold, train the rest

    # Load data
    fileNames = glob.glob(os.path.join(args.data_dir, "*"))
    fileNames = sorted(fileNames)[: args.N]         # TODO: why removing the las ones??

    # Split K fold
    N = len(fileNames)
    n = N // K

    idx = list(range(N))
    testIdx = idx[n * i: n * (i + 1)]
    trainIdx = list(set(idx) - set(testIdx))

    trainFileNames = [fileNames[i] for i in trainIdx]
    testFileNames = [fileNames[i] for i in testIdx]

    trainTransform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            AddGaussianNoise(0.0, 0.01),
        ]
    )

    trainTransformDepth = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1,), std=(0.2,)),
            AddGaussianNoise(0.0, 0.01),
        ]
    )

    # Create training dataset and dataloader
    trainDataset = GraspingDataset(
        trainFileNames,
        fields=modality,
        transform=trainTransform,
        transformDepth=trainTransformDepth,
    )
    trainLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True
    )

    testTransform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )

    testTransformDepth = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1,), std=(0.2,)),
            # AddGaussianNoise(0.0, 0.01),
        ]
    )

    # Create training dataset and dataloader
    testDataset = GraspingDataset(
        testFileNames,
        fields=modality,
        transform=testTransform,
        transformDepth=testTransformDepth,
    )
    testLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=32, shuffle=False, num_workers=12, pin_memory=True
    )


    return trainLoader, testLoader
