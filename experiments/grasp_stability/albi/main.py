import os
# from comet_ml import Experiment
import torch
from torch import optim
import numpy as np

from arguments import get_argparse
import utils
from logger import tensorboard_logger
from models import Model
from trainers import train, evaluation

from dataset import get_dataloader

import glob
from tqdm import tqdm


fieldsList = [
    # ["tactileColorL", "tactileDepthL", "visionColor"],
    ["tactileColorL", "tactileColorR", "visionColor"],
    ["visionColor"],
    ["tactileColorL", "tactileColorR"],
    # ["tactileDepthL"],
    # ["tactileColorL", "visionDepth"],
    # ["tactileColorL", "tactileDepthL"],
]

args = get_argparse()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

run_name = utils.get_run_name(args)
writer = tensorboard_logger(args)

if __name__ == '__main__':
    # K fold cross validation
    K = 5

    # Accuracies
    accs = []

    for i in range(K):
        print(f"Fold {i}/{K}")
        trainLoader, testLoader = get_dataloader(args, K, i, modality=fieldsList[args.modality])

        # DEF MODEL
        model = Model(fieldsList[args.modality]).to(device)
        # DEF optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_losses = []
        test_losses = []

        for epoch in tqdm(range(args.epochs)):
            losses_list, training_loss = train(trainLoader, model, optimizer, epoch, args, device, writer, modality=fieldsList[args.modality])
            acc = evaluation(testLoader, model, optimizer, epoch, args, device, writer, modality=fieldsList[args.modality])
            print()




    print()