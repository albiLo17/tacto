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
# device = 'cpu'



if __name__ == '__main__':

    for mod in range(3):
        args.modality = mod

        run_name = utils.get_run_name(args)
        writer = tensorboard_logger(args)

        # K fold cross validation
        K = args.K

        # Accuracies
        accs = []


        if K > 0:
            k_fold = K
        else:
            K_fold = 1
            K = int(1/args.split)
            print(K)

        for i in range(K_fold):
            print(f"Fold {i+1}/{K_fold}")
            trainLoader, testLoader = get_dataloader(args, K, i, modality=fieldsList[args.modality])

            # DEF MODEL
            model = Model(args, fieldsList[args.modality]).to(device)
            # DEF optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            train_losses = []
            test_losses = []
            test_accuracies = []

            for epoch in tqdm(range(args.epochs)):
                _, training_loss = train(trainLoader, model, optimizer, epoch, args, device, writer, modality=fieldsList[args.modality])
                acc, test_loss = evaluation(testLoader, model, optimizer, epoch, args, device, writer, modality=fieldsList[args.modality])

                print(f'EPOCH {epoch} - Test Loss: {test_loss}   Accuracy: {acc}')

                train_losses.append(training_loss)
                test_losses.append(test_loss)
                test_accuracies.append(acc)

                writer.save_losses(train_losses, test_losses, test_accuracies, epoch)
                writer.add_scalar(training_loss, epoch, "Trainig loss")
                writer.add_scalar(test_loss, epoch, "Test loss")
                writer.add_scalar(acc, epoch, "Test accuracy")

                writer.update_best_loss(test_loss, model)
                print()




    print()