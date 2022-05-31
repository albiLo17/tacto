# from comet_ml import Experiment
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random

from arguments import get_argparse
import utils
from logger import tensorboard_logger
from experiments.grasp_stability.albi.models.models import AE
from experiments.grasp_stability.albi.models.models_utils import ELBO, MSE
from trainers import train, evaluation

from dataset import get_dataloader

from tqdm import tqdm


fieldsList = [
    # ["tactileColorL", "tactileDepthL", "visionColor"],
    # ["tactileColorL", "tactileColorR", "visionColor"],
    ["visionColor"],
    # ["tactileColorL", "tactileColorR"],
    # ["tactileDepthL"],
    # ["tactileDepthR"],
    ["tactileColorL"],
    # ["tactileColorR"],
    # ["tactileColorL", "visionDepth"],
    # ["tactileColorL", "tactileDepthL"],
]
# fieldsList = [
#     # ["tactileColorL", "tactileDepthL", "visionColor"],
#     # ["tactileColorL", "tactileColorR", "visionColor"],
#     # ["visionColor"],
#     # ["tactileColorL", "tactileColorR"],
#     ["tactileColorL", "visionColor"],
#     # ["tactileDepthL"],
#     # ["tactileDepthR"],
#     # ["tactileColorL"],
#     # ["tactileColorR"],
#     # ["tactileColorL", "visionDepth"],
#     # ["tactileColorL", "tactileDepthL"],
# ]

args = get_argparse()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'



if __name__ == '__main__':

    # for mod in range(2):
    for mod in range(1):
        args.modality = mod

        run_name = utils.get_run_name(args)
        print(run_name)
        writer = tensorboard_logger(args)

        # K fold cross validation
        K = args.K

        # Accuracies
        accs = []

        # Cross validation or train/test
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
            model = AE(args, fieldsList[args.modality]).to(device)
            # DEF optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            if args.deterministic == 1:
                criterion = MSE(args, fieldsList[args.modality])
            else:
                criterion = ELBO(args, fieldsList[args.modality])

            train_losses = []
            test_losses = []

            for epoch in tqdm(range(args.epochs)):
                _, training_loss = train(args, trainLoader, model, optimizer, device, modality=fieldsList[args.modality],
                                         criterion=criterion, classification=False)

                images, test_loss = evaluation(args, testLoader, model, device, modality=fieldsList[args.modality],
                                            criterion=criterion, classification=False)

                # print(f'EPOCH {epoch} - Test Loss: {test_loss} ')

                train_losses.append(training_loss)
                test_losses.append(test_loss)

                writer.save_losses(epoch, train_losses, test_losses, None)
                writer.add_scalar(training_loss[0], epoch, "Trainig representation loss")
                writer.add_reconstruction_scalar(training_loss[1], epoch, "Trainig reconstruction loss", fieldsList[args.modality])
                writer.add_scalar(training_loss[2], epoch, "Trainig kl loss")
                writer.add_scalar(test_loss, epoch, "Test representation loss")
                element = random.randint(0, len(images['ground_truth'])-1)
                for k in fieldsList[args.modality]:
                    writer.show_reconstructed_images(images['ground_truth'][k][element], images['predictions'][k][element], epoch, f"Test {k} reconstruction image")

                writer.update_best_loss(test_loss, model)
                # print()



