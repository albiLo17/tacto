# from comet_ml import Experiment
import torch
from torch import optim
import torch.nn as nn
import numpy as np

from arguments import get_argparse
import utils
from logger import tensorboard_logger
from experiments.grasp_stability.albi.models.models import Multimodal
from trainers import train, evaluation

from dataset import get_dataloader

from tqdm import tqdm


fieldsList = [
    # ["tactileColorL", "tactileDepthL", "visionColor"],
    # ["tactileColorL", "tactileColorR", "visionColor"],
    ["tactileColorL",  "visionColor"],
    ["visionColor"],
    ["tactileColorL", ],
    # ["tactileColorL", "tactileColorR"],
    # ["tactileDepthL"],
    # ["tactileColorL", "visionDepth"],
    # ["tactileColorL", "tactileDepthL"],
]

# 0 is the VAE representation while 1 is the AE
paths = [
        {"visionColor": './experiments/representations/checkpoint/E2E_MULTIREP_AE_modality=0/E2E_MULTIREP_AE_modality=0.pt',
                 "tactileColorL": './experiments/representations/checkpoint/E2E_MULTIREP_AE_modality=0/E2E_MULTIREP_AE_modality=0.pt'},
        {"visionColor": './experiments/representations/checkpoint/AE_modality=0/AE_modality=0.pt',
                 "tactileColorL": './experiments/representations/checkpoint/AE_modality=1/AE_modality=1.pt'},
        {"visionColor": './experiments/representations/checkpoint/AE_modality=0/AE_modality=0.pt',
                 "tactileColorL": './experiments/representations/checkpoint/AE_modality=1/AE_modality=1.pt'}
        ]

args = get_argparse()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'



if __name__ == '__main__':

    for det in range(2):
        args.deterministic = 1

        for mod in range(3):
            args.modality = mod

            run_name = utils.get_run_name(args)
            writer = tensorboard_logger(args)

            path = paths[mod]
            if args.deterministic == 0:
                for key, value in path.items():
                    path[key] = value.replace('AE', 'VAE')

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
                model = Multimodal(args, fieldsList[args.modality], paths=path).to(device)
                # DEF optimizer
                optimizer = optim.Adam(model.parameters(), lr=args.lr)

                train_losses = []
                test_losses = []
                test_accuracies = []

                for epoch in tqdm(range(args.epochs)):
                    _, training_loss = train(args, trainLoader, model, optimizer, device,  modality=fieldsList[args.modality], criterion=nn.CrossEntropyLoss(), classification=True)
                    acc, test_loss = evaluation(args, testLoader, model, device, modality=fieldsList[args.modality], criterion=nn.CrossEntropyLoss(), classification=True)

                    print(f'EPOCH {epoch} - Test Loss: {test_loss}   Accuracy: {acc}')

                    train_losses.append(training_loss)
                    test_losses.append(test_loss)
                    test_accuracies.append(acc)

                    writer.save_losses(epoch, train_losses, test_losses, test_accuracies)
                    writer.add_scalar(training_loss, epoch, "Trainig loss")
                    writer.add_scalar(test_loss, epoch, "Test loss")
                    writer.add_scalar(acc, epoch, "Test accuracy")

                    writer.update_best_loss(test_loss, model)
                    print()




    print()