import argparse

def get_argparse(notebook=False, p=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', default="./experiments/", type=str)
    parser.add_argument('--plot_figures', default=True, type=bool, help="False: do not plot figures in tensorboard"
                                                                        "True: plot figures on tensorboard")


    # BASICS
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)  # 16 for simulation dataset
    parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate of graph neural network")
    parser.add_argument('--seed', default=1234, type=int, help="Random seed")

    # DATASET
    parser.add_argument('--data_dir', default="../data/grasp/", type=str)
    parser.add_argument("-N", default=10, type=int, help="number of datapoints")

    # MODELS
    parser.add_argument('--modality', default=0, type=int, help="See fieldsList in main.py")


    return parser.parse_args()


if __name__ == '__main__':
    args = get_argparse()
    print()
