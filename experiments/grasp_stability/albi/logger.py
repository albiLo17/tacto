from torch.utils.tensorboard import SummaryWriter
import os
import utils

import matplotlib.pyplot as plt
import torch
import numpy as np

class tensorboard_logger():
    def __init__(self, args):

        self.out_dir = args.logdir
        self.plot_figures = args.plot_figures
        self.best_test_loss = None

        # Get RUN name based on the arguments
        self.run_name = utils.get_run_name(args)

        self.check_point_dir = args.logdir + 'checkpoint/'


        self.figures_dir = os.path.join(self.check_point_dir + self.run_name + '/', 'figures/')
        self.loss_dir = os.path.join(self.check_point_dir + self.run_name + '/', 'losses/')
        self.models_dir = os.path.join(self.check_point_dir + self.run_name + '/', self.run_name + '.pt')

        logdir = args.logdir + 'logs/' + self.run_name

        self.writer = SummaryWriter(log_dir=logdir)

        self.check_folders()

    def check_folders(self):
        if not os.path.exists(self.check_point_dir):
            utils.make_dir(self.check_point_dir)
        if self.plot_figures:
            if not os.path.exists(self.figures_dir):
                utils.make_dir(self.figures_dir)
        if not os.path.exists(self.loss_dir):
            utils.make_dir(self.loss_dir)

    def save_losses(self, epoch, training_losses, test_losses, accuracies=None):
        # Save losses
        if epoch % 10 == 9:
            np.save(self.loss_dir + "train_losses.npy", np.array(training_losses))
            np.save(self.loss_dir + "test_losses.npy", np.array(test_losses))
            if accuracies is not None:
                np.save(self.loss_dir + "accuracies.npy", np.array(accuracies))

    def update_best_loss(self, test_loss, dynamics_net):
        # Save best model
        if self.best_test_loss == None or test_loss < self.best_test_loss:
            torch.save(dynamics_net, self.models_dir)
            self.best_test_loss = test_loss

    def add_scalar(self, loss, epoch, loss_name):
        self.writer.add_scalar(f"{loss_name}", loss, epoch)

    # def plot_images(self, global_step, init_graph, pred_graph, real_graph, adj, type='train'):
    #     if global_step % 1000 == 999 and self.plot_figures:     # TODO: modify this condition for test images
    #         fig = utils.plot_deform(init_graph, adj, pred_graph,
    #                                 real_graph, adj, fix_lim=False, save_path=f'{self.figures_dir}{type}_fig_{global_step}.png')
    #         self.writer.add_figure("Train image", fig, global_step)

    def show_reconstructed_images(self, ground_truth, prediction, epoch, tensoboard_name):
        fig, axs = plt.subplots(2)

        axs[0].imshow(ground_truth.transpose(0,1).transpose(1,2))
        axs[0].set_title("Ground Truth")

        axs[1].imshow(prediction.transpose(0,1).transpose(1,2))
        axs[1].set_title("prediction")

        self.writer.add_figure(tensoboard_name, fig, epoch)
