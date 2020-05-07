
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


# https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py
class WarmupLearninngRate:
    def __init__(self, optimizer, warmup_steps = 16000, init_lr=0.01, end_lr=0.2):

        # linearly warmup for the first args.warmup_updates
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.warmup_steps = warmup_steps
        self.lr_step = (end_lr - init_lr) / warmup_steps

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = end_lr * warmup_steps**0.5

        # initial learning rate
        self.lr = init_lr
        self.optimizer = optimizer

        self.optimizer.param_groups[0]['lr'] = self.lr
        self.num_updates = 0

    def step(self):
        """Update the learning rate after each update."""
        if self.num_updates < self.warmup_steps:
            self.lr = self.init_lr + self.num_updates*self.lr_step
        else:
            self.lr = self.decay_factor * self.num_updates**-0.5

        self.num_updates += 1
        self.optimizer.param_groups[0]['lr'] = self.lr

        return self.lr

#  # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping: 
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, verbose=False, checkpoint_file="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

        self.checkpoint_dest = os.path.join("model_checkpoints", checkpoint_file)

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print("Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), self.checkpoint_dest)
        self.val_loss_min = val_loss

def plot_and_save(train_metrics, val_matrics, title, label, save_file):
    plt.plot(train_metrics)
    plt.plot(val_matrics)
    plt.title(title)
    plt.ylabel(label)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')

    plt.savefig(save_file, dpi=600)
    plt.show()
    plt.clf()