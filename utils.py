import json
import logging
import os
import shutil
import torch

import torch
from sklearn.metrics import accuracy_score

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    '''
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # change the value of learning_rate in params
    '''
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__



def tokenizer(text):
    return list(text.lower())



def onehot(alist, vocab):
    """
    onehot encoder for torchtext field postprocessing
    :param alist: list of category indices (0 for unk)
    :param vocab: category vocabulary object
    :return _onehot: one hot representation of alist
    """
    # alist = [2, 1, 1]
    # _tensor = [[1, 0, 0]]
    _tensor = torch.tensor(alist).data.sub_(1).unsqueeze(1)

    # _onehot = [[0, 0], [0, 0], [0, 0]]
    _onehot = torch.zeros((len(alist), len(vocab) - 1), dtype=torch.float)

    # _onehot = [[0, 1], [1, 0], [1, 0]]
    _onehot.scatter_(1, _tensor, 1)
    return _onehot


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`
    In general, it is useful to have a logger so that every output to the terminal is saved in a permanent file.
    Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint, epoch):
    """
    Saves model and training parameters at checkpoint + 'e01.pth.tar'.
    If is_best==True, also saves checkpoint + 'best.pth.tar'
    :param state:
    :param is_best:
    :param checkpoint:
    :return:
    """
    filepath = os.path.join(checkpoint, 'e{:02d}.pth.tar'.format(epoch))
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists!")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of optimizer
    assuming it is present in checkpoint.
    :param checkpoint:
    :param model:
    :param optimizer:
    :return:
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class MetricCalculator():

    def __init__(self):
        self.accuracy = 0
        self.loss_accumulated = 0
        self.average_loss = 0
        self.updated_cnt = 0

        self.predicted_labels_holder = []
        self.actual_labels_holder = []

    def update(self, outputs, labels, loss):
        self.updated_cnt += 1

        predicted_labels = outputs.max(1)[1]
        self.predicted_labels_holder.append(predicted_labels)
        self.actual_labels_holder.append(labels)
        self.loss_accumulated += loss


    def calculate_metric(self):

        predicted_labels = torch.cat(self.predicted_labels_holder).cpu().numpy()
        actual_labels = torch.cat(self.actual_labels_holder).cpu().numpy()

        self.accuracy = accuracy_score(actual_labels, predicted_labels)
        self.average_loss = self.loss_accumulated / self.updated_cnt


    def reset(self):
        self.accuracy = 0
        self.loss_accumulated = 0
        self.average_loss = 0
        self.updated_cnt = 0

    def export(self):
        return {
            'loss': self.average_loss,
            'accuracy': self.accuracy,
        }


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    :param d:
    :param json_path:
    :return:
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)