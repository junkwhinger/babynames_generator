import argparse
import logging
import os
import pickle

import torch
import torch.optim as optim

import utils
import model.charRNN as net
from model.data_loader import DataLoader

from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/full_version', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir "
                                                         "containing weights to reload before training")


def train(model, optimizer, loss_fn, data_iterator, params, num_steps):

    metric_watcher = utils.MetricCalculator()

    model.train()


    for ix, batch in enumerate(data_iterator):
        inputs = batch.babyname[:, :-1].to(params.device)
        targets = batch.babyname[:, 1:].to(params.device)
        category = batch.sex.float().to(params.device)

        hidden = model.init_hidden(inputs.size(0))

        loss = 0.0

        for step in range(inputs.size(1)):
            outputs, hidden = model.forward(category, inputs[:, step], hidden)
            current_loss = loss_fn(outputs, targets[:, step])

            metric_watcher.update(outputs, targets[:, step], current_loss)
            loss += current_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metric_watcher.calculate_metric()
    metrics_string = "loss: {:05.3f}, acc: {:05.3f}".format(
        metric_watcher.average_loss,
        metric_watcher.accuracy
    )
    logging.info("- Train metrics: " + metrics_string)



def train_and_evaluate(model, train_data_iter, val_data_iter, optimizer, loss_fn, params, model_dir, restore_file):

    # if restore_file is given, load the checkpoint
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)


    best_val_acc = 0.0

    for epoch in range(params.num_epochs):

        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch
        num_steps = len(train_data_iter.dataset.examples) // params.batch_size + 1
        train(model, optimizer, loss_fn, train_data_iter, params, num_steps)

        val_metrics = evaluate(model, loss_fn, train_data_iter, params)
        val_acc = val_metrics['accuracy']
        is_best = val_acc > best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch+1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir,
                              epoch=epoch+1)

        if is_best:
            logging.info("-- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)





if __name__ == '__main__':
    # Load the parameters from the json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')

    assert os.path.isfile(json_path), "No json file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.device = torch.device(params.device)

    # set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    param_string = ""
    for k, v in params.__dict__.items():
        param_string += ("    - {}: {}\n".format(k, v))

    logging.info("-- initiating TRAINING..\n{}".format(param_string))

    # Create the input data pipeline
    logging.info("-- loading the dataset..")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    params.vocab_size = len(data_loader.BABYNAME.vocab)

    # specify the train and val dataset sizes
    params.train_size = len(data_loader.train_ds.examples)
    params.val_size = len(data_loader.val_ds.examples)

    train_iter = data_loader.train_iter
    val_iter = data_loader.val_iter

    # load model
    model = net.Net(params).to(params.device)

    # load optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # define loss function
    loss_fn = net.loss_fn

    # Train the model
    logging.info("-- start training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_iter, val_iter, optimizer, loss_fn, params, args.model_dir, args.restore_file)
