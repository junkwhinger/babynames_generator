import logging

import torch
import utils

def evaluate(model, loss_fn, data_iterator, params):

    metric_watcher = utils.MetricCalculator()

    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
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

    metric_watcher.calculate_metric()

    # compute mean of all metrics in summary
    # metrics_mean = {metric: np.nanmean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    metric_watcher.calculate_metric()
    metrics_string = "loss: {:05.3f}, acc: {:05.3f}".format(
        metric_watcher.average_loss,
        metric_watcher.accuracy
    )
    logging.info("- Eval metrics: " + metrics_string)

    return metric_watcher.export()
