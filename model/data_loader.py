import random
import numpy as np
import os
import sys

import torch
from torchtext import data

import utils

class DataLoader(object):

    def __init__(self, data_dir, params):

        device = torch.device(params.device)

        self.BABYNAME = data.Field(sequential=True, tokenize=utils.tokenizer,
                                   batch_first=True, init_token="<bos>", eos_token="<eos>")
        self.SEX = data.Field(sequential=False, use_vocab=True, postprocessing=utils.onehot)

        self.train_ds, self.val_ds = data.TabularDataset.splits(
            path=data_dir, skip_header=True, train='train/train_dataset.csv',
            validation='val/val_dataset.csv', format='csv',
            fields=[('babyname', self.BABYNAME), ('sex', self.SEX)]
        )

        self.build_vocab()

        self.train_iter, self.val_iter = data.BucketIterator.splits(
            (self.train_ds, self.val_ds), batch_sizes=(params.batch_size, params.batch_size), device=device,
            repeat=False, sort_key=lambda x: len(x.babyname))


    def build_vocab(self):
        self.BABYNAME.build_vocab(self.train_ds, self.val_ds)
        self.SEX.build_vocab(self.train_ds, self.val_ds)
        print("vocab built")



# model_dir = '../experiments/'
# json_path = os.path.join(model_dir, 'params.json')
# params = utils.Params(json_path)
# data_dir = '../data/full_version'
#
# data_loader = DataLoader(data_dir, params)
#
# print(data_loader.BABYNAME.vocab.freqs)
# print(data_loader.SEX.vocab.freqs)
#
# print(list(data_loader.train_ds.attributes))
#
# sample = next(iter(data_loader.train_iter))
# inputs = sample.babyname[:, :-1]
# labels = sample.babyname[:, 1:]
# category = sample.sex.float()
#
#
# print("category: ", category, category.size())
# print("inputs: ", inputs, inputs.size())
# print("labels: ", labels, labels.size())
#
# embed = torch.nn.Embedding(num_embeddings=30, embedding_dim=100)
#
# print(embed(inputs).size())
# print("-=-")
# print(category.type(), embed(inputs).type())
# inputs_combined = torch.cat([category, embed(inputs)[:, 1, :].squeeze(1)], 1)
# print("inputs_combined:", inputs_combined.size())

# torch.cat([category, embed], 1)
# print(torch.cat((category, embed), 1).size())