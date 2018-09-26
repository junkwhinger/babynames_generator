import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        self.params = params

        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.lstm = nn.LSTM(params.embedding_dim + params.nb_categories,
                            params.lstm_hidden_dim, num_layers=params.lstm_nb_layers,
                            batch_first=True)
        self.fc_inter = nn.Linear(params.lstm_hidden_dim, params.fc_inter)
        self.dropout = nn.Dropout(params.dropout)
        self.decoder = nn.Linear(params.fc_inter, params.vocab_size)

    def init_hidden(self, batch_size):
        return (torch.zeros(
                    self.params.lstm_nb_layers,
                    batch_size,
                    self.params.lstm_hidden_dim
                ).float().to(self.params.device),
                torch.zeros(
                    self.params.lstm_nb_layers,
                    batch_size,
                    self.params.lstm_hidden_dim
                ).float().to(self.params.device))

    def forward(self, category, inputs, hidden, isDebug=False):
        if isDebug: print("category:", category.size())
        if isDebug: print("inputs:", inputs.size())

        embed = self.embedding(inputs)
        if isDebug: print("embed:", embed.size())

        inputs_combined = torch.cat([category, embed], dim=1)
        if isDebug: print("inputs_combined:", inputs_combined.size())

        lstm_out, hidden = self.lstm(inputs_combined.unsqueeze(1), hidden)
        if isDebug: print("lstm_out:", lstm_out.size())
        if isDebug: print("last_hidden_state:", hidden[0].size())
        if isDebug: print("last_cell_state:", hidden[1].size())

        fc_inter_out = self.fc_inter(lstm_out.squeeze(1))
        if isDebug: print("fc_inter_out:", fc_inter_out.size())

        dropout_out = self.dropout(fc_inter_out)

        decoder_out = self.decoder(dropout_out)
        if isDebug: print("decoder_out:", decoder_out.size())

        return decoder_out, hidden




def loss_fn(outputs, labels):
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(outputs, labels)

    return loss