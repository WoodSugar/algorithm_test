# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/13

@Author : Shen Fang
"""
import time
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_utils import init_weights, count_parameters, epoch_time
from model_utils import CausalConv1d


# Encoder 没有问题
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, kernel_size, seq_len):
        super(Encoder, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"

        self.pos_embedding = nn.Embedding(seq_len, input_dim)

        self.in2hid = nn.Linear(input_dim, hid_dim)
        self.hid2in = nn.Linear(hid_dim, input_dim)

        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size,
                                              padding=(kernel_size - 1) // 2)
                                    for _ in range(n_layers)])

    def forward(self, input_data, device):
        # input data = [batch size, seq len, input dim]
        b_size = input_data.size(0)
        seq_len = input_data.size(1)

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(b_size, 1).to(device)
        # [batch size, seq len]

        pos_embedding = self.pos_embedding(pos)  # [batch size, seq len, input_dim]

        input_embedding = input_data + pos_embedding   # [batch size, seq len, input_dim]

        conv_input = self.in2hid(input_embedding).permute(0, 2, 1)  # [batch size, hid_dim, seq len]

        for conv_layer in self.convs:
            conved = conv_layer(conv_input)  # [batch size, 2 * hid_dim, seq len]
            conved = F.glu(conved, dim=1)  # [batch size, hid_dim, seq len]

            conved = conved + conv_input

            conv_input = conved

        conved = self.hid2in(conved.permute(0, 2, 1))  # [batch size, seq len, input dim]

        combined = (conved + input_embedding) * 0.5 # [batch size, seq len, input dim]

        return conved, combined


# Decoder部分，采用causal convolution

class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, kernel_size, seq_len):
        super(Decoder, self).__init__()

        self.pos_embedding = nn.Embedding(seq_len, input_dim)

        self.in2hid = nn.Linear(input_dim, hid_dim)
        self.hid2in = nn.Linear(hid_dim, input_dim)

        self.attn_in2hid = nn.Linear(input_dim, hid_dim)
        self.attn_hid2in = nn.Linear(hid_dim, input_dim)

        self.fc_out = nn.Linear(input_dim, input_dim)

        self.conv = nn.ModuleList([CausalConv1d(hid_dim, 2 * hid_dim, kernel_size, 1) for _ in range(n_layers)])

        self.hid_dim = hid_dim
        self.kernel_size = kernel_size

    def calculate_attn(self, decoder_in_embedded, decoder_in_conved, encoder_out_conved, encoder_out_combined):
        """
        :param decoder_in_embedded:  [batch size, tar len, input dim]
        :param decoder_in_conved:    [batch size, hid dim, tar len]
        :param encoder_out_conved:   [batch size, src len, input dim]
        :param encoder_out_combined: [batch size, src len, input dim]
        :return:
        """

        conved_emb = self.attn_hid2in(decoder_in_conved.permute(0, 2, 1))  # [batch size, tar len, input dim]
        combined = (conved_emb + decoder_in_embedded) * 0.5  # [batch size, tar len, input_dim]

        energy = torch.matmul(combined, encoder_out_conved.permute(0, 2, 1))  # [batch size, tar len, src len]

        attention = F.softmax(energy, dim=2)  # [batch size, tar len, src len]

        attended_encoding = torch.matmul(attention, encoder_out_combined)  # [batch size, tar len, input dim]

        attended_encoding = self.attn_in2hid(attended_encoding)  # [batch size, tar len, hid dim]

        attended_combined = (decoder_in_conved + attended_encoding.permute(0, 2, 1)) * 0.5

        return attention, attended_combined

    def forward(self, input_data, en_conved, en_combined, device):
        # input_data = [batch size, seq len, input_dim]
        # en_conved = en_combined = [batch size, seq len, emb dim]

        b_size = input_data.size(0)
        seq_len = input_data.size(1)

        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(b_size, 1).to(device)
        # [batch size, seq len]

        pos_embedding = self.pos_embedding(pos)  # [batch size, seq len, input_dim]

        input_embedding = input_data + pos_embedding

        conv_input = self.in2hid(input_embedding).permute(0, 2, 1)  # conv_input = [batch size, hid dim, seq len]

        for conv in self.conv:
            conved = conv(conv_input)  # [batch size, hid dim, seq len]

            conved = F.glu(conved, dim=1)  # [batch size, hid dim, seq len]

            attention, conved = self.calculate_attn(input_embedding, conved, en_conved, en_combined)

            conved = (conved + conv_input) * 0.5

            conv_input = conved

        conved = self.hid2in(conved.permute(0, 2, 1))  # conved = [batch size, seq len, input dim]

        output = self.fc_out(conved)  # [batch size, seq len, input dim]

        return output, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, device):
        """
        :param source: [batch size, seq len, input dim]
        :param target: [batch size, seq len, input dim]
        :param device:
        :return:
        """
        trg_len = target.size(1)
        encoder_conved, encoder_combined = self.encoder(source, device)

        if self.training:
            output, attention = self.decoder(torch.cat((source, target[:, :-1]), dim=1)[:, -trg_len:],
                                             encoder_conved, encoder_combined, device)

            return output

        else:
            decoder_input = torch.zeros_like(target)

            decoder_input[:, 0] = source[:, -1]  # [batch size, seq len, input dim]

            for i in range(trg_len):
                output, attention = self.decoder(decoder_input, encoder_conved, encoder_combined, device)
                if i != trg_len - 1:
                    decoder_input[:, i+1] = output[:, i]

            return output



def train_epoch(model, train_data, criterion, optimizer, device):
    epoch_loss = 0.0

    model.train()
    # for data in train_data:
    inputs, target = train_data

    inputs = inputs.to(device)
    target = target.to(device)

    optimizer.zero_grad()

    # forward
    prediction = model(inputs, target, device)

    # backward
    loss = criterion(prediction, target)
    loss.backward()

    # update parameters
    optimizer.step()

    epoch_loss += loss.item()

    return epoch_loss


def eval_epoch(model, valid_data, criterion, device):
    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        # for data in valid_data:
        inputs, target = valid_data
        inputs = inputs.to(device)
        target = target.to(device)

        prediction = model(inputs, target, device)

        loss = criterion(prediction, target)

        epoch_loss += loss.item()

    return epoch_loss


def train(N_EPOCHS, train_iterator, valid_iterator, criterion, optimizer, device):
    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, criterion, optimizer, device)
        valid_loss = eval_epoch(model, valid_iterator, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print("Epoch: {:02d}  |  Time: {}m {}s".format(epoch, epoch_mins, epoch_secs))
        print("\tTrain Loss: {:.4f}  |  Val. Loss: {:.4f}".format(train_loss, valid_loss))


if __name__ == '__main__':
    train_source = torch.randn(32, 6, 2)
    train_target = torch.randn(32, 6, 2)

    valid_source = torch.randn(32, 6, 2)
    valid_target = torch.randn(32, 6, 2)

    encoder = Encoder(input_dim=2, hid_dim=16, n_layers=2, kernel_size=3, seq_len=6)
    decoder = Decoder(input_dim=2, hid_dim=16, n_layers=2, kernel_size=3, seq_len=6)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq(encoder, decoder).to(device)

    print(count_parameters(model))

    print(model.apply(init_weights))

    optimizer = optim.Adam(model.parameters())

    loss_fu = nn.MSELoss()

    train(N_EPOCHS=200,
          train_iterator=(train_source, train_target),
          valid_iterator=(valid_source, valid_target),
          criterion = loss_fu, optimizer=optimizer, device=device)


