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
from model_utils import init_weights
from utils import count_parameters, epoch_time


class Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers):
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(in_dim, hid_dim, n_layers)
        self.hid = nn.Linear(hid_dim, hid_dim)

    def forward(self, input_data: torch.Tensor):
        """
        :param input_data: [B, T, C]
        :return:
            output: [T, B, hid dim]
            hidden: [n layers, B, hid dim]
        """

        output, hidden = self.rnn(input_data.permute(1, 0, 2))

        hidden = torch.tanh(self.hid(hidden))

        return output, hidden


class Attention(nn.Module):
    def __init__(self, en_hid_dim, de_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(en_hid_dim + de_hid_dim, de_hid_dim)
        self.v = nn.Linear(de_hid_dim, 1, bias=False)

    def forward(self, en_outputs: torch.Tensor, hidden: torch.Tensor):
        """
        :param en_outputs: [T, B, hid dim]
        :param hidden: [B, hid dim]
        :return:
            attention: [B, T]
        """
        T = en_outputs.size(0)

        hidden = hidden.unsqueeze(1).repeat(1, T, 1)  # [B, T, hid dim]

        en_outputs = en_outputs.permute(1, 0, 2)  # [B, T, hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, en_outputs), dim=2)))  # [B, T, hid dim]

        attention = self.v(energy).squeeze(2)  # [B, T]

        return F.softmax(attention, dim=-1)  # [B, T]


class Decoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers):
        super(Decoder, self).__init__()

        self.rnn = nn.GRU(in_dim + hid_dim, hid_dim, n_layers)
        self.fc_out = nn.Linear(2 * hid_dim + in_dim, in_dim)

        self.attn = Attention(hid_dim, hid_dim)

    def forward(self, input_data: torch.Tensor, hidden: torch.Tensor, en_outputs: torch.Tensor):
        """
        :param input_data: [B, input dim]
        :param hidden: [n layers, B, hid dim]
        :param en_outputs: [T, B, hid]
        :return:
            de_output: [B, hid dim]
            hidden: [n layers, B, hid dim]
        """
        input_data = input_data.unsqueeze(0)  # [1, B, input dim]

        a = self.attn(en_outputs, hidden[-1]).unsqueeze(1)  # [B, 1, T]

        en_outputs = en_outputs.permute(1, 0, 2)  # [B, T, hid]

        weighted = torch.bmm(a, en_outputs).permute(1, 0, 2)  # [1, B, hid]

        rnn_input = torch.cat((input_data, weighted), dim=-1)  # [1, B, input dim + hid]

        de_output, hidden = self.rnn(rnn_input, hidden)
        # de_output = [1, B, hid dim]
        # hidden = [n layers, B, hid dim]

        output_cat = torch.cat((de_output, hidden[-1].unsqueeze(0), input_data), dim=-1)
        # output_cat = [1, B, 2 * hid dim + input dim]

        de_output = self.fc_out(output_cat.squeeze(0))  # [B, input dim]

        return de_output, hidden


class GRU_Seq2Seq(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers):
        super(GRU_Seq2Seq, self).__init__()
        self.encoder = Encoder(in_dim, hid_dim, n_layers)
        self.decoder = Decoder(in_dim, hid_dim, n_layers)

    def forward(self, input_data, **kwargs):
        """
        :param input_data: dict, keys = {"flow_nx" = [B, N, SRC_Len, C], "flow_y" = [B, N, TRG_Len, C]}
        :param device: torch.device()
        :param teacher_ration: float.
        :return:
            prediction: [B, N, TRG_len, C]
            target:     [B, N, TRG_len, C]
        """
        device = kwargs["device"]
        teacher_ration = kwargs["ration"]

        source = input_data["flow_d0_x"].to(device)  # [B, N, T, C]
        target = input_data["flow_y"].to(device)  # [B, N, T, C]

        B, N, T, C = target.size()

        source = source.permute(1, 0, 2, 3)  # [N, B, T, C]
        target = target.permute(1, 0, 2, 3)  # [N, B, T, C]

        decoder_output = torch.zeros(N, B, T, C).to(device)

        encoder_result = [self.encoder(source[node_i]) for node_i in range(N)]
        # tuple : [([T, B, hid dim] , [n layers, B, hid dim]) * N]

        encoder_output = torch.cat([en_output[0].unsqueeze(0) for en_output in encoder_result], dim=0)
        # [N, T, B, hid dim]

        hidden = torch.cat([en_output[1].unsqueeze(0) for en_output in encoder_result], dim=0)
        # [N, n layers, B, hid dim]

        decoder_input = source[:, :, -1]  # [N, B, C]
        for i in range(T):
            decoder_result = [self.decoder(input_data=decoder_input[node_i],
                                           hidden=hidden[node_i],
                                           en_outputs=encoder_output[node_i]) for node_i in range(N)]
            # tuple : [ ([B, hid dim] , [n layers, B, hid dim]) * N]

            decoder_output[:, :, i] = torch.cat([de_result[0].unsqueeze(0) for de_result in decoder_result])
            #  [N, B, C]

            hidden = torch.cat([de_result[1].unsqueeze(0) for de_result in decoder_result])

            decoder_input = target[:, :, i] if random.random() < teacher_ration else decoder_output[:, :, i]
            # [N, B, C]

        return decoder_output.permute(1, 0, 2, 3), target.permute(1, 0, 2, 3)


def train_epoch(model, train_data, criterion, optimizer, device, ration):
    epoch_loss = 0.0

    model.train()
    # for data in train_data:

    optimizer.zero_grad()

    # forward
    prediction, target = model(train_data, device=device, ration=ration)

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
        prediction, target = model(valid_data, device=device, ration=0)

        loss = criterion(prediction, target)

        epoch_loss += loss.item()

    return epoch_loss


def train(N_EPOCHS, train_iterator, valid_iterator, criterion, optimizer, device):
    best_valid_loss = float("inf")
    ration = 1

    for epoch in range(N_EPOCHS):
        if (epoch + 1) % 5 == 0:
            ration *= 0.9

        start_time = time.time()

        train_loss = train_epoch(model, train_iterator, criterion, optimizer, device, ration)
        valid_loss = eval_epoch(model, valid_iterator, criterion, device)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print("Epoch: {:02d}  |  Time: {}m {}s".format(epoch, epoch_mins, epoch_secs))
        print("\tTrain Loss: {:.4f}  |  Val. Loss: {:.4f}".format(train_loss, valid_loss))


if __name__ == '__main__':
    train_source = torch.randn(32, 15, 6, 2)
    train_target = torch.randn(32, 15, 6, 2)

    valid_source = torch.randn(32, 15, 6, 2)
    valid_target = torch.randn(32, 15, 6, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRU_Seq2Seq(in_dim=2, hid_dim=16, n_layers=2).to(device)

    print("Number of parameters: ", count_parameters(model))

    print(model.apply(init_weights))

    optimizer = optim.Adam(model.parameters())

    loss_fu = nn.MSELoss()

    train(N_EPOCHS=200,
          train_iterator={"flow_d0_x": train_source, "flow_y": train_target},
          valid_iterator={"flow_d0_x": valid_source, "flow_y": valid_target},
          criterion=loss_fu, optimizer=optimizer, device=device)
