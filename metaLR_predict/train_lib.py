# -*- coding: utf-8 -*-
"""
@Time   : 2020/8/15

@Author : Shen Fang
"""
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from utils import epoch_time, Evaluate
from data_loader import LoadData


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.MSELoss, optimizer: torch.optim.Adam,
                device: torch.device, ration):
    epoch_loss = 0.0

    model.train()

    for data in train_loader:
        optimizer.zero_grad()

        prediction, target = model(data, device=device, ration=ration)

        loss = criterion(prediction, target)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader.dataset)


def eval_epoch(model, valid_loader, criterion, device):
    epoch_loss = 0.0

    model.eval()

    with torch.no_grad():
        for data in valid_loader:

            prediction, target = model(data, device=device, ration=0)

            loss = criterion(prediction, target)

            epoch_loss += loss.item()

    return epoch_loss / len(valid_loader.dataset)


def train(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, criterion: nn.MSELoss(),
          optimizer: torch.optim.Adam, scheduler: MultiStepLR, option, device: torch.device):
    train_file = None
    valid_file = None

    if option.log:
        train_file = option.log + "_train.csv"
        valid_file = option.log + "_valid.csv"

        print("[ INFO ] Training information will be written.")

    best_valid_loss = float("inf")
    ration = 1

    for each_epoch in range(option.epoch):
        if (each_epoch + 1) % 5 == 0:
            ration *= 0.9

        print("[ Epoch {:d}]".format(each_epoch))

        # Train One Epoch
        start_time = time.time()

        train_loss = train_epoch(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer,
                                 device=device, ration=ration)

        train_minute, train_second = epoch_time(start_time, time.time())
        print("  - (Training)   loss: {:2.4f}, time: {:2d} min {:2d} sec ".format(train_loss, train_minute, train_second))

        scheduler.step()

        # Validate One Epoch
        start_time = time.time()

        valid_loss = eval_epoch(model=model, valid_loader=valid_loader, criterion=criterion, device=device)

        valid_minute, valid_second = epoch_time(start_time, time.time())

        print("  - (Validation) loss: {:2.4f}, time: {:2d} min {:2d} sec ".format(valid_loss, valid_minute, valid_second))

        # Save Model
        model_state_dict = model.state_dict()

        checkpoint = {"model": model_state_dict,
                      "setting": option,
                      "epoch": each_epoch}

        if option.save_model:
            if option.save_mode == "best":
                model_name = option.log + ".pkl"
                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss

                    torch.save(checkpoint, model_name)
                    print("  - [ INFO ] The checkpoint file is updated.")

            elif option.save_mode == "all":
                model_name = option.log + "_epoch_{:2d}.pkl".format(each_epoch)
                torch.save(checkpoint, model_name)

        if train_file and valid_file:
            with open(train_file, "a") as train_obj, open(valid_file, "a") as valid_obj:
                train_obj.write("{:2d}, {:2.4f}, {:2d} min {:2d} sec \n".format(each_epoch, train_loss, train_minute, train_second))
                valid_obj.write("{:2d}, {:2.4f}, {:2d} min {:2d} sec \n".format(each_epoch, valid_loss, valid_minute, valid_second))


def test(model: nn.Module, test_loader: DataLoader, criterion: nn.MSELoss(), option, device: torch.device, eval_dim: tuple=(0, 1, 3)):
    test_loss = 0.0

    data_to_save = {"predict": np.zeros(shape=[1, option.num_nodes, option.trg_len, option.input_dim]),  # [B, N, TRG_len, C]
                    "target": np.zeros(shape=[1, option.num_nodes, option.trg_len, option.input_dim])}   # [B, N, TRG_len, C]

    model.eval()

    with torch.no_grad():
        for data in test_loader:
            prediction, target = model(data, device=device, ration=0)

            loss = criterion(prediction, target)

            test_loss += loss.item()

            recovered = recover_data(prediction, target, test_loader)

            data_to_save["predict"] = np.concatenate([data_to_save["predict"], recovered["predict"]], axis=0)
            data_to_save["target"] = np.concatenate([data_to_save["target"], recovered["target"]], axis=0)

    data_to_save["predict"] = np.delete(data_to_save["predict"], 0, axis=0)
    data_to_save["target"] = np.delete(data_to_save["target"], 0, axis=0)

    mae, mape, rmse = compute_performance(data_to_save["predict"], data_to_save["target"], dim=eval_dim)

    if option.log:
        test_file = option.log + "_test.csv"
        with open(test_file, "a") as test_obj:
            test_obj.write("MAE:  ")
            for item in mae:
                test_obj.write("  {:2.4f}".format(item))
            test_obj.write("\n")

            test_obj.write("MAPE: ")
            for item in mape:
                test_obj.write("  {:2.4f}".format(item))
            test_obj.write("\n")

            test_obj.write("RMSE: ")
            for item in rmse:
                test_obj.write("  {:2.4f}".format(item))
            test_obj.write("\n")

        result_file = option.log + "_result.h5"
        file_obj = h5py.File(result_file, "w")

        for i in range(option.trg_len):
            file_obj["predict_time_{:d}".format(i)] = data_to_save["predict"][:, :, i].transpose([1, 0, 2])  # [N, B(T), C]
            file_obj["target_time_{:d}".format(i)] = data_to_save["target"][:, :, i].transpose([1, 0, 2])  # [N, B(T), C]

    return np.mean(mae), np.mean(mape), np.mean(rmse), test_loss / (len(test_loader.dataset))


def recover_data(prediction, target, dataset):
    try:
        dataset = dataset.dataset
    except:
        dataset = dataset

    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1],
                                       prediction.to(torch.device("cpu")).numpy())  # [B, N, TRG_len, D]

    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1],
                                   target.to(torch.device("cpu")).numpy())  # [B, N, TRG_len, D]

    return {"predict": prediction, "target": target}


def compute_performance(prediction, target, dim):
    eval = Evaluate(dim)
    mae, mape, rmse = eval.total(target, prediction, ep=5)

    return mae, mape, rmse
