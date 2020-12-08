import torch
import numpy as np
from torch import nn, optim
import copy
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_lstm(model, train_dataset, val_dataset, n_epochs, optimizer, criterion):
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(1, n_epochs + 1):
        model = model.train()

        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history

def train_attention(model, train_loader, val_loader, n_epochs, optimizer, criterion):
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(1, n_epochs + 1):
        model = model.train()

        train_losses = []
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            prediction = model(data)

            loss = criterion(prediction, data) 

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device)
                prediction = model(data)

                loss = criterion(prediction, data) 
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses) / 100
        val_loss = np.mean(val_losses) / 100

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history

def predict_lstm(model, dataset, criterion):
    predictions, losses = [], []
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())

    return predictions, losses

def predict_attention(model, dataloader, criterion):
    predictions, losses = [], []
    with torch.no_grad():
        model = model.eval()
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            prediction = model(data)

            loss = criterion(prediction, data)

            predictions.append(prediction.cpu().numpy().flatten())
            losses.append(loss.item())

    return predictions, losses

