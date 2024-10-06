import importlib.util
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from model import MusicEmotionRecognition

spec = importlib.util.spec_from_file_location("byol_a", os.path.join(os.path.dirname(__file__), "byol-a/byol_a", "__init__.py"))
byol_a = importlib.util.module_from_spec(spec)
sys.modules["byol_a"] = byol_a
spec.loader.exec_module(byol_a)

from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.dataset import WaveInLMSOutDataset

device = torch.device('mps')
dtype = torch.float32
cfg = load_yaml_config("byol-a/config.yaml")

files = []

labels = pd.read_csv(os.path.join(os.path.dirname(__file__), "PMEmo2019", "annotations", "static_annotations.csv"))

files = [os.path.join(os.path.dirname(__file__), "PMEmo2019", "wav", f"{id}.wav") for id in labels['musicId'].values]

ds = WaveInLMSOutDataset(cfg, files, labels=labels[['Arousal(mean)', 'Valence(mean)']].to_numpy(), tfms=PrecomputedNorm([2.430965, 2.7521515]))
train_size = round(0.9 * len(ds))
test_size = len(ds) - train_size
train_ds, test_ds = random_split(ds, [train_size, test_size])
train_dl = DataLoader(train_ds, batch_size=16, pin_memory=False, shuffle=True, drop_last=False)
test_dl = DataLoader(test_ds, batch_size=len(test_ds), pin_memory=False, shuffle=False, drop_last=False)

model = MusicEmotionRecognition(device, embedding_dim=cfg.feature_d, encoder_weights="byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth").to(device)
model.train()

params = model.emotion_layer.parameters()
optimizer = torch.optim.Adam(params, lr=5e-5)
criterion = torch.nn.MSELoss()

def train(model, optimizer, criterion, dataloader):
    model.train()
    losses = []
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device, dtype=dtype)

        optimizer.zero_grad()
        loss = criterion(model(data), label)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().item())
    return np.mean(losses)

def eval(model, criterion, dataloader):
    model.eval()
    losses = []
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device, dtype=dtype)
        with torch.no_grad():
            output = model(data)
        losses.append(criterion(output, label).detach().cpu().item())
    return np.mean(losses)

n_epochs = 15

training_loss, validation_loss = [], []
save_path = os.path.join(os.path.dirname(__file__), "MusicEmotionRecognitionModel.pth")
min_loss = np.inf

for epoch in range(n_epochs):
    training_loss.append(train(model, optimizer, criterion, train_dl))
    print(f'Epoch {epoch}, training loss: {training_loss[-1]:.2E}', end='')
    val_loss = eval(model, criterion, test_dl)
    validation_loss.append(val_loss)
    print(f', validation loss: {validation_loss[-1]:.2E}')
    if validation_loss[-1] < min_loss:
        min_loss = validation_loss[-1]
        print(f"min_loss = {min_loss}")
        torch.save(model.state_dict(), save_path)