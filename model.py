import importlib.util
import os
import sys
import torch
import torch.nn as nn

class MusicEmotionRecognition(nn.Module):
    def __init__(self, embedding_dim = 2048):
        super(MusicEmotionRecognition, self).__init__()
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, activation='relu', batch_first=True),
            num_layers=2
        )
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.transformer_layer(x)
        x = x.mean(dim=-2)  # Aggregate the sequence
        return self.output_layer(x)