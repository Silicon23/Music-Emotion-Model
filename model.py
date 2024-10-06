import importlib.util
import os
import sys
import torch
import torch.nn as nn

spec = importlib.util.spec_from_file_location("byol_a", os.path.join(os.path.dirname(__file__), "byol-a/byol_a", "__init__.py"))
byol_a = importlib.util.module_from_spec(spec)
sys.modules["byol_a"] = byol_a
spec.loader.exec_module(byol_a)

from byol_a.models import AudioNTT2020

class MusicEmotionRecognition(nn.Module):
    def __init__(self, device = torch.device("cuda"), embedding_dim = 2048, encoder_weights = "byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth"):
        super(MusicEmotionRecognition, self).__init__()
        self.encoder = AudioNTT2020(d=embedding_dim)
        self.encoder.load_weight(encoder_weights, device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.emotion_layer = nn.Sequential(
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
        return self.emotion_layer(self.encoder(x))