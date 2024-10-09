"""
USAGE:
python -u inference.py /path/to/wav/file /path/to/encoder/weights /path/to/model/weights
"""
from model import MusicEmotionRecognition
import importlib.util
import os
import sys
import torch
import torchaudio
import argparse
import yaml
import numpy as np

spec = importlib.util.spec_from_file_location("byol_a", os.path.join(os.path.dirname(__file__), "byol-a/byol_a", "__init__.py"))
byol_a = importlib.util.module_from_spec(spec)
sys.modules["byol_a"] = byol_a
spec.loader.exec_module(byol_a)

from byol_a.dataset import MelSpectrogramLibrosa
from byol_a.augmentations import PrecomputedNorm
from byol_a.common import load_yaml_config
from byol_a.models import AudioNTT2020

cfg = load_yaml_config('byol-a/config.yaml')

parser = argparse.ArgumentParser(description='Music Emotion Recognition Inference')
parser.add_argument('input', type=str, help='Path to the input file')
parser.add_argument('encoder_weights', type=str, nargs='?', default='byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth', help='Path to the encoder weights file')
parser.add_argument('model_weights', type=str, nargs='?', default='MusicEmotionRecognitionModel.pth', help='Path to the model weights file')
args = parser.parse_args()

device = torch.device('mps')
encoder = AudioNTT2020(d = cfg.feature_d)
encoder.load_weight(args.encoder_weights, device)
model = MusicEmotionRecognition(embedding_dim = cfg.feature_d).to(device)
model.load_state_dict(torch.load(args.model_weights, map_location=device))
model.eval()


wav, sr = torchaudio.load(args.input)

assert sr == cfg.sample_rate, f"Sample rate of input file ({sr}) does not match expected sample rate ({cfg.sample_rate})"
assert wav.shape[0] == 1, f"Convert .wav files to single channel audio, {args.input} has {wav.shape[0]} channels."

max_length = 30

unit_length = int(cfg.unit_sec * sr)
n_chunks = wav.size(1) // unit_length

chunks = []

for i in range(n_chunks):
    chunks.append(wav[0][i * unit_length:(i+1) * unit_length])

to_melspecgram = MelSpectrogramLibrosa(
            fs=cfg.sample_rate,
            n_fft=cfg.n_fft,
            shift=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
)
tfms = PrecomputedNorm([-2.1819685, 3.0303779])

lms = []
for w in chunks:
    lms.append(tfms((to_melspecgram(w) + torch.finfo().eps).log().unsqueeze(0)))
lms = torch.stack(lms)

def process_sequence(x):
    # Truncate or pad the sequence to the max length
    if x.size(0) > max_length:
        x = x[:max_length, :]
    elif x.size(0) < max_length:
        repeat_factor = (max_length + x.size(0) - 1) // x.size(0)
        x = x.repeat(repeat_factor, 1)[:max_length, :]

    return x

sequence = encoder(lms)
length = sequence.shape[0]

embedding = []

for i in range(max(length-1, max_length//2) // (max_length//2)):
    start = i*(max_length//2)
    if length - i*(max_length//2) < max_length:
        embedding.append(process_sequence(sequence[start:length, :]))
    else:
        embedding.append(sequence[start:start+max_length, :])
embedding = torch.stack(embedding)

with torch.no_grad():
    arousal, valence = np.mean(model(embedding.to(device)).cpu().numpy(), axis=0)
    print(f"Arousal {arousal}, Valence {valence}")