import importlib.util
import os
import sys
import torch
import torchaudio
from torch.utils.data import Dataset

spec = importlib.util.spec_from_file_location("byol_a", os.path.join(os.path.dirname(__file__), "byol-a/byol_a", "__init__.py"))
byol_a = importlib.util.module_from_spec(spec)
sys.modules["byol_a"] = byol_a
spec.loader.exec_module(byol_a)

from byol_a.dataset import MelSpectrogramLibrosa
from byol_a.models import AudioNTT2020

class Wav2EmbeddingDataset(Dataset):
    """
    Based on WavInLMSOutDataset from byol_a.dataset

    Args:
        cfg: Configuration settings.
        audio_files: List of audio file pathnames (.wav format).
        labels: List of labels corresponding to the audio files.
        tfms: Transforms (augmentations), callable.
        use_librosa: True if using librosa for converting audio to log-mel spectrogram (LMS).
        device: Device to run embedding model on.
        embedding_dim: Dimension of embeddings.
        encoder_weights: Required, path to byol_a model .pth file (must be the right dimension).
        sequence_max_length: Maximum length of embedding sequence.
    """
    def __init__(self, cfg, audio_files, labels, tfms, use_librosa=False, device = torch.device("cuda"), embedding_dim = 2048, encoder_weights = "byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth", sequence_max_length = 30):
        # argment check
        assert (labels is None) or (len(audio_files) == len(labels)), 'The number of audio files and labels has to be the same.'
        super().__init__()

        # initializations
        self.cfg = cfg
        self.files = audio_files
        self.labels = labels
        self.tfms = tfms
        self.unit_length = int(cfg.unit_sec * cfg.sample_rate)
        self.to_melspecgram = MelSpectrogramLibrosa(
            fs=cfg.sample_rate,
            n_fft=cfg.n_fft,
            shift=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
        ) if use_librosa else AT.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2,
        )
        self.model = AudioNTT2020(d = embedding_dim)
        self.model.load_weight(encoder_weights, device)
        self.max_length = sequence_max_length
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])

        assert sr == self.cfg.sample_rate, f"Sample rate of input file ({sr}) does not match expected sample rate ({cfg.sample_rate})"
        assert wav.shape[0] == 1, f"Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels."

        n_chunks = wav.size(1) // self.unit_length

        chunks = []

        for i in range(n_chunks):
            chunks.append(wav[0][i * self.unit_length:(i+1) * self.unit_length])

        lms = []
        for w in chunks:
            if self.tfms is not None:
                lms.append(self.tfms((self.to_melspecgram(w) + torch.finfo().eps).log().unsqueeze(0)))
            else:
                lms.append((self.to_melspecgram(w) + torch.finfo().eps).log().unsqueeze(0))
        
        lms = torch.stack(lms)

        def process_sequence(x):
            # Truncate or pad the sequence to the max length
            if x.size(0) > self.max_length:
                x = x[:self.max_length, :]
            else:
                repeat_factor = (self.max_length + x.size(0) - 1) // x.size(0)
                x = x.repeat(repeat_factor, 1)[:self.max_length, :]

            return x

        embedding = process_sequence(self.model(lms))

        if self.labels is not None:
            return embedding, torch.tensor(self.labels[idx])
        
        return embedding