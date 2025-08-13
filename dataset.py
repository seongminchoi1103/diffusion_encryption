import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf

# =======================
# Configuration
# =======================
SAMPLE_RATE = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
SEGMENT_SECONDS = 5
MAX_LEN = SEGMENT_SECONDS * SAMPLE_RATE

# =======================
# Utilities
# =======================
def load_wav(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    elif len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    return y[np.newaxis, :], sr

def wav_to_mel(wav):
    wav_np = wav.squeeze(0) if isinstance(wav, torch.Tensor) else wav
    mel = librosa.feature.melspectrogram(
        y=wav_np,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db).unsqueeze(0)  # (1, F, T)

def mel_to_wav(mel_tensor):
    mel_db = mel_tensor.squeeze().cpu().numpy()
    mel_power = librosa.db_to_power(mel_db)
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=64
    )
    return torch.tensor(wav).unsqueeze(0)

def save_wav(tensor, path, sample_rate=SAMPLE_RATE):
    y = tensor.squeeze().cpu().numpy()
    sf.write(path, y, sample_rate)

# =======================
# Dataset Class
# =======================
class MelDataset(Dataset):
    def __init__(self, wav_dir):
        self.wav_paths = [
            os.path.join(wav_dir, f) for f in os.listdir(wav_dir) if f.endswith(".wav")
        ]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav, _ = load_wav(self.wav_paths[idx])
        mel = wav_to_mel(wav)
        return mel
