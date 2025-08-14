import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math


class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.source_dir = os.path.join(data_dir, "source")
        self.target_dir = os.path.join(data_dir, "target")

        self.target_files = {}
        for f in os.listdir(self.target_dir):
            if f.endswith(".wav"):
                key = f.replace(".wav", "")
                self.target_files[key] = os.path.join(self.target_dir, f)

        self.source_groups = {}
        for f in os.listdir(self.source_dir):
            if f.endswith(".wav"):
                parts = f.split("_")
                if len(parts) > 3:
                    key = "_".join(parts[:-3])
                    if key in self.target_files:
                        self.source_groups.setdefault(key, []).append(f)

        self.keys = [k for k in self.source_groups if len(self.source_groups[k]) > 0]
        if len(self.keys) == 0:
            raise ValueError("No matching source-target groups found.")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        source_files = self.source_groups[key]
        source_waveforms = []
        for f in source_files:
            path = os.path.join(self.source_dir, f)
            wav, _ = torchaudio.load(path)  # (1, T)
            source_waveforms.append(wav.squeeze(0))  # (T,)
        source_waveforms = torch.stack(source_waveforms, dim=0)
        target_path = self.target_files[key]
        target_waveform, _ = torchaudio.load(target_path)
        target_waveform = target_waveform.squeeze(0)
        return source_waveforms, target_waveform, source_files, os.path.basename(target_path)


def collate_fn(batch):
    max_len = 0
    max_num_sources = 0
    for sources, target, _, _ in batch:
        max_len = max(max_len, sources.shape[1], target.shape[0])
        max_num_sources = max(max_num_sources, sources.shape[0])

    padded_sources_batch = []
    padded_targets_batch = []
    source_names_batch = []
    target_names_batch = []

    for sources, target, source_names, target_name in batch:
        num_sources, length = sources.shape
        padded_sources = torch.zeros((max_num_sources, max_len))
        padded_sources[:num_sources, :length] = sources
        padded_target = torch.zeros(max_len)
        padded_target[:target.shape[0]] = target
        padded_sources_batch.append(padded_sources)
        padded_targets_batch.append(padded_target)
        source_names_batch.append(source_names)
        target_names_batch.append(target_name)

    padded_sources_batch = torch.stack(padded_sources_batch)  # (batch, max_num_sources, max_len)
    padded_targets_batch = torch.stack(padded_targets_batch)  # (batch, max_len)

    return padded_sources_batch, padded_targets_batch, source_names_batch, target_names_batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerAudioModel(nn.Module):
    def __init__(self, n_fft=512, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1

        # Input linear to d_model
        self.input_fc = nn.Linear(self.freq_bins, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output linear to freq_bins magnitude
        self.output_fc = nn.Linear(d_model, self.freq_bins)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, freq_bins, time_frames)
        x = x.permute(0, 2, 1)  # (batch, time_frames, freq_bins)
        x = self.input_fc(x)  # (batch, time_frames, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch, time_frames, d_model)
        x = self.dropout(x)
        x = self.output_fc(x)  # (batch, time_frames, freq_bins)
        x = x.permute(0, 2, 1)  # (batch, freq_bins, time_frames)
        x = F.relu(x)  # magnitude는 음수가 안되게 ReLU
        return x


def waveform_to_magphase(waveform, n_fft=512, hop_length=256):
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    mag = stft.abs()
    phase = stft.angle()
    return mag, phase


def magphase_to_waveform(mag, phase, n_fft=512, hop_length=256, length=None):
    complex_spec = torch.polar(mag, phase)
    waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, length=length)
    return waveform


def train(model, dataloader, epochs, device, save_path, n_fft=512, hop_length=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for sources, target, _, _ in tqdm(dataloader):
            batch_size, num_sources, length = sources.shape
            sources = sources.to(device)
            target = target.to(device)

            all_mags = []
            max_frames = 0
            for i in range(num_sources):
                mag, _ = waveform_to_magphase(sources[:, i, :], n_fft=n_fft, hop_length=hop_length)
                max_frames = max(max_frames, mag.shape[-1])
                all_mags.append(mag)

            target_mag, target_phase = waveform_to_magphase(target, n_fft=n_fft, hop_length=hop_length)
            max_frames = max(max_frames, target_mag.shape[-1])

            padded_mags = []
            for mag in all_mags:
                pad_len = max_frames - mag.shape[-1]
                if pad_len > 0:
                    mag = F.pad(mag, (0, pad_len))
                padded_mags.append(mag)
            padded_mags = torch.stack(padded_mags, dim=1)  # (batch, num_sources, freq_bins, time_frames)

            pad_len = max_frames - target_mag.shape[-1]
            if pad_len > 0:
                target_mag = F.pad(target_mag, (0, pad_len))

            outputs = []
            for i in range(num_sources):
                out_i = model(padded_mags[:, i, :, :])  # (batch, freq_bins, time_frames)
                outputs.append(out_i)

            sum_outputs = torch.stack(outputs, dim=0).sum(dim=0)

            loss = criterion(sum_outputs, target_mag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.6f}")

    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, "model.ckpt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def inference(model, dataloader, device, save_dir, n_fft=512, hop_length=256):
    model.to(device)
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for sources, target, source_names_list, target_names in tqdm(dataloader):
            batch_size, num_sources, length = sources.shape
            sources = sources.to(device)

            all_mags = []
            max_frames = 0
            for i in range(num_sources):
                mag, _ = waveform_to_magphase(sources[:, i, :], n_fft=n_fft, hop_length=hop_length)
                max_frames = max(max_frames, mag.shape[-1])
                all_mags.append(mag)

            padded_mags = []
            for mag in all_mags:
                pad_len = max_frames - mag.shape[-1]
                if pad_len > 0:
                    mag = F.pad(mag, (0, pad_len))
                padded_mags.append(mag)
            padded_mags = torch.stack(padded_mags, dim=1)

            outputs = []
            for i in range(num_sources):
                out_i = model(padded_mags[:, i, :, :]).cpu()
                outputs.append(out_i)

            sum_outputs = torch.stack(outputs, dim=0).sum(dim=0)  # (batch, freq_bins, time_frames)

            for b in range(batch_size):
                target_waveform = target[b].cpu()
                _, target_phase = waveform_to_magphase(target_waveform.unsqueeze(0), n_fft=n_fft, hop_length=hop_length)

                for i, src_name in enumerate(source_names_list[b]):
                    mag_pred = outputs[i][b]
                    phase = target_phase[0]
                    wav_rec = magphase_to_waveform(mag_pred, phase, n_fft=n_fft, hop_length=hop_length, length=target_waveform.shape[-1])
                    out_path = os.path.join(save_dir, f"gen_{src_name}")
                    torchaudio.save(out_path, wav_rec.unsqueeze(0), 16000)

                mag_sum = sum_outputs[b]
                wav_sum = magphase_to_waveform(mag_sum, phase, n_fft=n_fft, hop_length=hop_length, length=target_waveform.shape[-1])
                summed_path = os.path.join(save_dir, f"summed_{target_names[b]}")
                torchaudio.save(summed_path, wav_sum.unsqueeze(0), 16000)

                ref_path = os.path.join(save_dir, f"ref_{target_names[b]}")
                torchaudio.save(ref_path, target_waveform.unsqueeze(0), 16000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_path", default="model.ckpt")
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    dataset = AudioDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(args.mode == "train"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = TransformerAudioModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        train(model, dataloader, args.epochs, device, args.save_path)
    else:
        model.load_state_dict(torch.load(args.save_path, map_location=device))
        inference(model, dataloader, device, args.save_dir)
