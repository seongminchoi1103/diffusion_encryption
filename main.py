import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import librosa
import librosa.display
import soundfile as sf
from sklearn.model_selection import train_test_split

# =======================
# Configuration
# =======================
SAMPLE_RATE = 22050
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
SEGMENT_SECONDS = 5
MAX_LEN = SEGMENT_SECONDS * SAMPLE_RATE
NUM_DIFFUSION_STEPS = 2000


# =======================
# Utilities
# =======================
def load_wav(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)
    if len(y) > MAX_LEN:
        y = y[:MAX_LEN]
    elif len(y) < MAX_LEN:
        y = np.pad(y, (0, MAX_LEN - len(y)))
    return y[np.newaxis, :], sr  # shape: (1, samples)

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
    return torch.tensor(mel_db)  # (1, 1, F, T)

def mel_to_wav(mel_tensor):
    mel_db = mel_tensor.squeeze().cpu().numpy()  # shape: (n_mels, T)
    mel_power = librosa.db_to_power(mel_db)
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_iter=64
    )
    return torch.tensor(wav).unsqueeze(0)  # (1, samples)

def save_wav(tensor, path, sample_rate=SAMPLE_RATE):
    y = tensor.squeeze().cpu().numpy()
    sf.write(path, y, sample_rate)



# =======================
# Dataset
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


# =======================
# Model
# =======================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class AxialAttention(nn.Module):
    def __init__(self, channels, dim=None, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):  # (B, C, L)
        B, C, L = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.heads, self.head_dim, L).permute(0,1,3,2)
        k = k.view(B, self.heads, self.head_dim, L)
        v = v.view(B, self.heads, self.head_dim, L).permute(0,1,3,2)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0,1,3,2).contiguous().view(B, C, L)
        return self.proj(out)

class AxialAttention2D(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.height_attn = AxialAttention(channels, heads=heads)
        self.width_attn = AxialAttention(channels, heads=heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = x.permute(0,2,1,3).contiguous().view(B*H, C, W)
        x_h = self.height_attn(x_h)
        x_h = x_h.view(B, H, C, W).permute(0,2,1,3)

        x_w = x.permute(0,3,1,2).contiguous().view(B*W, C, H)
        x_w = self.width_attn(x_w)
        x_w = x_w.view(B, W, C, H).permute(0,2,3,1)

        return (x_h + x_w) / 2

class ResidualBlockWithAxialAttention(nn.Module):
    def __init__(self, channels, time_emb_dim=128, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.time_mlp = nn.Linear(time_emb_dim, channels)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        if use_attn:
            self.axial_attn = AxialAttention2D(channels, heads=4)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.relu(h)
        h = self.conv1(h)

        t_emb_ = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb_

        h = self.norm2(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.se(h)

        if self.use_attn:
            h = self.axial_attn(h)

        return x + h

class DiffusionUNet(nn.Module):
    def __init__(self, time_emb_dim=128):
        super().__init__()
        self.time_embedding = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoding path
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.res1 = ResidualBlockWithAxialAttention(64, time_emb_dim, use_attn=False)

        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res2 = ResidualBlockWithAxialAttention(128, time_emb_dim, use_attn=True)

        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.res3 = ResidualBlockWithAxialAttention(256, time_emb_dim, use_attn=True)

        self.enc4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.res6 = ResidualBlockWithAxialAttention(512, time_emb_dim, use_attn=True)

        # Middle block 추가
        self.middle_res = ResidualBlockWithAxialAttention(512, time_emb_dim, use_attn=True)

        # Decoding path
        self.dec3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.res7 = ResidualBlockWithAxialAttention(256, time_emb_dim, use_attn=True)

        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.res4 = ResidualBlockWithAxialAttention(128, time_emb_dim, use_attn=True)

        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.res5 = ResidualBlockWithAxialAttention(64, time_emb_dim, use_attn=False)

        # 디코더에 더 깊은 레이어 추가
        self.dec_extra = nn.Conv2d(64, 32, 3, padding=1)
        self.res_extra = ResidualBlockWithAxialAttention(32, time_emb_dim, use_attn=False)

        self.skip1_proj = nn.Conv2d(64, 32, 1)
        self.out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        x1 = F.relu(self.enc1(x))
        x1 = self.res1(x1, t_emb)

        x2 = F.relu(self.enc2(x1))
        x2 = self.res2(x2, t_emb)

        x3 = F.relu(self.enc3(x2))
        x3 = self.res3(x3, t_emb)

        x4 = F.relu(self.enc4(x3))
        x4 = self.res6(x4, t_emb)

        # Middle block 통과
        x4 = self.middle_res(x4, t_emb)

        x = F.relu(self.dec3(x4))
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear')
        x = x + x3
        x = self.res7(x, t_emb)

        x = F.relu(self.dec1(x))
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear')
        x = x + x2
        x = self.res4(x, t_emb)

        x = F.relu(self.dec2(x))
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = x + x1
        x = self.res5(x, t_emb)

        # 추가된 디코더 레이어에도 skip 연결 (enc1 → dec_extra)
        x = F.relu(self.dec_extra(x))
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x1_proj = self.skip1_proj(x1)
        x = x + x1_proj
        x = self.res_extra(x, t_emb)

        return self.out(x)

# =======================
# Diffusion Schedule
# =======================
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(NUM_DIFFUSION_STEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def q_sample(x_start, t, noise):
    device = x_start.device
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod.to(device)[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod).to(device)[t][:, None, None, None]
    return sqrt_alpha_cumprod * x_start + sqrt_one_minus * noise


def p_sample(x, model, t):
    device = x.device
    t_tensor = torch.tensor([t], device=device)  # 💡 여기가 핵심

    beta = betas[t].to(device)
    alpha = alphas[t].to(device)
    alpha_bar = alphas_cumprod[t].to(device)

    pred_noise = model(x, t_tensor)  # t를 텐서로
    coef1 = 1 / torch.sqrt(alpha)
    coef2 = beta / torch.sqrt(1 - alpha_bar)
    mean = coef1 * (x - coef2 * pred_noise)

    if t == 0:
        return mean
    noise = torch.randn_like(x)
    return mean + torch.sqrt(beta) * noise

# =======================
# Training
# =======================
def train(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # =====================
    # Load Dataset & Split
    # =====================
    dataset = MelDataset(args.wav_dir)
    total_indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(total_indices, test_size=0.1, random_state=42)

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # =====================
    # Model Init / Resume
    # =====================
    model = DiffusionUNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters : {total_params:,}")

    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model_state = model.state_dict()
        filtered_ckpt = {
            k: v for k, v in checkpoint.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered_ckpt, strict=False)
        print(f"✅ Loaded {len(filtered_ckpt)} matching parameters from checkpoint.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # =====================
    # Prepare log file
    # =====================
    log_path = os.path.join(args.checkpoint_dir, "train_log.txt")
    best_val_loss = float('inf')

    with open(log_path, "a") as f_log:
        f_log.write(f"Training started\n")
        f_log.write(f"Total trainable parameters: {total_params:,}\n")
        f_log.flush()

    # =====================
    # Training Loop
    # =====================
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{args.epochs}] Training")
        total_loss = 0

        for mel in pbar:
            mel = mel.to(device)
            t = torch.randint(0, NUM_DIFFUSION_STEPS, (mel.size(0),), device=device).long()
            noise = torch.randn_like(mel)
            noised = q_sample(mel, t, noise)
            pred_noise = model(noised, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

        # =====================
        # Validation
        # =====================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mel in val_loader:
                mel = mel.to(device)
                t = torch.randint(0, NUM_DIFFUSION_STEPS, (mel.size(0),), device=device).long()
                noise = torch.randn_like(mel)
                noised = q_sample(mel, t, noise)
                pred_noise = model(noised, t)
                val_loss += F.mse_loss(pred_noise, noise).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"🧪 Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f}")

        # =====================
        # Save checkpoint
        # =====================
        save_path = os.path.join(args.checkpoint_dir, f"model_mine_deeper.pt")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")

        # Validation loss 기준으로 best 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(args.checkpoint_dir, "model_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New best model saved to {best_path}")

        # =====================
        # Write log file
        # =====================
        with open(log_path, "a") as f_log:
            f_log.write(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n")
            f_log.flush()


# =======================
# Inference
# =======================
def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 1) 입력 WAV → Mel 변환
    wav, sr = load_wav(args.input_wav)
    mel = wav_to_mel(wav).unsqueeze(0).to(device)  # (1, 1, F, T)
    save_wav(mel_to_wav(mel).cpu(), os.path.join(args.out_dir, "input_mel.wav"))

    # 2) Fully Noised Mel 생성
    t_max = NUM_DIFFUSION_STEPS - 1
    noise = torch.randn_like(mel)
    noised = q_sample(mel, torch.tensor([t_max], device=device), noise)

    # 저장용 WAV
    noised_wav = mel_to_wav(noised.squeeze(0))
    save_wav(noised_wav.cpu(), os.path.join(args.out_dir, "fully_noised.wav"))

    # 3) Denoising
    x = noised.clone()
    with torch.no_grad():
        for t in reversed(range(NUM_DIFFUSION_STEPS)):
            x = p_sample(x, model, t)

    denoised = x
    reconstructed_wav = mel_to_wav(denoised.squeeze(0))
    save_wav(reconstructed_wav.cpu(), os.path.join(args.out_dir, "reconstructed.wav"))

    # 4) Mel Spectrogram 3개를 하나의 창에 출력
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Original Mel", "Fully Noised Mel", "Reconstructed Mel"]
    mel_specs = [mel, noised, denoised]

    for i, (spec, title) in enumerate(zip(mel_specs, titles)):
        im = axs[i].imshow(spec[0, 0].detach().cpu(), origin="lower", aspect="auto")
        axs[i].set_title(title)
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Mel bins")

    # 공통 colorbar 추가
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# =======================
# Entry Point
# =======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)

    # Training args
    parser.add_argument("--wav_dir", type=str, help="Directory of wav files")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)

    # Inference args
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--input_wav", type=str, help="Path to input wav file")
    parser.add_argument("--out_dir", type=str, default="outputs")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)


if __name__ == "__main__":
    main()
