import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import MelDataset
from model import DiffusionUNet

NUM_DIFFUSION_STEPS = 2000
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

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataset = MelDataset(args.wav_dir)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)

    model = DiffusionUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for mel in tqdm(train_loader, desc=f"[Epoch {epoch+1}]"):
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

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for mel in val_loader:
                mel = mel.to(device)
                t = torch.randint(0, NUM_DIFFUSION_STEPS, (mel.size(0),), device=device).long()
                noise = torch.randn_like(mel)
                noised = q_sample(mel, t, noise)
                pred_noise = model(noised, t)
                val_loss += F.mse_loss(pred_noise, noise).item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_last.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "model_best.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    train(args)
