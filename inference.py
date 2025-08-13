import os
import argparse
import torch
import matplotlib.pyplot as plt
from dataset import load_wav, wav_to_mel, mel_to_wav, save_wav
from model import DiffusionUNet

NUM_DIFFUSION_STEPS = 2000
def linear_beta_schedule(timesteps):
    return torch.linspace(0.0001, 0.02, timesteps)

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
    t_tensor = torch.tensor([t], device=device)
    beta = betas[t].to(device)
    alpha = alphas[t].to(device)
    alpha_bar = alphas_cumprod[t].to(device)
    pred_noise = model(x, t_tensor)
    coef1 = 1 / torch.sqrt(alpha)
    coef2 = beta / torch.sqrt(1 - alpha_bar)
    mean = coef1 * (x - coef2 * pred_noise)
    if t == 0:
        return mean
    noise = torch.randn_like(x)
    return mean + torch.sqrt(beta) * noise

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionUNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    wav, _ = load_wav(args.input_wav)
    mel = wav_to_mel(wav).unsqueeze(0).to(device)
    save_wav(mel_to_wav(mel).cpu(), os.path.join(args.out_dir, "input_mel.wav"))

    t_max = NUM_DIFFUSION_STEPS - 1
    noise = torch.randn_like(mel)
    noised = q_sample(mel, torch.tensor([t_max], device=device), noise)
    save_wav(mel_to_wav(noised.squeeze(0)).cpu(), os.path.join(args.out_dir, "fully_noised.wav"))

    x = noised.clone()
    with torch.no_grad():
        for t in reversed(range(NUM_DIFFUSION_STEPS)):
            x = p_sample(x, model, t)

    save_wav(mel_to_wav(x.squeeze(0)).cpu(), os.path.join(args.out_dir, "reconstructed.wav"))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Original Mel", "Fully Noised Mel", "Reconstructed Mel"]
    mel_specs = [mel, noised, x]

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--input_wav", type=str, help="Path to input wav file")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()
    inference(args)
