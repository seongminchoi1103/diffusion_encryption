import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
# Model Components
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
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Conv1d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
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

        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.res1 = ResidualBlockWithAxialAttention(64, time_emb_dim, use_attn=False)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.res2 = ResidualBlockWithAxialAttention(128, time_emb_dim, use_attn=True)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.res3 = ResidualBlockWithAxialAttention(256, time_emb_dim, use_attn=True)
        self.enc4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.res6 = ResidualBlockWithAxialAttention(512, time_emb_dim, use_attn=True)

        # Middle
        self.middle_res = ResidualBlockWithAxialAttention(512, time_emb_dim, use_attn=True)

        # Decoder
        self.dec3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.res7 = ResidualBlockWithAxialAttention(256, time_emb_dim, use_attn=True)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.res4 = ResidualBlockWithAxialAttention(128, time_emb_dim, use_attn=True)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.res5 = ResidualBlockWithAxialAttention(64, time_emb_dim, use_attn=False)
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
        x4 = self.middle_res(x4, t_emb)

        x = F.relu(self.dec3(x4))
        if x.shape[2:] != x3.shape[2:]:
            x = F.interpolate(x, size=x3.shape[2:], mode='bilinear')
        x = self.res7(x + x3, t_emb)

        x = F.relu(self.dec1(x))
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear')
        x = self.res4(x + x2, t_emb)

        x = F.relu(self.dec2(x))
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = self.res5(x + x1, t_emb)

        x = F.relu(self.dec_extra(x))
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = self.res_extra(x + self.skip1_proj(x1), t_emb)
        return self.out(x)
