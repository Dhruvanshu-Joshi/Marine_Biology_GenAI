# --- Imports ---
import math
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from PIL import Image
from torchvision import transforms
from typing_extensions import Self
from fathomnet.api import images


# --- Utility functions ---
def inv_mag(x):
    fft_ = torch.fft.fft2(x)
    fft_ = torch.fft.ifft2(1 * torch.exp(1j * (fft_.angle())))
    return fft_.real


# --- Model Components ---
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(
            channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Frequency
        self.kv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.q1X1_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.q1X1_2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv_conv = nn.Conv2d(
            channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False
        )
        self.project_outf = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))

        # Frequency branch
        x_fft = fft.fftn(x, dim=(-2, -1)).real
        x_fft1 = self.q1X1_1(x_fft)
        x_fft2 = F.gelu(x_fft1)
        x_fft3 = self.q1X1_2(x_fft2)
        qf = fft.ifftn(x_fft3, dim=(-2, -1)).real

        kf, vf = self.kv_conv(self.kv(out)).chunk(2, dim=1)
        qf = qf.reshape(b, self.num_heads, -1, h * w)
        kf = kf.reshape(b, self.num_heads, -1, h * w)
        vf = vf.reshape(b, self.num_heads, -1, h * w)
        qf, kf = F.normalize(qf, dim=-1), F.normalize(kf, dim=-1)
        attnf = torch.softmax(torch.matmul(qf, kf.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        outf = self.project_outf(torch.matmul(attnf, vf).reshape(b, -1, h, w))
        return outf


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        x = x + self.ffn(
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels, channel_red):
        super(UpSample, self).__init__()
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        if channel_red:
            self.post = nn.Conv2d(channels, channels // 2, 1, 1, 0)
        else:
            self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)
        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        output = torch.fft.ifft2(out)
        return self.post(torch.abs(output))


class UpSample1(nn.Module):
    def __init__(self, channels):
        super(UpSample1, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class UpS(nn.Module):
    def __init__(self, channels):
        super(UpS, self).__init__()
        self.Fups = UpSample(channels, True)
        self.Sups = UpSample1(channels)
        self.reduce = nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False)

    def forward(self, x):
        out = torch.cat([self.Fups(x), self.Sups(x)], dim=1)
        return self.reduce(out)


# --- Main Model ---
class mymodel(nn.Module):
    def __init__(self, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8], channels=[16, 32, 64, 128],
                 num_refinement=4, expansion_factor=2.66, ch=[64, 32, 16, 64]):
        super(mymodel, self).__init__()
        self.embed_conv_rgb = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.encoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(num_ch, num_ah, expansion_factor)
                            for _ in range(num_tb)])
            for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
        ])

        self.down1 = DownSample(channels[0])
        self.down2 = DownSample(channels[1])
        self.down3 = DownSample(channels[2])
        self.ups_1 = UpS(128)
        self.ups_2 = UpS(64)
        self.ups_3 = UpS(32)
        self.ups_4 = UpS(3)

        self.ups1 = UpSample1(32)
        self.reduces2 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.reduces1 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        self.decoders = nn.ModuleList([
            nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                            for _ in range(num_blocks[2])])
        ])
        self.decoders.append(
            nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                             for _ in range(num_blocks[1])])
        )
        self.decoders.append(
            nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                             for _ in range(num_blocks[0])])
        )

        self.refinement = nn.Sequential(*[
            TransformerBlock(channels[1], num_heads[0], expansion_factor)
            for _ in range(num_refinement)
        ])
        self.output = nn.Conv2d(8, 3, kernel_size=3, padding=1, bias=False)
        self.output1 = nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False)
        self.ups2 = UpSample1(16)
        self.outputl = nn.Conv2d(32, 8, kernel_size=3, padding=1, bias=False)

    def forward(self, RGB_input):
        fo_rgb = self.embed_conv_rgb(RGB_input)
        out_enc_rgb1 = self.encoders[0](fo_rgb)
        out_enc_rgb2 = self.encoders[1](self.down1(out_enc_rgb1))
        out_enc_rgb3 = self.encoders[2](self.down2(out_enc_rgb2))
        out_enc_rgb4 = self.encoders[3](self.down3(out_enc_rgb3))

        out_dec3 = self.decoders[0](self.reduces1(torch.cat([self.ups_1(out_enc_rgb4), out_enc_rgb3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces2(torch.cat([self.ups_2(out_dec3), out_enc_rgb2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups_3(out_dec2), out_enc_rgb1], dim=1))
        fr = self.refinement(fd)
        return self.output(self.outputl(fr))


# --- Scoring functions ---
def score_crop_quality(crop, weights=(0.2, 0.3, 0.5)):
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    b_score = min(max((brightness - 30) / (255 - 30), 0), 1)
    s_score = min(sharpness / 300.0, 1.0)
    c_score = min(contrast / 60.0, 1.0)
    return weights[0] * b_score + weights[1] * s_score + weights[2] * c_score
