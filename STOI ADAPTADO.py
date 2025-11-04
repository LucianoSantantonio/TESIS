# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:58:18 2025

@author: Luciano Santantonio
"""

import os
import numpy as np
import librosa

def load_and_preprocess(path, target_sr=10000):
    # Carga y convierte a mono + resamplea a 10 kHz
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def frame_audio(audio, frame_len, frame_shift):
    num_frames = 1 + int((len(audio) - frame_len) / frame_shift)
    frames = np.stack([audio[i * frame_shift : i * frame_shift + frame_len] for i in range(num_frames)])
    return frames

def compute_tf_representation(frames, n_fft=256):
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * win, n=n_fft))
    return spec

def get_band_indices(sr, n_fft):
    freqs = np.fft.rfftfreq(n_fft, d=1/sr)
    band_edges = [150, 190, 240, 300, 380, 480, 600, 760, 950, 1200, 1500,
                  1900, 2400, 3020, 4300, 5500]
    band_indices = []
    for i in range(13, 15):  
        f1, f2 = band_edges[i], band_edges[i+1]
        idx = np.where((freqs >= f1) & (freqs < f2))[0]
        band_indices.append(idx)
    return band_indices

def normalize_and_clip(Yj, Xj):
    eps = 1e-10
    alpha = np.sum(Xj * Yj) / (np.sum(Yj**2) + eps)
    Yj = alpha * Yj
    Yj = np.minimum(Yj, 1.5 * Xj)
    return Yj

def compute_stoi_adapted(clean_audio, mixed_audio, sr=10000):
    frame_len = 256
    frame_shift = 128

    clean_frames = frame_audio(clean_audio, frame_len, frame_shift)
    mixed_frames = frame_audio(mixed_audio, frame_len, frame_shift)
    min_frames = min(len(clean_frames), len(mixed_frames))
    clean_frames = clean_frames[:min_frames]
    mixed_frames = mixed_frames[:min_frames]

    clean_spec = compute_tf_representation(clean_frames)
    mixed_spec = compute_tf_representation(mixed_frames)

    band_indices = get_band_indices(sr, n_fft=256)

    djs = []
    for band in band_indices:
        Xj = np.linalg.norm(clean_spec[:, band], axis=1)
        Yj = np.linalg.norm(mixed_spec[:, band], axis=1)

        Yj_clipped = normalize_and_clip(Yj, Xj)

        mu_X = Xj - np.mean(Xj)
        mu_Y = Yj_clipped - np.mean(Yj_clipped)
        numerator = np.sum(mu_X * mu_Y)
        denominator = np.sqrt(np.sum(mu_X**2) * np.sum(mu_Y**2) + 1e-10)
        corr = numerator / denominator
        djs.append(corr)

    return np.mean(djs)
#%%
# === USO DEL SCRIPT ===
def main():
    # Archivos .mp3 exportados desde Moises
    vocal_path =   #solo voz
    mix_path   =  #voz+instrumental

    print("Cargando y procesando archivos .mp3...")
    vocal = load_and_preprocess(vocal_path)
    mix   = load_and_preprocess(mix_path)

    # Cortar al mismo largo
    min_len = min(len(vocal), len(mix))
    vocal = vocal[:min_len]
    mix   = mix[:min_len]

    print("Calculando STOI adaptado al canto...")
    score = compute_stoi_adapted(vocal, mix)
    print(f"\n✅ STOI adaptado al canto (bandas 13–15): {score:.4f}")

if __name__ == "__main__":
    main()

