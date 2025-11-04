# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 20:40:26 2025

@author: lucia
"""

import os
import librosa
import numpy as np
import pandas as pd

def calcular_spr(audio_path, low_band=(0, 2000), high_band=(2000, 4000), top_db=40):
    y, sr = librosa.load(audio_path, sr=None)

    # Detectar fragmentos con voz (omitir silencios)
    intervals = librosa.effects.split(y, top_db=top_db)

    if len(intervals) == 0:
        return np.nan  # No hay voz detectada

    # Concatenar todos los segmentos con voz
    y_voiced = np.concatenate([y[start:end] for start, end in intervals])

    # Calcular espectro de potencia
    S = np.abs(librosa.stft(y_voiced))**2
    freqs = librosa.fft_frequencies(sr=sr)

    # Energía en banda baja (ej: 0–1 kHz)
    low_energy = np.sum(S[(freqs >= low_band[0]) & (freqs < low_band[1]), :])
    
    # Energía en banda alta (ej: 2–4 kHz)
    high_energy = np.sum(S[(freqs >= high_band[0]) & (freqs < high_band[1]), :])

    if low_energy == 0:
        return np.nan  # Evitar división por cero

    spr = 10 * np.log10(high_energy / low_energy)
    return spr

# Ruta de la carpeta con los archivos .wav
carpeta = r"C:/Users/Luciano/Desktop/TESIS/Estimulos Tesis/SEPARADO/VOCALES/wav"  # 🔁 Cambiá esto por tu ruta real
resultados = []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".wav"):
        path_completo = os.path.join(carpeta, archivo)
        spr = calcular_spr(path_completo)
        resultados.append({'archivo': archivo, 'SPR_dB': spr})

# Guardar los resultados en un CSV
df = pd.DataFrame(resultados)
df.to_csv('resultados_SPR.csv', index=False)

print("✅ Proceso completado. Resultados guardados en 'resultados_SPR.csv'")
