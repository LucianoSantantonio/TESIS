# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:26:48 2025

@author: lucia
"""

import os
import librosa
import numpy as np
import pandas as pd

def normalized_autocorrelation_peak(audio, sr, frame_length=2048, hop_length=512):
    audio = audio / np.max(np.abs(audio))  # Normalizar amplitud general
    num_frames = 1 + (len(audio) - frame_length) // hop_length
    peaks = []

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        if end > len(audio):
            break
        frame = audio[start:end]

        ac = np.correlate(frame, frame, mode='full')
        ac = ac[len(ac)//2:]  # mitad derecha

        max_ac = np.max(np.abs(ac))
        if max_ac == 0:
            continue  # saltar este frame si la autocorrelación es toda cero

        ac /= max_ac  # normalizar

        peak = np.max(ac[1:])  # ignorar lag=0
        if not np.isnan(peak):
            peaks.append(peak)

    return np.mean(peaks) if peaks else np.nan


def analizar_directorio_y_guardar_csv(directorio, salida_csv):
    resultados = []

    for archivo in os.listdir(directorio):
        if archivo.endswith(".wav"):
            path = os.path.join(directorio, archivo)
            try:
                audio, sr = librosa.load(path, sr=None, mono=True)
                peak = normalized_autocorrelation_peak(audio, sr)
                if not np.isnan(peak):
                    resultados.append({"archivo": archivo, "normalized_autocorrelation_peak": peak})
                else:
                    print(f"⚠️  {archivo}: no se pudo calcular el NAP (todo NaN)")
            except Exception as e:
                print(f"❌ Error procesando {archivo}: {e}")

    if resultados:
        df = pd.DataFrame(resultados)
        df.to_csv(salida_csv, index=False)
        print(f"✅ Resultados guardados en: {salida_csv}")
    else:
        print("⚠️  No se generaron resultados válidos. Verificá los archivos de audio.")


#%% 📍 Uso
directorio_entrada = "C:/Users/lucia/OneDrive/Escritorio/TESIS/Estimulos Tesis/SEPARADO/VOCALES/wav"
archivo_salida = "resultados_NAP.csv"

analizar_directorio_y_guardar_csv(directorio_entrada, archivo_salida)
