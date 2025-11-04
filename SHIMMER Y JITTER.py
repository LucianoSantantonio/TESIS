# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:22:05 2025

@author: lucia
"""

import parselmouth
from parselmouth.praat import call
import os
import csv

def analyze_voice(wav_path):
    snd = parselmouth.Sound(wav_path)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)

    # Jitter
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # Shimmer
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return jitter, shimmer

def process_all_wavs(carpeta):
    resultados = []

    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(".wav"):
            ruta_completa = os.path.join(carpeta, archivo)
            try:
                jitter, shimmer = analyze_voice(ruta_completa)
                resultados.append([archivo, jitter, shimmer])
                print(f"Procesado: {archivo}")
            except Exception as e:
                print(f"Error en {archivo}: {e}")
                resultados.append([archivo, "ERROR", "ERROR"])

    # Guardar en CSV
    with open("resultados_jitter_shimmer.csv", mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Archivo", "Jitter (local)", "Shimmer (local)"])
        writer.writerows(resultados)

    print("\n✅ Análisis completo. Resultados guardados en 'resultados_jitter_shimmer.csv'")

#%% ▶️ USO
if __name__ == "__main__":
    carpeta_entrada = "C:/Users/lucia/OneDrive/Escritorio/TESIS/Estimulos Tesis/SEPARADO/VOCALES/wav"
    process_all_wavs(carpeta_entrada)