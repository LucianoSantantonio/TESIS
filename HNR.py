# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:44:09 2025

@author: lucia
"""

import os
import parselmouth
import pandas as pd

import os
import librosa
import parselmouth
import pandas as pd
import soundfile as sf

def recortar_silencio_librosa(ruta_audio, top_db=30):
    y, sr = librosa.load(ruta_audio, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    # Guardar el segmento sin silencio en un archivo temporal
    archivo_temp = "temp_trim.wav"
    sf.write(archivo_temp, y_trimmed, sr)
    return archivo_temp

def calcular_hnr_parselmouth(ruta_audio):
    try:
        snd = parselmouth.Sound(ruta_audio)
        harmonicity = snd.to_harmonicity_cc()
        valores = harmonicity.values
        valores = valores[(valores != float("-inf")) & (valores > -100)]  # filtrar errores
        if len(valores) == 0:
            return None
        return valores.mean()
    except Exception as e:
        print(f"Error procesando {ruta_audio}: {e}")
        return None

# Cambiar a tu carpeta con audios .wav
carpeta = r"C:/Users/lucia/OneDrive/Escritorio/TESIS/Estimulos Tesis/SEPARADO/VOCALES/wav"

resultados = []

for archivo in os.listdir(carpeta):
    if archivo.lower().endswith(".wav"):
        ruta_completa = os.path.join(carpeta, archivo)
        # Recortar silencio
        ruta_trim = recortar_silencio_librosa(ruta_completa, top_db=30)
        # Calcular HNR en segmento sin silencio
        hnr = calcular_hnr_parselmouth(ruta_trim)
        resultados.append({"Archivo": archivo, "HNR promedio (dB)": hnr})

# Guardar resultados en CSV
df = pd.DataFrame(resultados)
df.to_csv("resultados_HNR.csv", index=False, encoding="utf-8-sig")

print("✅ Análisis completado. Resultados guardados.csv")


