# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 17:54:03 2025

@author: lucia
"""

import os
import numpy as np
import librosa
import pandas as pd

def spectral_peak_energy(audio_path, sr=22050, n_fft=2048, hop_length=512):
    """
    Calcula el promedio de la energía del pico espectral de un archivo de audio.
    
    Parámetros:
        audio_path (str): Ruta al archivo de audio.
        sr (int): Frecuencia de muestreo para cargar el audio.
        n_fft (int): Tamaño de la FFT.
        hop_length (int): Salto entre ventanas.
    
    Retorna:
        float: Valor promedio del pico espectral (energía máxima por frame).
    """
    # Cargar audio
    y, _ = librosa.load(audio_path, sr=sr)
    
    # Calcular magnitud espectral (STFT)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    
    # Para cada frame, tomar el valor máximo (pico espectral)
    peak_energy_per_frame = np.max(S, axis=0)
    
    # Retornar promedio de pico espectral
    return np.mean(peak_energy_per_frame)

def procesar_carpeta_audio(carpeta, output_csv):
    """
    Procesa todos los archivos wav en una carpeta, calcula SPE y guarda en CSV.
    
    Parámetros:
        carpeta (str): Ruta a la carpeta con archivos .wav
        output_csv (str): Ruta del archivo CSV de salida
    """
    resultados = []
    
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith('.wav'):
            ruta_audio = os.path.join(carpeta, archivo)
            spe = spectral_peak_energy(ruta_audio)
            resultados.append({'archivo': archivo, 'spectral_peak_energy': spe})
            print(f'Procesado {archivo}: SPE = {spe:.2f}')
    
    # Guardar resultados en DataFrame y CSV
    df = pd.DataFrame(resultados)
    df.to_csv(output_csv, index=False)
    print(f'Resultados guardados en {output_csv}')

if __name__ == "__main__":
    carpeta_audios = r"C:/Users/lucia/OneDrive/Escritorio/TESIS/Estimulos Tesis/SEPARADO/VOCALES/wav"  # Cambia esto por tu carpeta
    archivo_salida = 'resultados_SPE.csv'  # Nombre del CSV de salida
    
    procesar_carpeta_audio(carpeta_audios, archivo_salida)
