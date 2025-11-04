# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 18:58:18 2025

@author: Luciano Santantonio
Adaptado para directorios separados con sufijo -vocals
"""

import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

def load_and_preprocess(path, target_sr=10000):
    """Carga y preprocesa audio"""
    audio, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def frame_audio(audio, frame_len, frame_shift):
    """Divide el audio en frames"""
    num_frames = 1 + int((len(audio) - frame_len) / frame_shift)
    frames = np.stack([audio[i * frame_shift : i * frame_shift + frame_len] 
                      for i in range(num_frames)])
    return frames

def compute_tf_representation(frames, n_fft=256):
    """Calcula representación tiempo-frecuencia"""
    win = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * win, n=n_fft))
    return spec

def get_band_indices(sr, n_fft):
    """Obtiene índices de bandas frecuenciales relevantes para canto"""
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
    """Normaliza y recorta las señales"""
    eps = 1e-10
    alpha = np.sum(Xj * Yj) / (np.sum(Yj**2) + eps)
    Yj = alpha * Yj
    Yj = np.minimum(Yj, 1.5 * Xj)
    return Yj

def compute_stoi_adapted(clean_audio, mixed_audio, sr=10000):
    """Calcula métrica STOI adaptada para canto"""
    frame_len = 256
    frame_shift = 128

    # Asegurar misma longitud
    min_len = min(len(clean_audio), len(mixed_audio))
    clean_audio = clean_audio[:min_len]
    mixed_audio = mixed_audio[:min_len]

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

def find_matching_pairs(vocal_dir, mix_dir):
    """
    Encuentra pares de archivos donde:
    - Archivos vocales terminan en -vocals
    - Archivos mezclados tienen el mismo nombre base pero sin -vocals
    """
    # Obtener listados de archivos
    vocal_files = [f for f in os.listdir(vocal_dir) 
                 if (f.endswith('.wav') or f.endswith('.mp3')) and '-vocals' in f]
    mix_files = [f for f in os.listdir(mix_dir) 
               if f.endswith('.wav') or f.endswith('.mp3')]
    
    pairs = []
    
    for v_file in vocal_files:
        # Extraer el nombre base (remover -vocals y extensión)
        base_name = v_file.split('-vocals')[0]
        
        # Buscar posibles coincidencias en los archivos mezclados
        possible_matches = [f for f in mix_files 
                          if f.startswith(base_name) and '-vocals' not in f]
        
        # Considerar diferentes extensiones
        exact_matches = [f for f in possible_matches 
                        if f.split('.')[0] == base_name]
        
        if exact_matches:
            # Preferir match exacto (mismo nombre base)
            m_file = exact_matches[0]
        elif possible_matches:
            # Si no hay match exacto, tomar el primero que empiece con el nombre base
            m_file = possible_matches[0]
        else:
            continue
        
        pairs.append((v_file, m_file))
    
    return pairs

def process_batch(vocal_dir, mix_dir, output_csv='stoi_results.csv'):
    """Procesa todos los pares de archivos en directorios separados"""
    file_pairs = find_matching_pairs(vocal_dir, mix_dir)
    
    if not file_pairs:
        print("⚠️ No se encontraron pares de archivos coincidentes")
        print("Requisitos:")
        print("- Archivos vocales deben terminar en -vocals (ej: 'canción1-vocals.wav')")
        print("- Archivos mezclados deben tener el mismo nombre base pero sin -vocals (ej: 'canción1.wav')")
        return
    
    results = []
    
    print(f"🔍 Encontrados {len(file_pairs)} pares de archivos para procesar")
    
    for v_file, m_file in tqdm(file_pairs, desc="Procesando archivos"):
        try:
            # Cargar y procesar audios
            vocal_path = os.path.join(vocal_dir, v_file)
            mix_path = os.path.join(mix_dir, m_file)
            
            vocal = load_and_preprocess(vocal_path)
            mix = load_and_preprocess(mix_path)
            
            # Calcular STOI
            stoi_score = compute_stoi_adapted(vocal, mix)
            
            results.append({
                'archivo_vocal': v_file,
                'archivo_mix': m_file,
                'stoi_score': stoi_score,
                'duracion_seg': len(vocal)/10000  # Asumiendo SR=10kHz
            })
            
        except Exception as e:
            print(f"⚠️ Error procesando {v_file} (vocal) y {m_file} (mix): {str(e)}")
    
    # Crear DataFrame y guardar CSV
    df = pd.DataFrame(results)
    df = df.sort_values('stoi_score', ascending=False)
    df.to_csv(output_csv, index=False, float_format='%.4f')
    
    print(f"\n✅ Procesamiento completado. Resultados guardados en {output_csv}")
    print(f"📊 Estadísticas:\n{df['stoi_score'].describe()}")
    
    return df

#%% === USO DEL SCRIPT ===
if __name__ == "__main__":
    # Configuración de directorios
    VOCAL_DIR = "C:/Users/lucia/OneDrive/Escritorio/TESIS/Estimulos Tesis/SEPARADO/VOCALES"  # Contiene archivos que terminan en -vocals
    MIX_DIR = "C:/Users/lucia/OneDrive/Escritorio/TESIS/Estimulos Tesis/PARA EL TEST"    # Contiene archivos con nombres equivalentes sin -vocals
    OUTPUT_CSV = "resultados_stoi.csv"     # Nombre del archivo de salida
    
    # Procesar todos los archivos
    process_batch(VOCAL_DIR, MIX_DIR, OUTPUT_CSV)