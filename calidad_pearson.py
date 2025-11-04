# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:59:49 2025

@author: Luciano
"""
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- Cargar datos ---
df = pd.read_excel("calidad.xlsx")

# --- Detectar columnas ---
general_cols = [c for c in df.columns if "general" in c.lower()]
vocal_cols = [c for c in df.columns if "voces" in c.lower()]

# --- Extraer número del estímulo para ordenar correctamente ---
def extract_num(col):
    match = re.search(r'(\d+)', col)
    return int(match.group(1)) if match else None

general_cols = sorted(general_cols, key=extract_num)
vocal_cols = sorted(vocal_cols, key=extract_num)

print("Columnas general:", general_cols)
print("Columnas vocal:", vocal_cols)

# --- Verificar que haya la misma cantidad de columnas ---
assert len(general_cols) == len(vocal_cols), "Las columnas general y vocal no coinciden en cantidad."

# =====================================================
# OPCIÓN A: correlación entre promedios por estímulo
# =====================================================
promedios_general = df[general_cols].mean()
promedios_vocal = df[vocal_cols].mean()

r_prom, p_prom = pearsonr(promedios_general, promedios_vocal)

print("\n🔹 RESULTADO A: Correlación entre promedios por estímulo")
print(f"Coeficiente de Pearson (r): {r_prom:.3f}")
print(f"Valor p: {p_prom:.4f}")

# --- Interpretación ---
if abs(r_prom) < 0.3:
    interpretacion_prom = "relación débil o casi nula"
elif abs(r_prom) < 0.5:
    interpretacion_prom = "relación moderada"
elif abs(r_prom) < 0.7:
    interpretacion_prom = "relación considerable"
else:
    interpretacion_prom = "relación fuerte"

significativo_prom = "estadísticamente significativa" if p_prom < 0.05 else "no significativa"

print(f"👉 Interpretación: Existe una {interpretacion_prom} entre las puntuaciones promedio de calidad general y vocal, "
      f"siendo {significativo_prom} (p = {p_prom:.4f}).")

# --- Gráfico de promedios ---
plt.figure(figsize=(6,5))
sns.regplot(x=promedios_general, y=promedios_vocal, ci=None, color="red")
plt.xlabel("Puntuación general")
plt.ylabel("Puntuación vocal")
plt.title("Correlación entre promedios por estímulo")
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# OPCIÓN B: correlación considerando todas las respuestas
# =====================================================
# Pasamos a formato largo
general_long = df[general_cols].melt(var_name='estimulo', value_name='general')
vocal_long = df[vocal_cols].melt(var_name='estimulo', value_name='vocal')

# --- Asegurar misma longitud ---
min_len = min(len(general_long), len(vocal_long))
df_long = pd.DataFrame({
    "general": general_long["general"].iloc[:min_len].values,
    "vocal": vocal_long["vocal"].iloc[:min_len].values
})

# --- Correlación global ---
r_all, p_all = pearsonr(df_long['general'], df_long['vocal'])

print("\n🔹 RESULTADO B: Correlación considerando todas las respuestas")
print(f"Coeficiente de Pearson (r): {r_all:.3f}")
print(f"Valor p: {p_all:.4f}")

# --- Interpretación ---
if abs(r_all) < 0.3:
    interpretacion_all = "relación débil o casi nula"
elif abs(r_all) < 0.5:
    interpretacion_all = "relación moderada"
elif abs(r_all) < 0.7:
    interpretacion_all = "relación considerable"
else:
    interpretacion_all = "relación fuerte"

significativo_all = "estadísticamente significativa" if p_all < 0.05 else "no significativa"

print(f"👉 Interpretación: Existe una {interpretacion_all} entre las puntuaciones individuales de calidad general y vocal, "
      f"siendo {significativo_all} (p = {p_all:.4f}).")

# --- Gráfico global ---
plt.figure(figsize=(6,5))
sns.scatterplot(x="general", y="vocal", data=df_long, alpha=0.5)
sns.regplot(x="general", y="vocal", data=df_long, ci=None, scatter=False, color="red")
plt.xlabel("Puntuación general")
plt.ylabel("Puntuación vocal")
plt.title("Correlación entre todas las respuestas")
plt.grid(True)
plt.tight_layout()
plt.show()
