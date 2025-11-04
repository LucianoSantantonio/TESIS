# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 21:45:32 2025

@author: Luciano
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. CARGAR DATOS ===
df = pd.read_excel("SUBJ VS OBJ.xlsx")

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()
print("\n--- Nombres de columnas detectados ---")
print(df.columns.tolist())

# === 2. IDENTIFICAR COLUMNA DE VALORACIÓN ===
# Buscamos una columna que contenga 'prom' o 'valora'
col_val_candidates = [c for c in df.columns if 'prom' in c or 'valora' in c]

if len(col_val_candidates) == 0:
    raise ValueError("❌ No se encontró ninguna columna que parezca ser la valoración promedio. Verificá el nombre en el Excel.")
else:
    col_val = col_val_candidates[0]
    print(f"\n✅ Columna de valoración detectada: '{col_val}'")

# Convertir todo a numérico
df = df.apply(pd.to_numeric, errors='coerce')

# === 3. CORRELACIONES (PEARSON) ===
print("\n--- RESULTADOS DE CORRELACIÓN (PEARSON) ---\n")

for col in df.columns:
    if col != col_val:
        subdata = df[[col_val, col]].dropna()
        if len(subdata) >= 3:
            r, p = stats.pearsonr(subdata[col_val], subdata[col])
            if p < 0.05:
                if abs(r) < 0.3:
                    interpretacion = "→ Correlación débil pero significativa."
                elif abs(r) < 0.6:
                    interpretacion = "→ Correlación moderada y significativa."
                else:
                    interpretacion = "→ Correlación fuerte y significativa."
            else:
                interpretacion = "→ No se encontró una correlación significativa."
            print(f"{col}: r = {r:.3f}, p = {p:.3f} {interpretacion}")
        else:
            print(f"{col}: no hay suficientes datos válidos para correlación.")

# === 4. CORRELACIONES (SPEARMAN) ===
print("\n--- RESULTADOS DE CORRELACIÓN (SPEARMAN) ---\n")

for col in df.columns:
    if col != col_val:
        subdata = df[[col_val, col]].dropna()
        if len(subdata) >= 3:
            rho, p = stats.spearmanr(subdata[col_val], subdata[col])
            if p < 0.05:
                if abs(rho) < 0.3:
                    interpretacion = "→ Correlación débil pero significativa."
                elif abs(rho) < 0.6:
                    interpretacion = "→ Correlación moderada y significativa."
                else:
                    interpretacion = "→ Correlación fuerte y significativa."
            else:
                interpretacion = "→ No se encontró una correlación significativa."
            print(f"{col}: rho = {rho:.3f}, p = {p:.3f} {interpretacion}")
            
# === 5. GRAFICAR RELACIÓN STOI vs VALORACIÓN ===
# === Gráfico de regresión lineal sin área de confianza ===
plt.figure(figsize=(7, 5))
sns.regplot(
    x="stoi", 
    y="prom", 
    data=df,
    ci=None,  # <-- elimina el área de confianza
    scatter_kws={"color": "royalblue", "s": 60, "alpha": 0.8},
    line_kws={"color": "darkorange", "linewidth": 2}
)

plt.title("Regresión Lineal", fontsize=13)
plt.xlabel("STOI adaptado", fontsize=12)
plt.ylabel("Calidad vocal subjetiva promedio", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()