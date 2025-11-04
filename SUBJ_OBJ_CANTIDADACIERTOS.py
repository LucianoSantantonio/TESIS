# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 22:12:33 2025

@author: Luciano
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 22:10:00 2025
@author: Luciano
Analiza la correlación entre los parámetros objetivos y la cantidad de respuestas correctas del test.
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. CARGAR DATOS ===
df = pd.read_excel("SUBvsOBJ_aciertos.xlsx")

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()
print("\n--- Nombres de columnas detectados ---")
print(df.columns.tolist())

# === 2. IDENTIFICAR COLUMNA DE RESPUESTAS CORRECTAS ===
# Buscamos una columna que contenga 'acierto', 'correct', o 'resp'
col_resp_candidates = [c for c in df.columns if 'acierto' in c or 'correct' in c or 'resp' in c]

if len(col_resp_candidates) == 0:
    raise ValueError("❌ No se encontró ninguna columna que parezca indicar la cantidad de respuestas correctas. Verificá el nombre en el Excel.")
else:
    col_resp = col_resp_candidates[0]
    print(f"\n✅ Columna de respuestas correctas detectada: '{col_resp}'")

# Convertir todo a numérico
df = df.apply(pd.to_numeric, errors='coerce')

# === 3. CORRELACIONES (PEARSON) ===
print("\n--- RESULTADOS DE CORRELACIÓN (PEARSON) ---\n")

for col in df.columns:
    if col != col_resp:
        subdata = df[[col_resp, col]].dropna()
        if len(subdata) >= 3:
            r, p = stats.pearsonr(subdata[col_resp], subdata[col])
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
    if col != col_resp:
        subdata = df[[col_resp, col]].dropna()
        if len(subdata) >= 3:
            rho, p = stats.spearmanr(subdata[col_resp], subdata[col])
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
        else:
            print(f"{col}: no hay suficientes datos válidos para correlación.")

# === 5. GRAFICAR RELACIÓN ENTRE SPR Y RESPUESTAS CORRECTAS ===
if "spr" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.regplot(
        x="spr", 
        y=col_resp, 
        data=df,
        ci=None,
        scatter_kws={"color": "royalblue", "s": 60, "alpha": 0.8},
        line_kws={"color": "darkorange", "linewidth": 2}
    )
    plt.title("Relación entre SPR y aciertos", fontsize=13)
    plt.xlabel("SPR (Singing Power Ratio)", fontsize=12)
    plt.ylabel("Cantidad de respuestas correctas", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ No se encontró la columna 'spr', no se genera el gráfico.")

