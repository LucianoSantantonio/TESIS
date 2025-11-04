# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 21:05:40 2025

@author: lucia
"""

import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, shapiro
import matplotlib.pyplot as plt

# ----------------------------------------
# 1️⃣ CARGA DE DATOS
# ----------------------------------------
# Asegurate de tener un Excel con dos columnas (sin encabezado):
# Columna 1 = "sí" o "no"
# Columna 2 = valoración (1–10)
df = pd.read_excel("IA-HUM vs calidadvocal.xlsx", header=None)
df.columns = ['creencia', 'valoracion']

print("Primeras filas del archivo:")
print(df.head(), "\n")

# Normalizamos texto (evita errores por mayúsculas o espacios)
df['creencia'] = df['creencia'].str.strip().str.lower()

# Separar grupos
grupo_si = df[df['creencia'] == 'si']['valoracion']
grupo_no = df[df['creencia'] == 'no']['valoracion']

# ----------------------------------------
# 2️⃣ TEST DE NORMALIDAD (Shapiro–Wilk)
# ----------------------------------------
print("Test de normalidad (Shapiro–Wilk):")
p_si = shapiro(grupo_si).pvalue
p_no = shapiro(grupo_no).pvalue
print(f"Grupo 'si' (cree que es IA): p = {p_si:.4f}")
print(f"Grupo 'no' (cree que es Humano): p = {p_no:.4f}\n")

# ----------------------------------------
# 3️⃣ ELECCIÓN AUTOMÁTICA DE PRUEBA
# ----------------------------------------
alpha = 0.05

if p_si > alpha and p_no > alpha:
    print("✅ Ambos grupos son normales → se usa t-test (Welch)\n")
    stat, p_val = ttest_ind(grupo_si, grupo_no, equal_var=False)
    test_name = "t de Student (Welch)"
else:
    print("⚠️ Al menos un grupo no es normal → se usa Mann–Whitney U\n")
    stat, p_val = mannwhitneyu(grupo_si, grupo_no, alternative='two-sided')
    test_name = "Mann–Whitney U"

# ----------------------------------------
# 4️⃣ RESULTADOS
# ----------------------------------------
print(f"Resultados del test {test_name}:")
print(f"Estadístico = {stat:.3f}")
print(f"p = {p_val:.4f}\n")

if p_val < alpha:
    print("✅ Se rechaza la hipótesis nula:")
    print("Existe una diferencia significativa en la valoración de la calidad vocal entre los estímulos percibidos como IA ('sí') y los percibidos como humanos ('no').")
else:
    print("❌ No se rechaza la hipótesis nula:")
    print("No hay evidencia de diferencias significativas en la valoración de la calidad vocal entre ambos grupos.")

# ----------------------------------------
# 5️⃣ GRÁFICO DE COMPARACIÓN
# ----------------------------------------
plt.figure(figsize=(6, 4))
df.boxplot(column='valoracion', by='creencia', grid=False)
plt.title('')
plt.suptitle('')
plt.xlabel('¿Crees que la canción fue generada por IA?')
plt.ylabel('Valoración de calidad vocal')
plt.xticks([1, 2], ['No', 'Sí'])
plt.tight_layout()
plt.show()
