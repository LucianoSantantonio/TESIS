# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 00:53:12 2025

@author: lucia
"""

import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ----------------------------------------
# 1️⃣  CARGA DE DATOS DESDE EXCEL SIN ENCABEZADOS
# ----------------------------------------
# ⚠️ Cambiá el nombre del archivo por el tuyo exacto (debe estar en la misma carpeta)
df = pd.read_excel("experiencia_IA.xlsx", header=None)

# Asignar nombres a las columnas
df.columns = ['experiencia_IA', 'aciertos']

print("Primeras filas del archivo leído:")
print(df.head(), "\n")

# ----------------------------------------
# 2️⃣  SEPARAR LOS GRUPOS
# ----------------------------------------
# Normalizamos las respuestas ("sí"/"no" con o sin mayúsculas/espacios)
df['experiencia_IA'] = df['experiencia_IA'].astype(str).str.strip().str.lower()
df['experiencia_IA'] = df['experiencia_IA'].replace({
    'alguna': 'Alguna',
    'ninguna': 'Ninguna'
})

grupo_si = df[df['experiencia_IA'] == 'Alguna']['aciertos']
grupo_no = df[df['experiencia_IA'] == 'Ninguna']['aciertos']

# ----------------------------------------
# 3️⃣  ESTADÍSTICAS DESCRIPTIVAS
# ----------------------------------------
print("Estadísticas descriptivas por grupo:")
print(df.groupby('experiencia_IA')['aciertos'].describe(), "\n")

# ----------------------------------------
# 4️⃣  TEST T DE STUDENT (Welch)
# ----------------------------------------
t_stat, p_val = ttest_ind(grupo_si, grupo_no, equal_var=False)

print("Resultados del test t de Student (Welch):")
print(f"t = {t_stat:.3f}")
print(f"p = {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print("\n✅ Se rechaza la hipótesis nula:")
    print("Existe una diferencia significativa en la cantidad de aciertos entre los grupos.")
else:
    print("\n❌ No se rechaza la hipótesis nula:")
    print("No hay evidencia de diferencias significativas en la cantidad de aciertos entre los grupos.")

# ----------------------------------------
# 5️⃣  GRÁFICO DE COMPARACIÓN
# ----------------------------------------
plt.figure(figsize=(6, 4))
df.boxplot(column='aciertos', by='experiencia_IA', grid=False)
plt.title('Distribución de aciertos')
plt.suptitle('')
plt.xlabel('Experiencia con IA')
plt.ylabel('Cantidad de aciertos')
plt.xticks([1, 2], ['Alguna', 'Ninguna'])
plt.tight_layout()
plt.show()
