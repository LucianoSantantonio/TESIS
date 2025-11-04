# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 01:02:17 2025

@author: lucia
"""

import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ----------------------------------------
# 1️⃣  CARGA DE DATOS DESDE EXCEL SIN ENCABEZADOS
# ----------------------------------------
# ⚠️ Cambiá el nombre del archivo por el tuyo exacto (debe estar en la misma carpeta)
df = pd.read_excel("trabajar en ambito relacionado a la musica.xlsx", header=None)

# Asignar nombres a las columnas
df.columns = ['trabajar en ambito relacionado a la musica', 'aciertos']

print("Primeras filas del archivo leído:")
print(df.head(), "\n")

# ----------------------------------------
# 2️⃣  SEPARAR LOS GRUPOS
# ----------------------------------------
# Normalizamos las respuestas ("sí"/"no" con o sin mayúsculas/espacios)
df['trabajar en ambito relacionado a la musica'] = df['trabajar en ambito relacionado a la musica'].astype(str).str.strip().str.lower()
df['trabajar en ambito relacionado a la musica'] = df['trabajar en ambito relacionado a la musica'].replace({
    'si': 'Sí',
    'no': 'No'
})

grupo_si = df[df['trabajar en ambito relacionado a la musica'] == 'Sí']['aciertos']
grupo_no = df[df['trabajar en ambito relacionado a la musica'] == 'No']['aciertos']

# ----------------------------------------
# 3️⃣  ESTADÍSTICAS DESCRIPTIVAS
# ----------------------------------------
print("Estadísticas descriptivas por grupo:")
print(df.groupby('trabajar en ambito relacionado a la musica')['aciertos'].describe(), "\n")

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
df.boxplot(column='aciertos', by='trabajar en ambito relacionado a la musica', grid=False)
plt.title('Distribución de aciertos')
plt.suptitle('')
plt.xlabel('¿Trabaja en un ámbito relacionado a la musica?')
plt.ylabel('Cantidad de aciertos')
plt.xticks([1, 2], ['No', 'Si'])
plt.tight_layout()
plt.show()
