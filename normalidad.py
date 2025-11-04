# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 18:03:32 2025

@author: Luciano
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 👉 Cargar el archivo Excel (modifica el nombre si es necesario)
archivo = "tus_respuestas.xlsx"

# 📥 Leer solo la columna L, desde la fila 2 hasta la 157
df = pd.read_excel(archivo, usecols="P", skiprows=0, nrows=156)

# Asegurarse que los datos no tengan NaNs
datos = df.iloc[:, 0].dropna().to_numpy()

# --- 1. Histograma + Curva Normal
sns.histplot(datos, kde=True, stat="density", bins=15)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(datos), np.std(datos))
plt.plot(x, p, 'r', linewidth=2)
plt.title("Histograma con curva normal")
plt.xlabel("Respuestas")
plt.ylabel("Densidad")
plt.show()

# --- 2. Prueba de Shapiro-Wilk
stat, p = stats.shapiro(datos)
print(f"Shapiro-Wilk: Estadístico={stat:.4f}, p-valor={p:.4f}")
if p > 0.05:
    print("✅ No se rechaza la normalidad (Shapiro-Wilk)")
else:
    print("❌ Se rechaza la normalidad (Shapiro-Wilk)")

# --- 3. Kolmogorov-Smirnov
stat, p = stats.kstest(stats.zscore(datos), 'norm')
print(f"Kolmogorov-Smirnov: Estadístico={stat:.4f}, p-valor={p:.4f}")

# --- 4. Gráfico Q-Q
stats.probplot(datos, dist="norm", plot=plt)
plt.title("Gráfico Q-Q")
plt.show()

# --- 5. Jarque-Bera
stat, p = stats.jarque_bera(datos)
print(f"Jarque-Bera: Estadístico={stat:.4f}, p-valor={p:.4f}")