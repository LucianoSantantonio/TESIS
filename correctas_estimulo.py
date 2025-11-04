# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:13:49 2025

@author: lucia
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Configuración ---
nombre_archivo_excel = 'aciertos_estimulos.xlsx'  # Asegúrate que el nombre coincida
total_ensayos_por_estimulo = 203  # <-- ¡IMPORTANTE: Ajusta este valor!

# 1. Cargar los datos desde Excel
try:
    df = pd.read_excel(nombre_archivo_excel)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{nombre_archivo_excel}'.")
    exit()
except Exception as e:
    print(f"Error al leer el archivo Excel: {e}")
    exit()

# 2. Extraer los datos y calcular el porcentaje
try:
    # Como tus datos están en la PRIMERA fila de datos (índice 0),
    # la seleccionamos con .iloc[0]
    # Los nombres de las columnas (ej. "Estimulo 1") serán el eje X.
    aciertos = df.iloc[0]
    
    # Calculamos el porcentaje
    porcentajes = (aciertos / total_ensayos_por_estimulo) * 100
    
    # Obtenemos los nombres de los estímulos (del encabezado)
    estimulos = aciertos.index
    
except IndexError:
    print("Error: El archivo Excel parece estar vacío o no tiene la fila de datos.")
    exit()
except ZeroDivisionError:
    print("Error: El 'total_ensayos_por_estimulo' no puede ser cero.")
    exit()

# 3. Crear el gráfico de barras
plt.figure(figsize=(12, 7))  # Tamaño del gráfico

plt.bar(
    estimulos,      # Eje X: "Estimulo 1", "Estimulo 2", etc.
    porcentajes,    # Eje Y: Los porcentajes calculados
    color='deepskyblue',
    edgecolor='black'
)

# 4. Añadir títulos y etiquetas
plt.title('Porcentaje de Respuestas Correctas por Estímulo', fontsize=16)
plt.xlabel('Estímulo', fontsize=12)
plt.ylabel('Aciertos (%)', fontsize=12)

# 5. Ajustar los ejes
plt.ylim(0, 100)  # Eje Y de 0 a 100%
plt.yticks(range(0, 101, 10))
# Rotamos las etiquetas del eje X si son largas (como "Estimulo 1")
plt.xticks([1,2,3,4,5,6,7,8,9,10])

# 6. Añadir una cuadrícula horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajusta el layout para que no se corten las etiquetas
plt.tight_layout()

# 7. Guardar el gráfico como una imagen
nombre_grafico_salida = 'grafico_aciertos_estimulos.png'
plt.savefig(nombre_grafico_salida)

print(f"¡Gráfico guardado exitosamente como '{nombre_grafico_salida}'!")