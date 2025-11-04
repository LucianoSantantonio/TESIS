# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 17:38:45 2025

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
    aciertos = pd.to_numeric(df.iloc[0], errors='coerce') # Añadido para manejar posibles no-numéricos
    
    # Calculamos el porcentaje
    porcentajes = (aciertos / total_ensayos_por_estimulo) * 100
    porcentajes = porcentajes.fillna(0) # Manejar NaNs después de to_numeric
    
    # Obtenemos los nombres de los estímulos (del encabezado)
    estimulos_str = aciertos.index.tolist() # Convertir a lista de strings
    
    # Extraer los números de los estímulos si son del tipo "Estimulo X"
    # Esto es útil si los nombres reales de tus columnas son 'Estimulo 1', 'Estimulo 2', etc.
    # Si tus encabezados son solo '1', '2', etc. esto NO es necesario y podemos usar `aciertos.index.astype(int)`
    estimulos_numericos = []
    for s in estimulos_str:
        try:
            # Intenta extraer el número. Asume que es 'Estimulo X' o 'X'
            if isinstance(s, str) and 'Estimulo' in s:
                estimulos_numericos.append(int(s.replace('Estimulo ', '')))
            else:
                estimulos_numericos.append(int(s)) # Si el encabezado es solo el número
        except ValueError:
            print(f"Advertencia: No se pudo convertir '{s}' a un número de estímulo. Se omitirá su colorización especial.")
            estimulos_numericos.append(None) # Para manejar casos donde no se puede parsear
    
    # Asegurarse de que `estimulos_numericos` tenga la misma longitud y orden que los datos.
    # Si los encabezados de tu Excel son SOLO los números (1, 2, ..., 10), simplificaremos esto:
    # estimulos_numericos = aciertos.index.astype(int).tolist() # Esto sería si los encabezados son '1', '2', etc.

except IndexError:
    print("Error: El archivo Excel parece estar vacío o no tiene la fila de datos.")
    exit()
except ZeroDivisionError:
    print("Error: El 'total_ensayos_por_estimulo' no puede ser cero.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al procesar los datos: {e}")
    exit()

# --- Definición de colores para cada barra ---
# Crear una lista de colores basada en las reglas
colores_barras = []
color_verde = 'mediumseagreen' # Puedes ajustar el tono de verde
color_celeste = 'deepskyblue' # El celeste que ya usabas

for i, estimulo_num in enumerate(estimulos_numericos):
    if estimulo_num in [1, 3, 4, 8]:
        colores_barras.append(color_verde)
    elif estimulo_num in [2, 5, 6, 7, 9, 10]:
        colores_barras.append(color_celeste)
    else:
        # Por si hay algún estímulo inesperado o no parseado
        colores_barras.append('gray') 


# 3. Crear el gráfico de barras
plt.figure(figsize=(12, 7))  # Tamaño del gráfico

plt.bar(
    estimulos_str,      # Eje X: Los nombres originales de los estímulos ("Estimulo 1", etc.)
    porcentajes,        # Eje Y: Los porcentajes calculados
    color=colores_barras, # <-- Aquí pasamos la lista de colores
    edgecolor='black'
)

# 4. Añadir títulos y etiquetas
plt.title('Porcentaje de Respuestas Correctas por Estímulo', fontsize=16)
plt.xlabel('Estímulo', fontsize=12)
plt.ylabel('Aciertos (%)', fontsize=12)

# 5. Ajustar los ejes
plt.ylim(0, 100)  # Eje Y de 0 a 100%
plt.yticks(range(0, 101, 10)) # Marcas del eje Y de 10 en 10

# Si tus etiquetas X son "Estimulo 1", "Estimulo 2", etc., puede que necesites rotarlas
# para que no se superpongan si tienes muchos estímulos.
# Si tus etiquetas son solo los números 1 a 10, puedes usar:
# plt.xticks(range(1, 11))
# O si quieres los nombres completos en el eje X:
plt.xticks([1,2,3,4,5,6,7,8,9,10])


# 6. Añadir una cuadrícula horizontal
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Ajusta el layout para que no se corten las etiquetas
plt.tight_layout()

# 7. Guardar el gráfico como una imagen
nombre_grafico_salida = 'grafico_aciertos_estimulos_colores.png' # Cambié el nombre para no sobrescribir
plt.savefig(nombre_grafico_salida)

print(f"¡Gráfico guardado exitosamente como '{nombre_grafico_salida}'!")