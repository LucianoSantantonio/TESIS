# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 18:04:27 2025

@author: Luciano
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Simulación de datos: listas de respuestas por estímulo
# ¡DATOS ORIGINALES Y CORREGIDOS, SIN ERRORES!
respuestas = {
    '1': (6,6,4,8,8,8,6,5,8,7,6,7,6,6,4,4,7,6,7,8,4,6,8,8,8,5,7,5,9,8,5,8,7,5,6,4,4,7,4,6,8,6,6,10,5,9,5,7,8,5,8,6,6,8,8,4,9,6,7,9,9,10,9,6,8,8,7,8,4,7,8,4,7,7,5,8,7,7,6,8,8,8,7,7,1,7,8,8,5,3,3,3,6,6,7,6,5,7,6,5,5,6,4,2,7,8,4,10,6,2,3,7,7,6,7,7,7,6,7,5,3,8,6,3,6,3,7,5,7,7,4,5,7,3,6,1,5,7,4,7,6,5,7,3,1,8,6,7,7,3,5,8,5,7,8,8),
    '2': (8,5,3,4,6,6,4,5,6,6,4,3,8,4,6,3,8,6,5,8,5,7,6,3,7,3,7,6,9,7,1,5,7,4,7,2,1,6,7,6,7,7,6,10,1,7,5,4,8,2,7,3,7,6,5,5,7,5,7,8,9,9,6,6,7,4,6,8,6,4,7,10,8,7,2,8,8,10,9,6,8,7,5,8,1,6,7,8,6,5,2,6,5,6,8,8,6,4,5,5,5,6,3,1,4,7,2,9,5,2,8,7,8,5,6,2,6,7,4,4,3,8,4,5,5,5,5,5,5,6,7,6,7,3,5,5,7,6,5,5,8,6,5,7,1,5,5,6,8,5,6,5,8,7,8,8),
    '3': (6,5,5,5,6,7,2,5,5,7,4,7,7,4,9,2,7,6,4,8,7,9,7,2,8,2,7,7,7,6,2,7,6,4,7,5,2,6,7,6,8,5,10,8,1,7,6,8,7,1,7,6,7,6,7,7,7,5,9,8,9,9,5,7,7,8,8,5,6,5,7,8,5,6,4,8,7,9,7,7,5,8,6,7,1,9,7,8,4,3,5,4,8,7,8,6,9,5,6,4,6,4,5,5,6,6,7,8,7,1,6,7,5,5,4,4,7,6,6,6,5,7,8,5,7,7,6,2,3,6,7,10,6,3,7,2,8,7,6,6,7,5,5,7,1,9,4,5,7,4,6,5,8,7,8,6),
    '4': (6,4,4,4,8,7,2,6,6,6,7,7,6,5,9,7,6,5,5,8,8,8,6,7,7,2,5,7,9,7,2,8,4,6,8,3,5,5,6,7,8,4,9,9,8,8,7,8,7,1,8,5,4,9,9,4,8,5,8,9,8,8,9,7,5,7,7,9,2,6,8,8,6,5,5,8,7,9,4,5,6,6,6,9,1,10,6,9,5,2,8,4,2,6,7,7,4,3,5,6,3,6,5,5,7,7,6,8,6,5,5,7,5,6,5,2,6,6,5,4,5,8,8,5,7,7,5,6,7,5,7,9,5,7,6,7,3,5,8,7,7,5,4,7,1,8,5,7,5,3,5,4,7,8,9,8),
    '5': (8,5,1,2,7,5,6,2,6,4,2,5,6,5,6,2,3,4,5,8,3,6,4,6,7,2,5,4,8,5,1,5,7,5,8,7,1,4,1,6,8,7,8,9,3,8,4,7,5,4,6,2,2,9,6,5,7,3,6,7,8,7,6,4,6,4,8,9,2,8,6,1,3,5,3,7,5,9,3,5,6,7,5,6,1,7,8,7,4,4,2,1,2,4,6,7,8,3,5,3,3,5,2,1,3,6,2,6,6,3,2,5,4,4,5,4,6,6,4,2,1,6,4,6,5,2,1,2,7,5,6,4,5,3,3,4,4,3,3,7,5,6,6,1,1,5,5,3,8,1,5,4,8,7,7,6),
    '6': (8,5,5,8,8,7,7,5,5,5,4,8,7,8,10,3,3,6,5,8,3,10,5,5,8,2,7,8,8,7,3,5,6,4,6,7,6,6,6,6,9,8,8,4,5,8,7,4,6,3,8,2,7,7,7,7,8,5,8,10,9,9,9,8,8,3,7,9,6,7,9,4,9,7,6,6,7,9,5,6,10,8,6,8,1,8,7,7,7,7,10,6,8,7,7,8,9,7,6,5,5,5,4,2,6,7,8,7,7,1,5,5,4,7,6,5,8,7,8,6,1,10,6,7,5,6,5,8,7,4,3,7,5,4,8,8,5,6,5,6,7,5,7,1,1,4,7,6,6,6,5,5,8,7,8,8),
    '7': (6,3,3,9,8,7,8,6,5,8,3,4,6,4,10,4,6,6,6,6,5,10,7,4,8,2,6,7,9,6,2,5,5,6,5,2,4,6,4,7,7,9,8,4,5,7,5,4,8,1,8,2,4,6,9,7,7,6,7,9,8,9,5,6,9,2,8,8,4,5,8,10,5,7,4,5,7,9,3,8,6,8,6,8,3,10,7,8,5,4,8,3,4,5,6,5,8,6,4,5,3,4,5,5,5,7,8,7,5,1,7,5,5,5,6,7,4,6,2,7,1,8,6,6,6,7,6,7,7,4,9,8,5,4,6,7,5,8,7,6,7,6,7,7,1,8,6,3,5,4,3,8,8,7,8,7),
    '8': (7, 6, 6, 4, 9, 5, 5, 7, 6, 9, 8, 8, 6, 6, 10, 5, 8, 8, 6, 7, 8, 8, 4, 10, 8, 2, 7, 7, 8, 5, 3, 8, 4, 6, 8, 7, 3, 5, 5, 4, 7, 8, 9, 10, 5, 7, 8, 8, 7, 4, 10, 3, 2, 6, 8, 6, 8, 5, 7, 9, 9, 9, 8, 6, 9, 9, 8, 8, 4, 8, 8, 10, 6, 6, 3, 6, 7, 9, 5, 6, 3, 8, 8, 9, 4, 6, 8, 7, 8, 2, 8, 7, 4, 7, 7, 6, 7, 7, 5, 7, 3, 7, 5, 5, 8, 7, 6, 8, 6, 4, 3, 7, 5, 7, 6, 8, 7, 7, 9, 8, 1, 8, 7, 5, 6, 8, 7, 3, 8, 5, 7, 1, 5, 5, 7, 5, 4, 4, 6, 3, 8, 4, 6, 7, 1, 10, 4, 7, 5, 3, 7, 3, 8, 8, 8, 6),
    '9': (8, 5, 1, 2, 7, 7, 8, 5, 6, 5, 4, 2, 7, 3, 10, 3, 5, 7, 5, 7, 8, 9, 6, 8, 7, 1, 4, 6, 8, 6, 4, 5, 7, 4, 7, 8, 1, 5, 7, 4, 8, 7, 8, 6, 1, 8, 4, 5, 4, 1, 7, 2, 2, 7, 8, 3, 7, 4, 5, 8, 8, 9, 3, 5, 6, 6, 8, 8, 4, 3, 7, 1, 6, 6, 4, 7, 6, 9, 8, 7, 5, 6, 6, 8, 3, 4, 7, 4, 5, 6, 10, 3, 1, 6, 6, 8, 7, 4, 5, 4, 3, 5, 4, 5, 7, 8, 8, 5, 6, 2, 7, 7, 5, 2, 5, 4, 8, 8, 4, 4, 2, 7, 3, 6, 6, 8, 5, 8, 3, 6, 3, 1, 6, 2, 4, 5, 5, 8, 4, 5, 6, 6, 6, 1, 1, 5, 4, 4, 5, 4, 5, 2, 9, 8, 8, 8),
    '10': (6, 3, 2, 6, 8, 8, 8, 6, 8, 8, 7, 3, 8, 4, 10, 3, 5, 7, 6, 8, 5, 10, 5, 9, 8, 2, 5, 8, 8, 7, 4, 8, 8, 3, 7, 8, 3, 6, 3, 6, 6, 8, 9, 10, 1, 8, 5, 8, 5, 3, 9, 3, 6, 6, 6, 4, 7, 7, 7, 9, 9, 9, 4, 5, 6, 8, 8, 7, 4, 6, 8, 8, 6, 6, 3, 8, 6, 10, 8, 7, 5, 7, 5, 8, 2, 4, 6, 5, 6, 7, 10, 7, 6, 7, 7, 8, 8, 6, 5, 3, 3, 6, 6, 2, 6, 8, 7, 6, 7, 2, 6, 6, 7, 8, 5, 3, 5, 8, 6, 4, 1, 6, 4, 5, 7, 8, 8, 4, 3, 5, 6, 7, 6, 4, 3, 7, 5, 8, 5, 6, 5, 5, 7, 7, 1, 8, 7, 5, 8, 4, 5, 2, 5, 8, 6, 8),
}

# Calcular medias y errores IC95
etiquetas_x = list(respuestas.keys())
medias = [np.mean(respuestas[k]) for k in etiquetas_x]
desvios = [np.std(respuestas[k], ddof=1) for k in etiquetas_x]

# Obtener el tamaño de muestra real para cada grupo
n_samples = [len(respuestas[k]) for k in etiquetas_x]

# Calcular error estándar y IC95 para cada grupo
errores_estandar = [desvios[i] / np.sqrt(n_samples[i]) for i in range(len(etiquetas_x))]
error_ic95 = [1.96 * se for se in errores_estandar]

# Calcular límites superior e inferior del IC95
limites_inferiores = [medias[i] - error_ic95[i] for i in range(len(medias))]
limites_superiores = [medias[i] + error_ic95[i] for i in range(len(medias))]

# --- INICIO DE MODIFICACIONES PARA COLORES Y LEYENDA ---
# Definir los estímulos para cada categoría (Humano e IA)
estimulos_humano = ['1', '3', '4', '8']
estimulos_ia = ['2', '5', '6', '7', '9', '10']

# Crear colores diferenciados para Humano e IA basados en la lista de estímulos
colores = []
for etiqueta in etiquetas_x:
    if etiqueta in estimulos_humano:
        colores.append('lightgreen')
    elif etiqueta in estimulos_ia:
        colores.append('lightblue')
    else:
        colores.append('gray') # Color por defecto si un estímulo no está en ninguna lista
# --- FIN DE MODIFICACIONES PARA COLORES Y LEYENDA ---


# --- Gráfico ---
x = np.arange(len(etiquetas_x))
plt.figure(figsize=(14, 9))

# Crear barras con barras de error
bars = plt.bar(x, medias, yerr=error_ic95, capsize=5,
               color=colores, edgecolor='black', alpha=0.7)

# Configurar ejes y etiquetas con fuentes más grandes
plt.xticks(x, etiquetas_x, fontsize=12)
plt.yticks(range(0,11,1), fontsize=12) # Asegura que los ticks del eje Y también sean grandes
plt.ylim(0, 10.5)
plt.xlabel("Estímulo", fontsize=14)
plt.ylabel("Promedio con IC95", fontsize=14)

# Agregar texto con los valores exactos (fuentes más grandes)
for i, (media, inf, sup) in enumerate(zip(medias, limites_inferiores, limites_superiores)):
    
    # Texto de la media (arriba de la barra de error IC95)
    plt.text(i, media + error_ic95[i] + 0.2, f'{media:.2f}', 
             ha='center', va='bottom', fontsize=15, fontweight='bold') # <-- Promedio Grande
    
    # Texto del IC95 (abajo, rotado)
    plt.text(i, 0.2, f'[{inf:.2f}, {sup:.2f}]', 
             ha='center', va='bottom', fontsize=15, rotation=90) # <-- IC95 Grande

# --- MODIFICACIÓN DE LEYENDA ---
legend_elements = [
    Patch(facecolor='lightgreen', edgecolor='black', alpha=0.7, label='Humano'),
    Patch(facecolor='lightblue', edgecolor='black', alpha=0.7, label='IA')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12) # <-- Leyenda Grande
# --- FIN MODIFICACIÓN DE LEYENDA ---


plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Guardar la imagen
plt.savefig("grafico_colores_personalizados.png")

# Esto debe ir después de savefig si quieres que la imagen se muestre brevemente y luego se guarde
# Pero para este entorno, es mejor solo guardar.
# plt.show() 

print("Gráfico guardado como 'grafico_colores_personalizados.png' con fuentes grandes y colores personalizados.")