# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 03:09:28 2025

@author: lucia
"""

import matplotlib.pyplot as plt
import numpy as np

# Ejemplo de datos (10 estímulos)
estimulos = [f"Estímulo {i}" for i in range(1, 11)]
aciertos = [109, 149, 119, 122, 186, 106, 95, 142, 171, 129]      # Cantidad de aciertos
desaciertos = [94, 54, 84, 81, 17, 97, 108, 61, 32, 74]      # Cantidad de desaciertos

# Crear posiciones en el eje X
x = np.arange(len(estimulos))  # posiciones de 0 a 9
ancho = 0.35                   # ancho de las barras

# Crear figura
plt.figure(figsize=(12,6))

# Graficar barras
plt.bar(x - ancho/2, aciertos, width=ancho, color="blue", label="Aciertos")
plt.bar(x + ancho/2, desaciertos, width=ancho, color="red", label="Desaciertos")

# Configurar ejes
plt.xticks(x, estimulos, rotation=45)
plt.ylabel("Cantidad")
plt.title("Cantidad de Aciertos y Desaciertos por Estímulo")
plt.legend()

# Mostrar
plt.tight_layout()
plt.show()
