# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 19:49:12 2025

@author: Luciano
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- INICIO DE NUEVOS DATOS PROCESADOS ---
# Datos procesados de las cadenas de texto, reconociendo '10' como un solo número.
# Versión final 100% corregida.

raw_data = {
    '1': "874689726878776775785510877576785245855846858105945848464510797788107897774673767869876687917985542559475565742683975386945788971077387574966748886338578573588574835107737783438647978275745577481076877245798843",
    '2': "8512345167833926584573945537757147396377657871674826102637574798837699863610964679989464101678635554885245573247386288795643498729374664753638844784658567557694656387496834476789597677448687101023332565555857",
    '3': "665877439555877105966879528276671854772676731051766618165589849789778987657747677967781067187845768692755675557669846668566268648755838810166810677189867577510558255577767267545439957466553768481023422243778644",
    '4': "654674458578785867458796672559727368534768581087647177528848478779688694888465868536386711068468425738836365788587357673572685568488778468551051055247856575954536359773848841773595785444675687867675573378656",
    '5': "85115561542244242227435324254251585851316674138214154222446266663535534251264468446364413882121137641223621261551284524414547271345632225365551321446513233855158763556441432595865583657825444644143145744",
    '6': "7442876366467841044448485482748713447587865884583261872417566789865727563819674510646584713877410686774155363557777155577566510592719676647442552554868857131073567662862747952683895968563476875815862633344766",
    '7': "64458576346448473767779727264861654654696788257377167437107856889567257477636547948659563368548365625536344548676175575667458177187767847491056566668767731043557338877747431571793966563765687875677673435644",
    '8': "65488566468108761079847896109274873104487368475910577872973371059577995789574881077368104689979449775866676682667658788733667666971071087177768846583597148478457310465483786776381032853695866653665657977767275597734",
    '9': "83124881875114141436347276144462342761496544117334177221425447772437744161474479737164414745310317683331253524465177354541865127243474210163152353316766131634484527827572614126939566627778171077442135165834",
    '10': "7324687268893651034578510498257773108178373665961938538105626275799737478448848747810977665414776310765786723265557876166796651559878154678103156669665928757731074838452674947231782693878774677898943537454565758"
}

def parse_string_data(s):
    """
    Analiza una cadena de dígitos, tratando '10' como un solo número
    y todos los demás dígitos como números individuales.
    """
    data_list = []
    i = 0
    while i < len(s):
        # Comprueba si es '10'
        if s[i] == '1' and i + 1 < len(s) and s[i+1] == '0':
            data_list.append(10)
            i += 2  # Salta ambos '1' y '0'
        # Maneja los otros dígitos
        else:
            data_list.append(int(s[i]))
            i += 1
    return tuple(data_list) # Convierte a tupla para mantener el formato original

# Crea el nuevo diccionario 'respuestas'
respuestas = {}
for key in raw_data:
    respuestas[key] = parse_string_data(raw_data[key])

# --- FIN DE NUEVOS DATOS PROCESADOS ---


# --- INICIO DE CÓDIGO DE ANÁLISIS Y GRÁFICO ---

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

# Definir los estímulos para cada categoría (Humano e IA)
estimulos_humano = ['1', '3', '4', '8']
estimulos_ia = ['2', '5', '6', '7', '9', '10']

# Crear colores diferenciados para Humano e IA
colores = []
for etiqueta in etiquetas_x:
    if etiqueta in estimulos_humano:
        colores.append('lightgreen')
    elif etiqueta in estimulos_ia:
        colores.append('lightblue')
    else:
        colores.append('gray') 

# --- Gráfico ---
x = np.arange(len(etiquetas_x))
plt.figure(figsize=(14, 9))

bars = plt.bar(x, medias, yerr=error_ic95, capsize=5,
               color=colores, edgecolor='black', alpha=0.7)

plt.xticks(x, etiquetas_x, fontsize=12)
plt.yticks(range(0,11,1), fontsize=12) 
plt.ylim(0, 10.5)
plt.xlabel("Estímulo", fontsize=14)
plt.ylabel("Promedio con IC95", fontsize=14)

# Agregar texto con los valores exactos
for i, (media, inf, sup) in enumerate(zip(medias, limites_inferiores, limites_superiores)):
    plt.text(i, media + error_ic95[i] + 0.2, f'{media:.2f}', 
             ha='center', va='bottom', fontsize=15, fontweight='bold')
    plt.text(i, 0.2, f'[{inf:.2f}, {sup:.2f}]', 
             ha='center', va='bottom', fontsize=15, rotation=90) 

# Leyenda
legend_elements = [
    Patch(facecolor='lightgreen', edgecolor='black', alpha=0.7, label='Humano'),
    Patch(facecolor='lightblue', edgecolor='black', alpha=0.7, label='IA')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=12) 

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Guardar la imagen
plt.savefig("grafico_nuevos_datos_FINAL.png")

print("Gráfico con NUEVOS DATOS (AHORA SI) guardado como 'grafico_nuevos_datos_FINAL.png'.")