# ===============================================
# ANÁLISIS DE VARIANZA (ANOVA) ENTRE GRUPOS ETARIOS
# ===============================================

import pandas as pd
from scipy.stats import f_oneway, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

# ----------------------------------------
# 1️⃣  CARGA DE DATOS
# ----------------------------------------
df = pd.read_excel("grupo_etario.xlsx", header=None)
df.columns = ['grupo_generacional', 'aciertos']

# Normalización de texto
df['grupo_generacional'] = df['grupo_generacional'].str.strip().str.lower()

mapeo_grupos = {
    'generación x': 'Generación X', 'generacion x': 'Generación X', 'gen x': 'Generación X',
    'generación z': 'Generación Z', 'generacion z': 'Generación Z', 'gen z': 'Generación Z',
    'millenials': 'Millenials', 'millennials': 'Millenials', 'millennial': 'Millenials',
    'baby boomers': 'Baby Boomers', 'baby boomer': 'Baby Boomers', 'boomer': 'Baby Boomers'
}

df['grupo_generacional'] = df['grupo_generacional'].map(mapeo_grupos).fillna(df['grupo_generacional'])
grupos_validos = ['Generación Z', 'Millenials', 'Generación X', 'Baby Boomers']
df_filtrado = df[df['grupo_generacional'].isin(grupos_validos)]

# ----------------------------------------
# 2️⃣  ESTADÍSTICAS DESCRIPTIVAS
# ----------------------------------------
print("\n📊 Estadísticas descriptivas por grupo generacional:")
print(df_filtrado.groupby('grupo_generacional')['aciertos'].describe(), "\n")

# ----------------------------------------
# 3️⃣  PRUEBAS DE SUPUESTOS
# ----------------------------------------
print("🧩 PRUEBAS DE SUPUESTOS DEL ANOVA:")

# Normalidad (Shapiro-Wilk)
print("\n🔹 Test de normalidad (Shapiro-Wilk):")
p_shapiro = {}
for grupo in grupos_validos:
    datos = df_filtrado[df_filtrado['grupo_generacional'] == grupo]['aciertos']
    if len(datos) >= 3:  # Shapiro requiere al menos 3 observaciones
        stat, p = shapiro(datos)
        p_shapiro[grupo] = p
        print(f"  {grupo:15s} → p = {p:.4f}")
    else:
        print(f"  {grupo:15s} → No se pudo evaluar (n < 3)")
        p_shapiro[grupo] = None

# Evaluación global de normalidad
valores_validos = [p for p in p_shapiro.values() if p is not None]
if valores_validos:
    normalidad_cumple = sum(p > 0.05 for p in valores_validos) / len(valores_validos)
    if normalidad_cumple >= 0.75:
        interpretacion_normalidad = "✅ La mayoría de los grupos cumple el supuesto de normalidad."
    else:
        interpretacion_normalidad = "⚠️ Algunos grupos no cumplen con la normalidad (puede afectar la validez del ANOVA)."
else:
    interpretacion_normalidad = "⚠️ No se pudo evaluar adecuadamente la normalidad en todos los grupos."
print("\n" + interpretacion_normalidad)

# Homogeneidad de varianzas (Levene)
print("\n🔹 Test de homogeneidad de varianzas (Levene):")
grupos = [df_filtrado[df_filtrado['grupo_generacional'] == g]['aciertos'] for g in grupos_validos]
stat_levene, p_levene = levene(*grupos)
print(f"  Levene W = {stat_levene:.3f}, p = {p_levene:.4f}")
if p_levene > 0.05:
    interpretacion_levene = "✅ Se cumple el supuesto de homogeneidad de varianzas."
else:
    interpretacion_levene = "⚠️ No se cumple el supuesto de homogeneidad de varianzas."
print(interpretacion_levene)

from scipy.stats import kruskal
import scikit_posthocs as sp

# ----------------------------------------
# 4️⃣ TEST DE KRUSKAL–WALLIS (no paramétrico)
# ----------------------------------------
print("="*65)
print("TEST DE KRUSKAL–WALLIS (no paramétrico)")
print("="*65)

# Agrupamos los valores por grupo
grupos = [grupo["aciertos"].values for nombre, grupo in df_filtrado.groupby("grupo_generacional")]

# Kruskal–Wallis
H, p_kw = kruskal(*grupos)
print(f"H = {H:.3f}")
print(f"p = {p_kw:.4f}")

if p_kw < 0.05:
    print("\n✅ Se rechaza la hipótesis nula:")
    print("Existen diferencias significativas entre al menos dos grupos.\n")
else:
    print("\n❌ No se rechaza la hipótesis nula:")
    print("No hay evidencia de diferencias significativas entre los grupos.\n")

# ----------------------------------------
# 5️⃣ POST-HOC DE DUNN (si Kruskal fue significativo)
# ----------------------------------------
if p_kw < 0.05:
    print("="*65)
    print("PRUEBA POST-HOC DE DUNN (con corrección Bonferroni)")
    print("="*65)

    dunn = sp.posthoc_dunn(df_filtrado, val_col='aciertos', group_col='grupo_generacional', p_adjust='bonferroni')
    print(dunn, "\n")

    # Mostrar comparaciones significativas
    print("📊 RESUMEN DE DIFERENCIAS ENTRE GRUPOS (Dunn):\n")
    for i in dunn.index:
        for j in dunn.columns:
            if i != j and dunn.loc[i, j] < 0.05:
                print(f"✅ {i} vs {j} → p = {dunn.loc[i, j]:.4f}")
else:
    print("No se realiza prueba post-hoc, ya que Kruskal–Wallis no fue significativo.")

# ----------------------------------------
# 6️⃣  VISUALIZACIÓN: BOXPLOT
# ----------------------------------------
plt.figure(figsize=(9, 6))
df_filtrado.boxplot(column='aciertos', by='grupo_generacional', grid=False)
plt.title('Distribución de aciertos por grupo generacional')
plt.suptitle('')
plt.xlabel('Grupo Generacional')
plt.ylabel('Cantidad de aciertos (0–10)')
plt.tight_layout()
plt.show()

# ----------------------------------------
# 7️⃣  VISUALIZACIÓN: BARRAS CON ERROR ESTÁNDAR
# ----------------------------------------
plt.figure(figsize=(9, 6))
estadisticas = df_filtrado.groupby('grupo_generacional')['aciertos'].agg(['mean', 'std', 'count'])
estadisticas['se'] = estadisticas['std'] / (estadisticas['count'] ** 0.5)
estadisticas = estadisticas.reindex(grupos_validos)

barras = plt.bar(estadisticas.index, estadisticas['mean'],
                 yerr=estadisticas['se'], capsize=8,
                 color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                 edgecolor='black', linewidth=0.5)

plt.title('Media de aciertos por grupo generacional (± error estándar)', fontsize=13, fontweight='bold')
plt.xlabel('Grupo Generacional', fontweight='bold')
plt.ylabel('Media de aciertos', fontweight='bold')
plt.ylim(0, 10.5)
for i, barra in enumerate(barras):
    height = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., height + 0.1, f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------
# 8️⃣  INTERPRETACIÓN FINAL (para informe)
# ----------------------------------------
print("\n🧾 INTERPRETACIÓN (para incluir en el informe):")

if p_val < alpha:
    interpretacion = (
        f"El análisis de varianza (ANOVA de una vía) mostró diferencias estadísticamente significativas "
        f"entre los grupos etarios en la cantidad de respuestas correctas (F = {f_stat:.3f}, p = {p_val:.4f}). "
        f"Esto indica que la pertenencia generacional influye de manera significativa en el desempeño. "
        f"{interpretacion_normalidad} {interpretacion_levene} "
        f"Las comparaciones post-hoc mediante la prueba de Tukey permiten identificar entre qué grupos "
        f"se manifiestan esas diferencias."
    )
else:
    interpretacion = (
        f"El análisis de varianza (ANOVA de una vía) no mostró diferencias estadísticamente significativas "
        f"en la cantidad de aciertos entre los distintos grupos etarios (F = {f_stat:.3f}, p = {p_val:.4f}). "
        f"Esto sugiere que, dentro de la muestra analizada, la edad o generación de pertenencia no influyó "
        f"de forma significativa en el rendimiento. "
        f"{interpretacion_normalidad} {interpretacion_levene}"
    )

print("\n" + interpretacion)

