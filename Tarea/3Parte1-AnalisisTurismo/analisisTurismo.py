from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
# Ubicación del archivo actual
carpeta_actual = Path(__file__).resolve().parent
# La muestra quedó guardada una carpeta antes que este script
ruta_muestra = carpeta_actual.parent / "muestra_2000.csv"
datos = pd.read_csv(
    ruta_muestra,
    sep=",",
    quotechar='"',
    encoding="latin1"
)
print("Columnas del archivo:")
print(datos.columns.tolist())
print("\nTamaño de la muestra:", datos.shape)
print(datos.head(3))
# Me quedo solamente con las columnas vinculadas a gastos
columnas_gasto = [col for col in datos.columns if "gasto" in col.lower()]
print("\nColumnas de gasto encontradas:")
print(columnas_gasto)
# Paso los valores a numéricos, por si alguno quedó leído como texto
for columna in columnas_gasto:
    datos[columna] = pd.to_numeric(datos[columna], errors="coerce")
print("\nResumen descriptivo de los gastos:")
print(datos[columnas_gasto].describe().round(2))
# Colores usados en los gráficos
colores = [
    "#1a2744", "#2563eb", "#0ea5e9", "#38bdf8",
    "#7dd3fc", "#93c5fd", "#bfdbfe", "#dbeafe"
]
azul_oscuro = "#1a2744"
azul = "#2563eb"
celeste_claro = "#dbeafe"
rojo = "#dc2626"
verde = "#16a34a"

# Intervalos de confianza
print("\n" + "=" * 60)
print("INTERVALOS DE CONFIANZA AL 95%")
print("=" * 60)
confianza = 0.95
alfa = 1 - confianza
intervalos = {}
for columna in columnas_gasto:
    muestra = datos[columna].dropna()
    n = len(muestra)
    media = muestra.mean()
    desvio = muestra.std(ddof=1)
    error_estandar = desvio / np.sqrt(n)
    t_critico = stats.t.ppf(1 - alfa / 2, df=n - 1)
    margen_error = t_critico * error_estandar
    limite_inferior = media - margen_error
    limite_superior = media + margen_error
    intervalos[columna] = {
        "n": n,
        "media": media,
        "desvio": desvio,
        "error_estandar": error_estandar,
        "t_critico": t_critico,
        "margen_error": margen_error,
        "li": limite_inferior,
        "ls": limite_superior
    }
    print(f"\n{columna}")
    print(f"  n = {n}")
    print(f"  media = {media:.2f}")
    print(f"  desvío = {desvio:.2f}")
    print(f"  error estándar = {error_estandar:.4f}")
    print(f"  t crítico = {t_critico:.4f}")
    print(f"  margen de error = {margen_error:.4f}")
    print(f"  IC 95% = ({limite_inferior:.2f} ; {limite_superior:.2f})")
# Gráfico de intervalos de confianza
fig, ejes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    "Intervalos de confianza al 95% para variables de gasto",
    fontsize=14,
    fontweight="bold",
    color=azul_oscuro,
    y=1.01
)
# Gráfico tipo forest plot
ax1 = ejes[0]
lista_columnas = list(intervalos.keys())
posiciones = np.arange(len(lista_columnas))
for i, columna in enumerate(lista_columnas):
    resultado = intervalos[columna]
    ax1.errorbar(
        resultado["media"],
        i,
        xerr=resultado["margen_error"],
        fmt="o",
        color=colores[i % len(colores)],
        markersize=8,
        capsize=6,
        linewidth=2
    )
    ax1.text(
        resultado["ls"] + 2,
        i,
        f"({resultado['li']:.1f} ; {resultado['ls']:.1f})",
        va="center",
        fontsize=8.5,
        color="#374151"
    )
ax1.set_yticks(posiciones)
ax1.set_yticklabels(
    [col.replace("Gasto", "Gasto\n") for col in lista_columnas],
    fontsize=9
)
ax1.set_xlabel("USD")
ax1.set_title("Medias e intervalos de confianza", fontweight="bold", color=azul_oscuro)
ax1.grid(axis="x", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
# Gráfico de barras
ax2 = ejes[1]
medias = [intervalos[col]["media"] for col in lista_columnas]
errores = [intervalos[col]["margen_error"] for col in lista_columnas]
etiquetas = [col.replace("Gasto_", "").replace("Gasto", "") for col in lista_columnas]
barras = ax2.bar(
    etiquetas,
    medias,
    color=colores[:len(lista_columnas)],
    edgecolor="white",
    linewidth=1.5,
    alpha=0.85
)
ax2.errorbar(
    etiquetas,
    medias,
    yerr=errores,
    fmt="none",
    color=azul_oscuro,
    capsize=6,
    linewidth=2
)
for barra, media, error in zip(barras, medias, errores):
    ax2.text(
        barra.get_x() + barra.get_width() / 2,
        media + error + max(medias) * 0.01,
        f"{media:.1f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=azul_oscuro
    )
ax2.set_ylabel("Gasto promedio")
ax2.set_title("Promedios con margen de error", fontweight="bold", color=azul_oscuro)
ax2.tick_params(axis="x", rotation=30)
ax2.grid(axis="y", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(carpeta_actual / "ic_gastos_Parte1.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigura guardada: ic_gastos_Parte1.png")

# Test de hipótesis para gasto en alojamiento

print("\n" + "=" * 60)
print("TEST DE HIPÓTESIS: GASTO EN ALOJAMIENTO")
print("=" * 60)
columna_alojamiento = [col for col in columnas_gasto if "aloj" in col.lower()]
if len(columna_alojamiento) == 0:
    columna_alojamiento = columnas_gasto[0]
else:
    columna_alojamiento = columna_alojamiento[0]
print(f"\nVariable utilizada: {columna_alojamiento}")
gasto_alojamiento = datos[columna_alojamiento].dropna()
n = len(gasto_alojamiento)
media = gasto_alojamiento.mean()
desvio = gasto_alojamiento.std(ddof=1)
mu_0 = 350
alfa_test = 0.05
# H0: la media es mayor o igual a 350
# H1: la media es menor a 350
t_observado, p_bilateral = stats.ttest_1samp(gasto_alojamiento, mu_0)
p_valor = p_bilateral / 2
if t_observado > 0:
    p_valor = 1 - (p_bilateral / 2)
t_critico = stats.t.ppf(alfa_test, df=n - 1)
print("\nHipótesis:")
print("  H0: μ ≥ 350")
print("  H1: μ < 350")
print(f"  alfa = {alfa_test}")
print(f"\nDatos de la muestra:")
print(f"  n = {n}")
print(f"  media = {media:.4f}")
print(f"  desvío = {desvio:.4f}")
print(f"  valor de referencia = {mu_0}")
print(f"\nResultados del test:")
print(f"  t observado = {t_observado:.4f}")
print(f"  t crítico = {t_critico:.4f}")
print(f"  p-valor = {p_valor:.6f}")
print(f"  ¿t cae en región crítica? {t_observado < t_critico}")
print(f"  ¿p-valor < alfa? {p_valor < alfa_test}")
if p_valor < alfa_test:
    decision = "Se rechaza H0"
    conclusion = (
        "Con un nivel de significación del 5%, hay evidencia suficiente "
        "para afirmar que el gasto promedio en alojamiento es menor a 350 USD."
    )
else:
    decision = "No se rechaza H0"
    conclusion = (
        "Con un nivel de significación del 5%, no hay evidencia suficiente "
        "para afirmar que el gasto promedio en alojamiento sea menor a 350 USD."
    )
print(f"\nDecisión: {decision}")
print(f"Conclusión: {conclusion}")
# Gráfico del test de hipótesis
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Test de hipótesis para el gasto en alojamiento",
    fontsize=13,
    fontweight="bold",
    color=azul_oscuro
)
# Distribución t
ax1 = ejes[0]
grados_libertad = n - 1
x = np.linspace(
    stats.t.ppf(0.0001, grados_libertad),
    stats.t.ppf(0.9999, grados_libertad),
    500
)
y = stats.t.pdf(x, grados_libertad)
ax1.plot(x, y, color=azul_oscuro, linewidth=2)
ax1.fill_between(
    x,
    y,
    where=(x < t_critico),
    color=rojo,
    alpha=0.35,
    label=f"Región crítica: T < {t_critico:.3f}"
)
ax1.fill_between(
    x,
    y,
    where=(x >= t_critico),
    color=celeste_claro,
    alpha=0.6
)
ax1.axvline(
    t_observado,
    color=azul,
    linewidth=2.5,
    label=f"T observado = {t_observado:.3f}"
)
ax1.axvline(
    t_critico,
    color=rojo,
    linestyle="--",
    linewidth=1.8,
    label=f"T crítico = {t_critico:.3f}"
)
ax1.set_xlabel("Estadístico t")
ax1.set_ylabel("Densidad")
ax1.set_title("Distribución t bajo H0", fontweight="bold", color=azul_oscuro)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
# Histograma del gasto en alojamiento
ax2 = ejes[1]
ax2.hist(
    gasto_alojamiento,
    bins=40,
    color=azul,
    alpha=0.6,
    edgecolor="white",
    density=True,
    label="Datos de la muestra"
)
ax2.axvline(
    media,
    color=azul_oscuro,
    linewidth=2.5,
    label=f"media = {media:.1f}"
)
ax2.axvline(
    mu_0,
    color=rojo,
    linestyle="--",
    linewidth=2.5,
    label=f"μ0 = {mu_0}"
)
ic_alojamiento = intervalos.get(columna_alojamiento)
if ic_alojamiento:
    ax2.axvspan(
        ic_alojamiento["li"],
        ic_alojamiento["ls"],
        color=verde,
        alpha=0.15,
        label=f"IC 95%: ({ic_alojamiento['li']:.1f}; {ic_alojamiento['ls']:.1f})"
    )
texto_resultado = "Se rechaza H0" if p_valor < alfa_test else "No se rechaza H0"
color_texto = verde if p_valor < alfa_test else rojo
color_fondo = "#f0fdf4" if p_valor < alfa_test else "#fef2f2"
ax2.text(
    0.97,
    0.97,
    f"p-valor = {p_valor:.4f}\n{texto_resultado}",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    fontweight="bold",
    color=color_texto,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor=color_fondo,
        edgecolor=color_texto,
        alpha=0.9
    )
)
ax2.set_xlabel("Gasto en alojamiento")
ax2.set_ylabel("Densidad")
ax2.set_title("Distribución de la variable analizada", fontweight="bold", color=azul_oscuro)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(
    carpeta_actual / "test_hipotesis_alojamiento_Parte2.png",
    dpi=150,
    bbox_inches="tight"
)
plt.close()
print("\nFigura guardada: test_hipotesis_alojamiento_Parte2.png")
# ------------------------------------------------------------
# Resumen final
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print("\nIntervalos de confianza:")
for columna, resultado in intervalos.items():
    print(
        f"  {columna:35s}: "
        f"({resultado['li']:.2f} ; {resultado['ls']:.2f}) "
        f"| media = {resultado['media']:.2f}"
    )
print(f"\nTest de hipótesis sobre {columna_alojamiento}:")
print("  H0: μ ≥ 350")
print("  H1: μ < 350")
print(f"  t = {t_observado:.4f}")
print(f"  p-valor = {p_valor:.6f}")
print(f"  alfa = {alfa_test}")
print(f"  decisión = {decision}")