import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
muestra = pd.read_csv(
    BASE_DIR.parent / "muestra_2000.csv",
    sep=",",
    quotechar='"',
    encoding="latin1"
)
muestra.head()
alpha = 0.05

def prepararDatos():

    estadia = [
        c for c in muestra.columns
        if "estadia" in c.lower()
    ]

    lugar_salida = [
        c for c in muestra.columns
        if "lugar salida" in c.lower()
    ]

    datos = pd.DataFrame({
        "estadia": muestra[estadia[0]],
        "lugar_salida": muestra[lugar_salida[0]]
    }).dropna()

    return datos

datosJuntos = prepararDatos()

def mostrarEstadisticos():

    print("\n/// Frecuencias de Estadía ///")
    frecuencias = datosJuntos["estadia"].value_counts()
    for valor, cantidad in frecuencias.items():
        print(f"{valor}: {cantidad}")

    print("\n/// Frecuencias de Lugar de Salida ///")
    frecuencias_lugar = datosJuntos["lugar_salida"].value_counts()
    for valor, cantidad in frecuencias_lugar.items():
        print(f"{valor}: {cantidad}")

mostrarEstadisticos()

def crearTablaContingencia():

    tabla = pd.crosstab(
        datosJuntos["estadia"],
        datosJuntos["lugar_salida"]
    )

    return tabla

tabla = crearTablaContingencia()

def realizarTest():

    chi2, p_valor, gl, esperados = stats.chi2_contingency(tabla)

    return chi2, p_valor, gl, esperados

chi2, p_valor, gl, esperados = realizarTest()

def mostrarResultado(alpha=0.05):

    print("\n// Test Chi-Cuadrado //")
    print(f"Chi² = {chi2:.4f}")
    print(f"Grados de libertad = {gl}")
    print(f"p-valor = {p_valor:.6f}")

    if p_valor < alpha:
        print("\nSe rechaza H₀.")
        print("Existe evidencia suficiente para afirmar")
        print("que Estadía y LugarSalida están asociadas.")
    else:
        print("\nNo se rechaza H₀.")
        print("No existe evidencia suficiente para afirmar")
        print("que exista asociación entre ambas variables.")

mostrarResultado(alpha)

def graficarTabla():

    tabla.plot(
        kind="bar",
        stacked=True,
        figsize=(9,5)
    )

    plt.title("Lugar de salida según la estadía")
    plt.xlabel("Estadía")
    plt.ylabel("Cantidad de personas")

    texto = (
        f"Chi² = {chi2:.2f}\n"
        f"p = {p_valor:.4f}"
    )

    plt.text(
        0.98,
        0.95,
        texto,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white")
    )

    plt.tight_layout()
    plt.show()

def graficarTestHipotesis():

    x = np.linspace(0, chi2 + 10, 500)
    y = stats.chi2.pdf(x, gl)

    chi_critico = stats.chi2.ppf(1 - alpha, gl)

    plt.figure(figsize=(8,5))

    plt.plot(x, y)

    plt.fill_between(
        x,
        y,
        where=(x >= chi_critico),
        alpha=0.3,
        color="red",
        label="Región crítica"
    )

    plt.axvline(
        chi2,
        color="green",
        linewidth=2,
        label=f"Chi² observado = {chi2:.2f}"
    )

    plt.axvline(
        chi_critico,
        color="red",
        linestyle="--",
        label=f"Chi² crítico = {chi_critico:.2f}"
    )

    plt.title("Distribución Chi-Cuadrado bajo H₀")
    plt.xlabel("Chi²")
    plt.ylabel("Densidad")
    plt.legend()

    plt.tight_layout()
    plt.show()

graficarTestHipotesis()
graficarTabla()