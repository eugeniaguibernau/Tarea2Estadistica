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

    gasto_alimentacion = [
        c for c in muestra.columns
        if "gastoalimentacion" in c.lower()
    ]

    gasto_generico = [
        c for c in muestra.columns
        if "gastocompra" in c.lower()
    ]

    for col in gasto_alimentacion:
        muestra[col] = pd.to_numeric(muestra[col], errors="coerce")

    for col in gasto_generico:
        muestra[col] = pd.to_numeric(muestra[col], errors="coerce")

    datos = pd.DataFrame({
        "alimentacion": muestra[gasto_alimentacion[0]],
        "compras": muestra[gasto_generico[0]]
    }).dropna()

    return datos

datosJuntos = prepararDatos()

def mostrarEstadisticos():

    print("\n/// Estadísticos descriptivos de alimentación ///")
    print(datosJuntos["alimentacion"].describe().round(2))

    print("\n/// Estadísticos descriptivos de compras ///")
    print(datosJuntos["compras"].describe().round(2))

mostrarEstadisticos()

def realizarTest(alpha=0.05):

    resultado = stats.ttest_rel(
        datosJuntos["alimentacion"],
        datosJuntos["compras"]
    )

    if resultado.statistic > 0:
        p_valor = resultado.pvalue / 2
    else:
        p_valor = 1 - resultado.pvalue / 2

    return resultado, p_valor

resultado, p_valor = realizarTest(alpha)

def mostrarResultado(resultado, p_valor, alpha=0.05):

    print("\n// Test de hipótesis //")
    print(f"Estadístico t = {resultado.statistic:.4f}")
    print(f"p-valor = {p_valor:.6f}")

    if p_valor < alpha:
        print("\nSe rechaza H₀.")
        print("Existe evidencia suficiente para afirmar que")
        print("los uruguayos gastan más en alimentación.")
    else:
        print("\nNo se rechaza H₀.")
        print("No existe evidencia suficiente para afirmar")
        print("que los uruguayos gasten más en alimentación.")

mostrarResultado(resultado, p_valor, alpha)

def graficarComparacionGastos():

    plt.style.use('_mpl-gallery')

    medias = [
        datosJuntos["alimentacion"].mean(),
        datosJuntos["compras"].mean()
    ]

    nombres = [
        "Alimentación",
        "Compras\nGenéricas"
    ]

    fig, ax = plt.subplots(figsize=(8,5))

    barras = ax.bar(
        nombres,
        medias,
        color=["steelblue", "lightskyblue"]
    )

    ax.bar_label(barras, fmt="%.1f", padding=3)

    ax.set_title("Gasto promedio por categoría")
    ax.set_ylabel("USD")

    texto = (
        f"t = {resultado.statistic:.2f}\n"
        f"p = {p_valor:.4f}"
    )

    ax.text(
        0.98,
        0.95,
        texto,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black")
    )

    plt.tight_layout()
    plt.show()

def graficarTestHipotesis():

    plt.style.use('_mpl-gallery')

    gl = len(datosJuntos) - 1
    t_critico = stats.t.ppf(1 - alpha, gl)

    x = np.linspace(-4, 4, 500)
    y = stats.t.pdf(x, gl)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(x, y, color="steelblue")

    ax.fill_between(
        x,
        y,
        where=(x >= t_critico),
        color="red",
        alpha=0.3,
        label="Región crítica"
    )

    ax.axvline(
        resultado.statistic,
        color="green",
        linewidth=2,
        label=f"T observado = {resultado.statistic:.2f}"
    )

    ax.axvline(
        t_critico,
        color="red",
        linestyle="--",
        label=f"T crítico = {t_critico:.2f}"
    )

    ax.set_title("Distribución t bajo H₀")
    ax.set_xlabel("t")
    ax.set_ylabel("Densidad")

    ax.legend()

    plt.tight_layout()
    plt.show()

graficarTestHipotesis()
graficarComparacionGastos()