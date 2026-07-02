import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def conseguirBins():
    minimo = math.floor(muestra["GastoTotal"].min())
    maximo = math.ceil(muestra["GastoTotal"].max())

    rango = float(maximo - minimo)

    cantidadIntervalos = math.sqrt(int(muestra["GastoTotal"].count()))

    amplitudDeIntervalo = int(rango / cantidadIntervalos)
    
    bins = np.arange(minimo, maximo + amplitudDeIntervalo, amplitudDeIntervalo)

    return bins

def conseguirTablaFrecuenciaGastoTotal():
    bins = conseguirBins()

    muestra["intervaloGastos"] = pd.cut(muestra["GastoTotal"], bins=bins)
    
    tablaFrecuencia = (
        muestra
        .groupby("intervaloGastos", observed=False) # Si se cambia el valor de observed a True, se podrá visualizar una tabla resumida con intervalos efectivos
        .agg(frequencia=("GastoTotal", "count"))
        .sort_index()
    )

    tablaFrecuencia["frecuencia_absoluta_acumulada"] = (
        tablaFrecuencia["frequencia"].cumsum()
    )

    tablaFrecuencia["frecuencia_relativa"] = (
        tablaFrecuencia["frequencia"] / tablaFrecuencia["frequencia"].sum()
    )

    tablaFrecuencia["frecuencia_relativa_acumulada"] = (
        tablaFrecuencia["frecuencia_relativa"].cumsum()
    )

    tablaFrecuencia["marcaDeClase"] = [
        (intervalo.left + intervalo.right) / 2
        for intervalo in tablaFrecuencia.index
    ]

    return tablaFrecuencia

def conseguirHistogramaGastoTotal():
    valores = muestra["GastoTotal"]
    bins = conseguirBins()

    fig, ax = plt.subplots(tight_layout=True)
    counts, bins, patches = plt.hist(valores, bins=bins, rwidth=0.8)

    plt.xticks(bins, rotation=45)
    ax.bar_label(patches, padding=3, label_type="edge")

    plt.title("Histograma de Gastos Totales")
    plt.xlabel("Límites de Intervalo")
    plt.ylabel("Frecuencia")

    plt.show()

def conseguirDiagramaDeCajaGastoTotal():
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].boxplot(muestra["GastoTotal"])
    axs[0].set_title("Con valores atípicos")
    axs[0].set_ylabel("Gasto Total")

    axs[1].boxplot(muestra["GastoTotal"], showfliers=False)
    axs[1].set_title("Sin valores atípicos")

    plt.tight_layout()
    plt.show()

def conseguirDiagramaDeDispersionGastoTotal():
    gastosTotales = muestra["GastoTotal"]
    gente = muestra["Gente"]

    plt.scatter(gastosTotales, gente, c = np.sqrt(gastosTotales ** 2 + gente ** 2), alpha=0.3)
    plt.title("Diagrama de Dispersión de Gastos Totales y Gente")
    plt.ylabel("Gente")
    plt.xlabel("Gastos")
    plt.show()

tablaFrecuencia = conseguirTablaFrecuenciaGastoTotal()
print(tablaFrecuencia)

# Descomentar para ejecutar las funciones deseadas.

# conseguirHistogramaGastoTotal()
# conseguirDiagramaDeCajaGastoTotal()
# conseguirDiagramaDeDispersionGastoTotal()