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

def conseguirTablaFrecuenciaDestino():
    tablaFrecuencia = (
        muestra
        .groupby("Destino")
        .agg(frequency=("Destino", "count"))
        .sort_values(by='frequency', ascending=True)
    )

    tablaFrecuencia["frecuencia_absoluta_acumulada"] = (
        tablaFrecuencia["frequency"].cumsum()
    )

    tablaFrecuencia["frecuencia_relativa"] = (
        tablaFrecuencia["frequency"] / tablaFrecuencia["frequency"].sum()
    )

    tablaFrecuencia["frecuencia_relativa_acumulada"] = (
        tablaFrecuencia["frecuencia_relativa"].cumsum()
    )

    tablaFrecuencia["porcentajes"] = (
        tablaFrecuencia["frecuencia_relativa"] * 100
    )

    return tablaFrecuencia

def formato(pct):
    return f'{pct:.1f}%' if pct >= 3 else ''

def graficarDestinoBarra():
    plt.style.use('_mpl-gallery')

    destinos = tablaFrecuencia.index
    frecuencias = tablaFrecuencia["frequency"]

    fig, ax = plt.subplots(figsize=(10, 5))

    barras = ax.bar(destinos, frecuencias)
    ax.bar_label(barras, fmt='%d', padding=3)

    ax.set_title("Frecuencia absoluta por destino")
    ax.set_xlabel("Destino")
    ax.set_ylabel("Frecuencia")

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def graficarDestinoPie():
    plt.style.use('_mpl-gallery-nogrid')

    destinos = tablaFrecuencia.index
    porcentajes = tablaFrecuencia["porcentajes"]

    colors = plt.get_cmap('Blues')(np.linspace(0.3, 0.8, len(destinos)))

    fig, ax = plt.subplots(figsize=(12, 10))

    wedges, texts, autotexts = ax.pie(
        porcentajes,
        colors=colors,
        autopct=formato,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )

    plt.setp(autotexts, color="white", weight="bold")

    ax.set_title("Porcentaje por destino")

    ax.legend(
        wedges,
        destinos,
        title="Destino",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.tight_layout()
    plt.show()

tablaFrecuencia = conseguirTablaFrecuenciaDestino()

# print(tablaFrecuencia)
# graficarDestinoBarra()
# graficarDestinoPie()