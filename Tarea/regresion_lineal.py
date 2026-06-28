from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RUTA_CSV = Path(__file__).parent / "muestra_2000.csv"


def main():
    datos = pd.read_csv(RUTA_CSV)

    # X = variable independiente (predictora), Y = variable dependiente (respuesta)
    # Modelamos el GastoTotal en funcion de la cantidad de Gente
    df = datos[["Gente", "GastoTotal"]].dropna()
    x = df["Gente"].to_numpy()
    y = df["GastoTotal"].to_numpy()

    # Regresion lineal simple: y = b0 + b1 * x
    res = stats.linregress(x, y)
    b1 = res.slope          # pendiente
    b0 = res.intercept      # ordenada al origen
    r = res.rvalue          # coeficiente de correlacion de Pearson
    r2 = r ** 2             # coeficiente de determinacion

    print(f"n = {len(df)} observaciones")
    print(f"Recta de regresion:  GastoTotal = {b0:.4f} + {b1:.4f} * Gente")
    print(f"  b0 (ordenada al origen) = {b0:.4f}")
    print(f"  b1 (pendiente)          = {b1:.4f}")
    print(f"Coeficiente de correlacion r  = {r:.4f}")
    print(f"Coeficiente de determinacion R^2 = {r2:.4f}  ({r2*100:.2f}%)")
    print(f"p-valor (pendiente) = {res.pvalue:.4g}")

    # Grafico: nube de puntos + recta ajustada
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.4, label="Datos observados")
    x_linea = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_linea, b0 + b1 * x_linea, color="red", linewidth=2,
             label=f"y = {b0:.1f} + {b1:.1f}x   (R^2 = {r2:.3f})")
    plt.xlabel("Gente")
    plt.ylabel("GastoTotal")
    plt.title("Regresion Lineal Simple: GastoTotal vs Gente")
    plt.legend()
    plt.grid(True, alpha=0.3)
    salida = Path(__file__).parent / "regresion_gente_gasto.png"
    plt.savefig(salida, dpi=120, bbox_inches="tight")
    print(f"Grafico guardado en: {salida}")


if __name__ == "__main__":
    main()
