import numpy as np
import pandas as pd

SEMILLA = 2026

def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv)

datos_completos = cargar_datos("/Users/eugeniaguibernau/Desktop/Tarea2Estadistica/Tarea/emisivo.csv")

muestra_2000 = datos_completos.sample(
    n=2000,
    replace=False,
    random_state=SEMILLA
).reset_index(drop=True)

muestra_2000.to_csv("muestra_2000.csv", index=False)

def main():
    datos = cargar_datos("emisivo.csv")

    print(type(datos))
    print(datos.shape)