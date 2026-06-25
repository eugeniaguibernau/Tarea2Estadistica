from pathlib import Path
import pandas as pd

SEMILLA = 2026

def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv)

RUTA_CSV = Path(__file__).parent / "emisivo.csv"

def main():
    datos_completos = cargar_datos(RUTA_CSV)

    print(type(datos_completos))
    print(datos_completos.shape)

    muestra_2000 = datos_completos.sample(
        n=2000,
        replace=False,
        random_state=SEMILLA
    ).reset_index(drop=True)

    ruta_salida = Path(__file__).parent / "muestra_2000.csv"
    muestra_2000.to_csv(ruta_salida, index=False)

    print(f"Archivo generado: {ruta_salida}")
    print(f"Filas de la muestra: {muestra_2000.shape[0]}")

if __name__ == "__main__":
    main()