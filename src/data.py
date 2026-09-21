import pandas as pd
import os


def cargar_y_optimizar_datos(path_gz, path_parquet_salida):
    """
    Lee un archivo .gz y lo guarda en formato Parquet usando fastparquet
    para evitar conflictos de extensiones en PyArrow.
    """
    if os.path.exists(path_parquet_salida):
        print(f"Cargando versión optimizada Parquet desde: {path_parquet_salida}")
        return pd.read_parquet(path_parquet_salida, engine='fastparquet')

    print(f"Descomprimiendo y leyendo {path_gz} por primera vez...")
    df = pd.read_csv(path_gz, compression='gzip', low_memory=False)

    # Asegurar compatibilidad de cadenas de texto
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    print(f"Guardando copia optimizada en: {path_parquet_salida} ...")
    df.to_parquet(path_parquet_salida, index=False, engine='fastparquet')
    print("¡Archivo Parquet guardado con éxito!")

    return df