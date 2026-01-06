import pandas as pd
import numpy as np
import os

# -------- CONFIGURACIÓN. --------
archivo_csv = 'Dataset_Termopar_Tipo_k.csv'  # <-- pon aquí el nombre real del archivo
columna_referencia = 'Tref(°C)'
sensores = [
    'Tsensor1_K(°C)',
    'Tsensor2_K(°C)',
    'Tsensor3_K(°C)'
]

# -------- CARGAR ARCHIVO --------
try:
    df = pd.read_csv(archivo_csv)
    print("Archivo cargado correctamente ✅")
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el archivo: {archivo_csv}")

# -------- VER COLUMNAS --------
print("\nColumnas detectadas:")
print(df.columns.tolist())

# -------- VALIDAR COLUMNAS --------
columnas_necesarias = [columna_referencia] + sensores
faltantes = [c for c in columnas_necesarias if c not in df.columns]

if faltantes:
    raise ValueError(f"Faltan columnas en el archivo: {faltantes}")

# -------- VISTA PREVIA --------
print("\nPrimeras 10 filas del DataFrame:")
print(df.head(10))