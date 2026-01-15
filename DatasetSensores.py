import pandas as pd
import numpy as np
import os

# -------- CONFIGURACIÓN --------
archivo_csv = 'molding_machine.csv'  # nombre de tu dataset

# -------- CARGAR ARCHIVO --------
try:
    df = pd.read_csv(archivo_csv)
    print("📁 Archivo cargado correctamente: ", archivo_csv)
except FileNotFoundError:
    raise FileNotFoundError(f"No se encontró el archivo: {archivo_csv}")

# -------- MIRAR COLUMNAS --------
print("\n📊 Columnas detectadas en el dataset:")
print(df.columns.tolist())

# -------- VALIDAR COLUMNAS DE TEMPERATURA (MISMO NOMBRE QUE EN TU MODELO) --------
temp_cols = [col for col in df.columns if "R_SHTHTR" in col and "TMP" in col]

if not temp_cols:
    raise ValueError("❌ No se encontraron columnas de temperatura con el patrón 'R_SHTHTRxxTMP'")

print("\n🔎 Columnas de temperatura (sensores) detectadas y usadas en los modelos:")
for c in temp_cols:
    print("   -", c)

# -------- VISTA PREVIA DEL DATAFRAME --------
print("\n📋 Primeras 10 filas de las columnas de temperatura:")
print(df[temp_cols].head(10))

print(f"\n✅ Total de sensores de temperatura detectados: {len(temp_cols)}")

# -------- OPCIONAL: mostrar estadísticas básicas --------
print("\n📈 Estadísticas básicas de las temperaturas:")
print(df[temp_cols].describe())
