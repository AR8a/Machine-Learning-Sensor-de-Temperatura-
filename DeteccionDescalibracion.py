import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt

# —————————————————————————————
# PARTE A — ENTRENAMIENTO
# —————————————————————————————

def entrenar_y_guardar_modelo(ruta_csv,
                              nombre_modelo="model_descalibracion.pkl",
                              nombre_scaler="scaler_descalibracion.pkl",
                              nombre_media="mean_sensors.pkl"):
    print("📌 Cargando dataset para entrenamiento...")
    df = pd.read_csv(ruta_csv)

    temp_cols = [col for col in df.columns if "TMP" in col and "R_SHTHTR" in col]
    print(f"🔎 Columnas de temperatura detectadas: {temp_cols}")

    data = df[temp_cols].copy()
    data = data.fillna(data.mean())

    mean_per_sensor = data.mean()
    data_dev = data - mean_per_sensor

    scaler = StandardScaler()
    data_dev_scaled = scaler.fit_transform(data_dev)

    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    model.fit(data_dev_scaled)

    pickle.dump(model, open(nombre_modelo, "wb"))
    pickle.dump(scaler, open(nombre_scaler, "wb"))
    pickle.dump(mean_per_sensor, open(nombre_media, "wb"))

    print("✔ Modelo entrenado y parámetros guardados.")

    pred_train = model.predict(data_dev_scaled)
    anomalies_train = np.where(pred_train == -1)[0]

    plt.figure(figsize=(12,6))
    plt.title("Entrenamiento - Detección de Des-calibración")
    plt.plot(df.index, data[temp_cols[0]], label=temp_cols[0], color="blue")
    plt.scatter(df.index[anomalies_train],
                data[temp_cols[0]].iloc[anomalies_train],
                color="red", marker="x", label="Anomalías")
    plt.xlabel("Ciclo")
    plt.ylabel("Temperatura")
    plt.legend()
    plt.grid(True)
    plt.show()

# —————————————————————————————
# PARTE B — CARGAR MODELO
# —————————————————————————————

model = None
scaler = None
mean_per_sensor = None

def cargar_modelo(nombre_modelo="model_descalibracion.pkl",
                  nombre_scaler="scaler_descalibracion.pkl",
                  nombre_media="mean_sensors.pkl"):
    global model, scaler, mean_per_sensor

    model = pickle.load(open(nombre_modelo, "rb"))
    scaler = pickle.load(open(nombre_scaler, "rb"))
    mean_per_sensor = pickle.load(open(nombre_media, "rb"))

    print("📌 Modelo cargado.")

# —————————————————————————————
# DETECTAR DES-CALIBRACIÓN (1 Lectura)
# —————————————————————————————

def detectar_descalibracion(nueva_lectura, mostrar_grafica=False):
    if model is None or scaler is None or mean_per_sensor is None:
        raise RuntimeError("Modelo no cargado. Usa cargar_modelo() primero.")

    df_new = pd.DataFrame([nueva_lectura])
    df_new = df_new.fillna(mean_per_sensor)
    df_new = df_new[mean_per_sensor.index]

    df_dev_new = df_new - mean_per_sensor
    X_scaled_new = scaler.transform(df_dev_new)

    pred = model.predict(X_scaled_new)[0]

    if mostrar_grafica:
        plt.figure(figsize=(8,4))
        plt.plot(df_new.columns, df_new.values.flatten(), 'o-', label="Temp observada")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.legend()
        plt.show()

    return pred == -1

# —————————————————————————————
# FUNCIÓN INTERACTIVA PARA INGRESAR LECTURA
# —————————————————————————————

def pedir_temperaturas():
    """
    Pide 3 temperaturas al usuario y las pasa al modelo
    para detectar des-calibracion.
    """
    sensores = list(mean_per_sensor.index[:3])  # tomar 3 columnas de sensores

    print("\n👉 Ingresa 3 nuevas temperaturas:")

    nueva_lectura = {}
    for sensor in sensores:
        while True:
            try:
                valor = float(input(f"Ingrese valor para {sensor}: "))
                nueva_lectura[sensor] = valor
                break
            except ValueError:
                print("❌ Entrada inválida. Ingresa un número.")

    # Completar el resto de sensores con la media histórica
    for sensor in mean_per_sensor.index[3:]:
        nueva_lectura[sensor] = mean_per_sensor[sensor]

    alerta = detectar_descalibracion(nueva_lectura)
    return alerta

# —————————————————————————————
# BLOQUE PRINCIPAL
# —————————————————————————————

if __name__ == "__main__":
    print("\n--- ENTRENANDO MODELO ---")
    entrenar_y_guardar_modelo("molding_machine.csv")

    print("\n--- CARGANDO MODELO ENTRENADO ---")
    cargar_modelo()

    alerta = pedir_temperaturas()

    if alerta:
        print("\n🚨 ALERTA: Posible des-calibración detectada.")
    else:
        print("\n✔ Lectura normal.")