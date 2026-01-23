import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# CONFIGURACIÓN

SENSORES_USADOS = [
    "R_SHTHTR29TMP",
    "R_SHTHTR30TMP",
    "R_SHTHTR31TMP"
]

MODELO_FILE = "model_descalibracion.pkl"
SCALER_FILE = "scaler_descalibracion.pkl"
MEDIA_FILE  = "mean_sensors.pkl"

# ENTRENAMIENTO

def entrenar_y_guardar_modelo(ruta_csv):
    print("📌 Cargando dataset...")
    df = pd.read_csv(ruta_csv)

    data = df[SENSORES_USADOS].dropna()

    # Media histórica (referencia de calibración)
    mean_per_sensor = data.mean()

    # Desviación respecto a la media → des-calibración
    data_dev = data - mean_per_sensor

    scaler = StandardScaler()
    data_dev_scaled = scaler.fit_transform(data_dev)

    model = OneClassSVM(
        kernel="rbf",
        gamma="auto",
        nu=0.05
    )
    model.fit(data_dev_scaled)

    # Guardar artefactos
    pickle.dump(model, open(MODELO_FILE, "wb"))
    pickle.dump(scaler, open(SCALER_FILE, "wb"))
    pickle.dump(mean_per_sensor, open(MEDIA_FILE, "wb"))

    print("✔ Modelo de des-calibración entrenado y guardado.")

    # Visualización de entrenamiento
    pred = model.predict(data_dev_scaled)
    anomalies = np.where(pred == -1)[0]

    plt.figure(figsize=(10,5))
    plt.plot(data.index, data[SENSORES_USADOS[0]], label=SENSORES_USADOS[0])
    plt.scatter(
        data.index[anomalies],
        data[SENSORES_USADOS[0]].iloc[anomalies],
        color="red",
        marker="x",
        label="Des-calibración"
    )
    plt.title("Entrenamiento – Detección de des-calibración")
    plt.xlabel("Ciclo")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()

# CARGAR MODELO

model = None
scaler = None
mean_per_sensor = None

def cargar_modelo():
    global model, scaler, mean_per_sensor

    model = pickle.load(open(MODELO_FILE, "rb"))
    scaler = pickle.load(open(SCALER_FILE, "rb"))
    mean_per_sensor = pickle.load(open(MEDIA_FILE, "rb"))

    print("📌 Modelo cargado correctamente.")

# DETECCIÓN DE DES-CALIBRACIÓN (UNA LECTURA)

def detectar_descalibracion(nueva_lectura, mostrar_grafica=True):
    df_new = pd.DataFrame([nueva_lectura])

    # Desviación respecto a la referencia
    df_dev = df_new - mean_per_sensor
    X_scaled = scaler.transform(df_dev)

    pred = model.predict(X_scaled)[0]

    if mostrar_grafica:
        plt.figure(figsize=(7,4))
        plt.plot(df_new.columns, df_new.values.flatten(), 'o-')
        plt.ylabel("Temperatura (°C)")
        plt.title("Lectura ingresada")
        plt.grid(True)
        plt.show()

    return pred == -1

# ENTRADA POR TECLADO

def pedir_temperaturas():
    print("\n👉 Ingresa las temperaturas:")

    nueva_lectura = {}

    for sensor in SENSORES_USADOS:
        while True:
            try:
                nueva_lectura[sensor] = float(
                    input(f"Ingrese valor para {sensor}: ")
                )
                break
            except ValueError:
                print("❌ Ingresa un número válido.")

    return nueva_lectura

# MAIN

if __name__ == "__main__":
    entrenar_y_guardar_modelo("molding_machine.csv")
    cargar_modelo()

    lectura = pedir_temperaturas()
    alerta = detectar_descalibracion(lectura)

    if alerta:
        print("\n🚨 ALERTA: Posible des-calibración detectada.")
    else:
        print("\n✔ Lectura normal.")
