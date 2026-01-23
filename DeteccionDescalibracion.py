import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt

# —————————————————————————————
# PARTE A — ENTRENAMIENTO


def entrenar_y_guardar_modelo(ruta_csv,
                              nombre_modelo="model_descalibracion.pkl",
                              nombre_scaler="scaler_descalibracion.pkl",
                              nombre_media="mean_sensors.pkl"):
    print("Cargando dataset para entrenamiento...")
    df = pd.read_csv(ruta_csv)

    # Detectar columnas de temperatura automáticamente
    temp_cols = [col for col in df.columns if "TMP" in col and "R_SHTHTR" in col]
    print(f"🔎 Columnas de temperatura detectadas: {temp_cols}")

    data = df[temp_cols].copy()
    data = data.fillna(data.mean())

    # Centrar (desviación respecto a media histórica)
    mean_per_sensor = data.mean()
    data_dev = data - mean_per_sensor

    # Escalado
    scaler = StandardScaler()
    data_dev_scaled = scaler.fit_transform(data_dev)

    # Entrenar OneClass SVM
    model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
    model.fit(data_dev_scaled)

    # Guardar modelo + scaler + medias
    pickle.dump(model, open(nombre_modelo, "wb"))
    pickle.dump(scaler, open(nombre_scaler, "wb"))
    pickle.dump(mean_per_sensor, open(nombre_media, "wb"))

    print("✔ Modelo entrenado y parámetros guardados.")

    # Gráfica de entrenamiento
    pred_train = model.predict(data_dev_scaled)
    anomalies_train = np.where(pred_train == -1)[0]

    plt.figure(figsize=(12,6))
    plt.title("Entrenamiento - Detección de des-calibración")
    plt.plot(df.index, data[temp_cols[0]], label=temp_cols[0], color="blue")
    plt.scatter(df.index[anomalies_train],
                data[temp_cols[0]].iloc[anomalies_train],
                color="red", marker="x", label="Anomalías", alpha=0.6)
    plt.xlabel("Ciclo de inyección")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.grid(True)
    plt.show()

# —————————————————————————————
# PARTE B — CARGAR MODELO


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

    print("Modelo, scaler y medias cargados.")

# —————————————————————————————
# DETECTAR DES-CALIBRACIÓN (INDIVIDUAL)


def detectar_descalibracion(nueva_lectura, mostrar_grafica=False):
    if model is None or scaler is None or mean_per_sensor is None:
        raise RuntimeError("Modelo no cargado. Usa cargar_modelo() primero.")

    # Crear DataFrame de 1 sola lectura
    df_new = pd.DataFrame([nueva_lectura])
    df_new = df_new.fillna(mean_per_sensor)
    df_new = df_new[mean_per_sensor.index]

    # Desviación y escalado
    df_dev_new = df_new - mean_per_sensor
    X_scaled_new = scaler.transform(df_dev_new)

    pred = model.predict(X_scaled_new)[0]

    # Mostrar gráfica si se pide
    if mostrar_grafica:
        plt.figure(figsize=(8,4))
        plt.title("📊 Lectura nueva de temperaturas")
        plt.plot(df_new.columns, df_new.values.flatten(), 'o-', label="Temperaturas")
        plt.xticks(rotation=90)
        plt.ylabel("°C")
        plt.grid(True)
        plt.legend()
        plt.show()

    return pred == -1  # True si descalibrado

# —————————————————————————————
# NUEVA FUNCIÓN: PROBAR VARIAS LECTURAS DESDE CSV


import os

def probar_varias_desde_csv(ruta_csv_nuevas, mostrar_graficas=False):
    """Lee un CSV y evalúa si cada fila está descalibrada."""
    if not isinstance(ruta_csv_nuevas, str) or not os.path.isfile(ruta_csv_nuevas):
        print(f"❌ ERROR: '{ruta_csv_nuevas}' no es un archivo válido.")
        return None

    df_test = pd.read_csv(ruta_csv_nuevas)
    resultados = []

    for idx, fila in df_test.iterrows():
        lectura = fila.to_dict()
        es_descalibrado = detectar_descalibracion(lectura, mostrar_grafica=mostrar_graficas)

        resultados.append({
            "index": idx,
            "descalibrado": es_descalibrado
        })

        estado = "❌ DES-CALIBRADO" if es_descalibrado else "✔ NORMAL"
        print(f"Fila {idx} → {estado}")

    return pd.DataFrame(resultados)


# BLOQUE PRINCIPAL

if __name__ == "__main__":
    print("\n--- ENTRENANDO MODELO ---")
    entrenar_y_guardar_modelo("molding_machine.csv")

    print("\n--- CARGANDO MODELO ENTRENADO ---")
    cargar_modelo()

    print("\n--- PROBANDO UNA LECTURA INDIVIDUAL ---")
    ejemplo = {
        "R_SHTHTR01TMP": 219.9, "R_SHTHTR02TMP": 220.1,
        "R_SHTHTR03TMP": 220.2, "R_SHTHTR04TMP": 219.7,
        "R_SHTHTR05TMP": 220.0, "R_SHTHTR06TMP": 220.3,
        "R_SHTHTR07TMP": 219.8, "R_SHTHTR08TMP": 220.1,
        "R_SHTHTR09TMP": 220.0, "R_SHTHTR10TMP": 220.2,
        "R_SHTHTR11TMP": 220.1, "R_SHTHTR12TMP": 220.0,
        "R_SHTHTR13TMP": 220.3, "R_SHTHTR14TMP": 219.9,
        "R_SHTHTR15TMP": 220.0, "R_SHTHTR16TMP": 220.1,
        "R_SHTHTR17TMP": 220.2, "R_SHTHTR18TMP": 220.0,
        "R_SHTHTR19TMP": 220.1, "R_SHTHTR20TMP": 220.1,
        "R_SHTHTR21TMP": 220.0, "R_SHTHTR22TMP": 220.2,
        "R_SHTHTR23TMP": 219.8, "R_SHTHTR24TMP": 220.0,
        "R_SHTHTR25TMP": 220.1, "R_SHTHTR26TMP": 220.4,
        "R_SHTHTR27TMP": 220.0, "R_SHTHTR28TMP": 220.2,
        "R_SHTHTR29TMP": 220.1, "R_SHTHTR30TMP": 220.0,
        "R_SHTHTR31TMP": 220.3, "R_SHTHTR32TMP": 219.9,
        "R_SHTHTR33TMP": 220.0
    }

    alerta = detectar_descalibracion(ejemplo, mostrar_grafica=True)

    resultados = probar_varias_desde_csv("nuevas_temperaturas.csv", mostrar_graficas=True)
    print(resultados)

    if alerta:
        print("ALERTA: Posible des-calibración detectada.")
    else:
        print("✔ Lectura normal.")

        
    print("\n--- PROBANDO VARIAS LECTURAS DESDE CSV ---")
    resultados = probar_varias_desde_csv("nuevas_temperaturas.csv", mostrar_graficas=False)
    print(resultados)


