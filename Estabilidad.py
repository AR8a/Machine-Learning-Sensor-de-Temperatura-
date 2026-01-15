import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ENTRENAMIENTO

def entrenar_modelo_estabilidad(ruta_csv="molding_machine.csv",
                                model_file="model_est_sensor.pkl",
                                scaler_file="scaler_est_sensor.pkl",
                                mean_file="mean_est_sensor.pkl"):
    """
    Entrena un modelo OneClassSVM para detectar si una lectura completa
    de temperaturas está dentro de comportamiento ‘estable’.
    """

    # Cargar datos históricos
    df = pd.read_csv(ruta_csv)

    # Seleccionar columnas de temperatura de sensores
    temp_cols = [col for col in df.columns if "R_SHTHTR" in col and "TMP" in col]
    print("🔎 Sensores usados para estabilidad:", temp_cols)

    # Eliminamos filas con NaN y conservamos solo valores numéricos
    data = df[temp_cols].dropna()

    # 👉 Entrenamos el modelo con las lecturas completas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # El modelo OneClassSVM aquí representa la “zona estable”
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.02)
    model.fit(X_scaled)

    # Guardar el modelo y parámetros
    pickle.dump(model, open(model_file, "wb"))
    pickle.dump(scaler, open(scaler_file, "wb"))
    pickle.dump(temp_cols, open(mean_file, "wb"))

    print("✅ Modelo de estabilidad entrenado y guardado.")

# CARGAR MODELO

def cargar_modelo_estabilidad(model_file="model_est_sensor.pkl",
                              scaler_file="scaler_est_sensor.pkl",
                              mean_file="mean_est_sensor.pkl"):
    """
    Carga el modelo, scaler y columnas de sensores.
    """
    model = pickle.load(open(model_file, "rb"))
    scaler = pickle.load(open(scaler_file, "rb"))
    temp_cols = pickle.load(open(mean_file, "rb"))
    return model, scaler, temp_cols

# 3️⃣ EVALUAR NUEVA LECTURA

def evaluar_estabilidad(lectura, model, scaler, temp_cols):
    """
    lectura: dict con pares {sensor: valor}
    Ejemplo:
       {"R_SHTHTR01TMP":220.1, "R_SHTHTR02TMP":219.8, ...}
    """

    df_new = pd.DataFrame([lectura])

    # Asegurarnos de que estén todas las columnas
    df_new = df_new.reindex(columns=temp_cols)

    # Si faltan sensores, imputamos cero. (alternativo: usar media histórica)
    df_new = df_new.fillna(df_new.mean())

    # Escalar
    X_new_scaled = scaler.transform(df_new)

    # Predecir: 1 → estable, -1 → inestable
    pred = model.predict(X_new_scaled)[0]

    return pred

# MOSTRAR GRAFICA Y ALERTA

def mostrar_grafica_estabilidad(lectura, temp_cols):
    """
    Grafica la lectura y resalta si es normal o no
    """
    # Preparar datos
    values = [lectura.get(col, np.nan) for col in temp_cols]

    plt.figure(figsize=(10,5))
    plt.plot(temp_cols, values, 'o-', label="Lectura actual")
    plt.title("Evaluación de estabilidad por sensor")
    plt.xticks(rotation=90)
    plt.ylabel("Temperatura (°C)")
    plt.grid(True)
    plt.legend()
    plt.show()

# BLOQUE PRINCIPAL

if __name__ == "__main__":
    print("\n--- Entrenando modelo de estabilidad ---")
    entrenar_modelo_estabilidad("molding_machine.csv")

    print("\n--- Cargando modelo entrenado ---")
    model_est, scaler_est, temp_cols = cargar_modelo_estabilidad()

    print("\n--- Evaluar nueva lectura ---")

    # Ejemplo: lectura de un sensor con TODOS los sensores
    nueva_lectura = {
        "R_SHTHTR01TMP": 220.3,
        "R_SHTHTR02TMP": 220.0,
        "R_SHTHTR03TMP": 219.9,
        "R_SHTHTR04TMP": 220.1,
        "R_SHTHTR05TMP": 220.2,
        "R_SHTHTR06TMP": 220.0,
        "R_SHTHTR07TMP": 220.1,
        "R_SHTHTR08TMP": 220.0,
        "R_SHTHTR09TMP": 219.8,
        "R_SHTHTR10TMP": 220.3,
        "R_SHTHTR11TMP": 220.0,
        "R_SHTHTR12TMP": 220.1,
        "R_SHTHTR13TMP": 220.2,
        "R_SHTHTR14TMP": 220.0,
        "R_SHTHTR15TMP": 220.3,
        "R_SHTHTR16TMP": 220.1,
        "R_SHTHTR17TMP": 220.4,
        "R_SHTHTR18TMP": 220.1,
        "R_SHTHTR19TMP": 220.0,
        "R_SHTHTR20TMP": 220.2,
        "R_SHTHTR21TMP": 220.0,
        "R_SHTHTR22TMP": 220.1,
        "R_SHTHTR23TMP": 220.0,
        "R_SHTHTR24TMP": 220.0,
        "R_SHTHTR25TMP": 220.2,
        "R_SHTHTR26TMP": 220.1,
        "R_SHTHTR27TMP": 220.0,
        "R_SHTHTR28TMP": 220.2,
        "R_SHTHTR29TMP": 220.0,
        "R_SHTHTR30TMP": 220.1,
        "R_SHTHTR31TMP": 220.3,
        "R_SHTHTR32TMP": 220.0,
        "R_SHTHTR33TMP": 220.1,
    }

    pred_est = evaluar_estabilidad(nueva_lectura, model_est, scaler_est, temp_cols)

    if pred_est == 1:
        print("✔ ESTABLE — La lectura está dentro del rango normal.")
    else:
        print("🚨 INESTABLE — La lectura podría estar fuera de lo normal.")

    print("\n--- Mostrando gráfica ---")
    mostrar_grafica_estabilidad(nueva_lectura, temp_cols)

