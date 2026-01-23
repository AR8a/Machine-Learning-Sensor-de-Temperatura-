import pandas as pd
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

MODELO_FILE = "model_estabilidad.pkl"
SCALER_FILE = "scaler_estabilidad.pkl"

# ENTRENAMIENTO

def entrenar_modelo_estabilidad(ruta_csv="molding_machine.csv"):
    print("📌 Cargando dataset...")
    df = pd.read_csv(ruta_csv)

    data = df[SENSORES_USADOS].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # One-Class SVM aprende la zona ESTABLE
    model = OneClassSVM(
        kernel="rbf",
        gamma="auto",
        nu=0.02   # pequeño → zona estable estricta
    )
    model.fit(X_scaled)

    pickle.dump(model, open(MODELO_FILE, "wb"))
    pickle.dump(scaler, open(SCALER_FILE, "wb"))

    print("✔ Modelo de estabilidad entrenado y guardado.")

# CARGAR MODELO

def cargar_modelo_estabilidad():
    model = pickle.load(open(MODELO_FILE, "rb"))
    scaler = pickle.load(open(SCALER_FILE, "rb"))
    return model, scaler

# EVALUAR ESTABILIDAD (1 LECTURA)

def evaluar_estabilidad(lectura, model, scaler, mostrar_grafica=True):
    df_new = pd.DataFrame([lectura])

    X_scaled = scaler.transform(df_new)
    pred = model.predict(X_scaled)[0]   # 1 = estable, -1 = inestable

    if mostrar_grafica:
        plt.figure(figsize=(7,4))
        plt.plot(df_new.columns, df_new.values.flatten(), 'o-')
        plt.ylabel("Temperatura (°C)")
        plt.title("Lectura ingresada (estabilidad)")
        plt.grid(True)
        plt.show()

    return pred

# ENTRADA POR TECLADO

def pedir_temperaturas():
    print("\n👉 Ingresa las temperaturas:")

    lectura = {}
    for sensor in SENSORES_USADOS:
        while True:
            try:
                lectura[sensor] = float(
                    input(f"Ingrese valor para {sensor}: ")
                )
                break
            except ValueError:
                print("❌ Ingresa un número válido.")

    return lectura

# MAIN

if __name__ == "__main__":
    print("\n--- ENTRENANDO MODELO DE ESTABILIDAD ---")
    entrenar_modelo_estabilidad("molding_machine.csv")

    print("\n--- CARGANDO MODELO ---")
    model_est, scaler_est = cargar_modelo_estabilidad()

    print("\n--- EVALUAR NUEVA LECTURA ---")
    lectura = pedir_temperaturas()

    pred = evaluar_estabilidad(lectura, model_est, scaler_est)

    if pred == 1:
        print("\n✔ ESTABLE — La lectura está dentro del comportamiento normal.")
    else:
        print("\n🚨 INESTABLE — La lectura muestra inestabilidad.")
