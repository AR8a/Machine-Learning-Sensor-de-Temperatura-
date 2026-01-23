import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np

# —————————————————————————————
# CARGAR DATASET Y ENTRENAR MODELO

df = pd.read_csv("molding_machine.csv")

temp_cols = [col for col in df.columns if "R_SHTHTR" in col and "TMP" in col]
data = df[temp_cols].dropna()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.05)
model.fit(data_scaled)

print("\n✔ Modelo de detección de anomalías entrenado con éxito.")

# —————————————————————————————
# CALCULAR MEDIA HISTÓRICA (para completar entradas)


mean_per_sensor = data.mean()

# —————————————————————————————
# FUNCIÓN INTERACTIVA: INGRESAR 3 NUEVAS TEMPERATURAS


def probar_temperaturas():
    print("\n👉 Ingresa valores de temperatura nuevos:")

    nueva_lectura = {}

    # Tomamos los primeros 3 sensores para pedir valores:
    sensores = temp_cols[:3]

    for sensor in sensores:
        while True:
            try:
                valor = float(input(f"Ingrese valor para {sensor}: "))
                nueva_lectura[sensor] = valor
                break
            except ValueError:
                print("❌ Entrada inválida. Ingresa un número válido.")

    # Completar el resto con la media histórica
    for sensor in temp_cols[3:]:
        nueva_lectura[sensor] = mean_per_sensor[sensor]

    return nueva_lectura

# —————————————————————————————
# PREDICCIÓN CON LOS 3 VALORES


nueva = probar_temperaturas()

# Transformar la entrada igual que en entrenamiento
df_new = pd.DataFrame([nueva])
df_new_scaled = scaler.transform(df_new)

pred = model.predict(df_new_scaled)[0]

if pred == -1:
    print("\n🚨 ALERTA: La lectura es una **anomalía**.")
else:
    print("\n✔ Lectura normal.")

# Grafica
plt.figure(figsize=(8,4))
plt.plot(temp_cols, df_new.values.flatten(), 'o-', label="Temp ingresadas + medias")
plt.xticks(rotation=90)
plt.ylabel("Temperatura (°C)")
plt.title("Temperaturas de prueba (completadas con media histórica)")
plt.grid(True)
plt.legend()
plt.show()

