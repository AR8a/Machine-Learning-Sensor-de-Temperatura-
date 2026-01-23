import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# 1. CARGAR DATASET

df = pd.read_csv("molding_machine.csv")

# 2. SELECCIONAR SOLO LOS 3 SENSORES

sensores = [
    "R_SHTHTR29TMP",
    "R_SHTHTR30TMP",
    "R_SHTHTR31TMP"
]

data = df[sensores].dropna()

# 3. ESCALADO Y ENTRENAMIENTO

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

model = OneClassSVM(
    kernel="rbf",
    gamma=0.001,
    nu=0.05
)
model.fit(data_scaled)

print("✔ Modelo entrenado con éxito usando 3 sensores.")

# 4. INGRESAR TEMPERATURAS POR TECLADO

print("\n👉 Ingresa nuevas temperaturas:")

nuevos_valores = []

for sensor in sensores:
    while True:
        try:
            valor = float(input(f"Ingrese valor para {sensor}: "))
            nuevos_valores.append(valor)
            break
        except ValueError:
            print("❌ Ingresa un número válido.")

# 5. PREDICCIÓN

df_new = pd.DataFrame([nuevos_valores], columns=sensores)
df_new_scaled = scaler.transform(df_new)

pred = model.predict(df_new_scaled)[0]

if pred == -1:
    print("\n🚨 ALERTA: La lectura es una **ANOMALÍA**.")
else:
    print("\n✔ Lectura **NORMAL**.")

# -----------------------------------------
# 6. GRÁFICA

plt.figure(figsize=(7, 4))
plt.plot(sensores, nuevos_valores, marker="o", linestyle="-")
plt.ylabel("Temperatura (°C)")
plt.title("Temperaturas ingresadas")
plt.grid(True)
plt.show()
