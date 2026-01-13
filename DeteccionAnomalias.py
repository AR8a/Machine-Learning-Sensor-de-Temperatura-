import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import numpy as np

# 1️⃣ Cargar el dataset
df = pd.read_csv("molding_machine.csv")

# 2️⃣ Seleccionar solo las columnas de temperatura
temp_cols = [col for col in df.columns if "R_SHTHTR" in col and "TMP" in col]
data = df[temp_cols].dropna()  # eliminar filas con NaN

# 3️⃣ Escalar los datos (muy importante)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 4️⃣ Entrenar modelo de detección de anomalías
model = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.05)
model.fit(data_scaled)

# 5️⃣ Predicción (1 = normal, -1 = anomalía)
pred = model.predict(data_scaled)

# 6️⃣ Agregar columna de resultado
df_result = data.copy()
df_result["anomalía"] = np.where(pred == -1, "Sí", "No")

# 7️⃣ Visualizar resultados (ejemplo con una zona del molde)
plt.figure(figsize=(10,5))
plt.plot(data[temp_cols[0]], label='Temperatura', color='blue')
plt.scatter(df_result.index[df_result["anomalía"]=="Sí"],
            data[temp_cols[0]][df_result["anomalía"]=="Sí"],
            color='red', label='Anomalía')
plt.title(f"Detección de anomalías en {temp_cols[0]}")
plt.xlabel("Ciclo de inyección")
plt.ylabel("Temperatura (°C)")
plt.legend()
plt.show()

# 8️⃣ Mostrar resumen
print(df_result["anomalía"].value_counts())
