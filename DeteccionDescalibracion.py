from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset_Termopar_Tipo_k.csv')
sensor = 'Tsensor1_K(°C)'

x = df[[sensor]]
y = df['Tref(°C)']

modelo_calibracion = LinearRegression()
modelo_calibracion.fit(x,y)

df['temperatura_estimada'] = modelo_calibracion.predict(x)
df['residual'] = df['temperatura_estimada'] - df['Tref(°C)']
df['residual_mean'] = df['residual'].rolling(window=100, min_periods = 1).mean()
df['residual_std']  = df['residual'].rolling(window=100, min_periods = 1).std()

UMBRAL = 0.02  # °C aceptables
df['descalibrado'] = df['residual_mean'].abs() > UMBRAL

if df['descalibrado'].any():
    print("ALERTA: Se detectó des-calibración en el sensor.")
else:
    print("Sensor dentro de tolerancias.")

print("Cálculo de des-calibración completado.")
print("Primeras filas de residual_mean y descalibrado:")
print(df[['residual_mean','descalibrado']].head())

print("=== Resumen del modelo de calibración ===")
print("Coeficiente (a):", modelo_calibracion.coef_[0])
print("Intercept (b):", modelo_calibracion.intercept_)
print("Score (R²):", modelo_calibracion.score(x, y))


# Mostrar gráfico
plt.figure(figsize=(10,5))
plt.plot(df['residual_mean'], label='Residual promedio')
plt.axhline(UMBRAL, color='red', linestyle='--', label=f'Umbral +{UMBRAL}')
plt.axhline(-UMBRAL, color='red', linestyle='--', label=f'Umbral -{UMBRAL}')
plt.title("Residual promedio con umbrales de tolerancia")
plt.xlabel("Índice de muestra")
plt.ylabel("Residual promedio (°C)")
plt.legend()
plt.grid()
plt.show()

