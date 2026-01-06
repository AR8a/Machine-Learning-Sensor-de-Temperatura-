from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# -------- LIMPIAR CONSOLA --------
os.system('cls' if os.name == 'nt' else 'clear')

# -------- SELECCIONAS UN ARCHIVO --------
# Crear una ventana raíz (aunque no se va a mostrar)
root = tk.Tk()
root.withdraw()  # Ocultar la ventana raíz

# -------- ABRIR UN CUADRO DE DIALOGO PARA SELECCIONAR EL ARCHIVO --------
file_path = filedialog.askopenfilename(title="Selecciona un archivo", filetypes=[("Archivos CSV," "*.csv")]
)

if not file_path:
    print("No se seleccionó ningún archivo. Saliendo del programa.")
    exit()

print(f"Archivo seleccionado: {file_path}")

# -------- CONFIGURACION DINÁMICA --------
SENSOR_ID = 1
nu_value = 0.05

ERROR_COLUMN = f'Error_T{SENSOR_ID}'
T_PROMEDIO_COLUMN = 'Tpromedio(°C)'

CSV_FILE = file_path # Usamos el archivo seleccionado

print(f"Configuración lista para el Sensor {SENSOR_ID}")
print(f"Analizando las variables: {T_PROMEDIO_COLUMN} y {ERROR_COLUMN}")

# -------- CARGA INTELIGENTE DEL ARCHIVO --------
file_extension = os.path.splitext(CSV_FILE)[1].lower()

if file_extension == '.csv':
    df_base = pd.read_csv(CSV_FILE)
elif file_extension in ['.xlsx', '.xls']:
    df_base = pd.read_excel(CSV_FILE)
else:
    raise ValueError("Formato de archivo no soportado. Usa CSV o Excel (.xlsx)")

# -------- ADAPTACIÓN DE COLUMNAS (TERMOPAR TIPO K) --------
df_base = df_base.rename(columns={
    'Tref(°C)': 'Tpromedio(°C)',
    'Tsensor1_K(°C)': 'Tsensor1(°C)',
    'Tsensor2_K(°C)': 'Tsensor2(°C)',
    'Tsensor3_K(°C)': 'Tsensor3(°C)'
})

print("Columnas detectadas en el archivo:", df_base.columns.tolist())

# Seleccionar las dos variables clave de forma dinámica
features = [T_PROMEDIO_COLUMN, ERROR_COLUMN]
X = df_base[features]

print(f"Total de registros iniciales: {len(df_base)}")
print(f"Variables a modelar: {T_PROMEDIO_COLUMN} y {ERROR_COLUMN}")

# Gráfica de la distribución inicial
plt.figure(figsize=(8, 5))
plt.scatter(df_base[T_PROMEDIO_COLUMN], df_base[ERROR_COLUMN], c='green', s=5, alpha=0.7)
plt.title(f'Gráfico 1: Distribución de datos crudos (Sensor {SENSOR_ID})')
plt.xlabel(T_PROMEDIO_COLUMN)
plt.ylabel(ERROR_COLUMN)
plt.grid(True, linestyle='--')
plt.show()

# -------- ESTANDARIZACIÓN DE LOS DATOS --------
# Instancia de la herramienta de escalado
scaler = StandardScaler()

# Transformar los datos para que tengan Media = 0 y Desviación Estándar = 1
X_scaled = scaler.fit_transform(X)

X_train_scaled = X_scaled

# Nombres dinámicos de las columnas estandarizadas
T_SCALED_COLUMN = f'Tpromedio_Scaled_S{SENSOR_ID}'
ERROR_SCALED_COLUMN = f'Error_Scaled_S{SENSOR_ID}'

# DataFrame df_plot con los datos estandarizados para la gráfica de contorno.
df_plot = pd.DataFrame(X_scaled, columns=[T_SCALED_COLUMN, ERROR_SCALED_COLUMN], index=df_base.index)


# Gráfica de la distribución de datos estandarizada
plt.figure(figsize=(8, 5))
plt.scatter(df_plot[T_SCALED_COLUMN], df_plot[ERROR_SCALED_COLUMN], c='purple', s=5, alpha=0.7)
plt.title(f'Gráfico 2: Distribución de datos estandarizados (Sensor {SENSOR_ID})')
plt.xlabel(f'{T_PROMEDIO_COLUMN} (Estandarizado)')
plt.ylabel(f'{ERROR_COLUMN} (Estandarizado)')
plt.grid(True, linestyle='--')
plt.show()

# -------- IMPLEMENTACIÓN Y ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING --------
# Se usa el nu_value definido en la configuración
model = OneClassSVM(kernel='rbf', gamma='auto', nu=nu_value)

model.fit(X_train_scaled)
print(f"Entrenamiento de la IA para Sensor {SENSOR_ID} finalizado.")

# -------- DETECCIÓN Y RESULTADOS --------
df_base['prediction'] = model.predict(X_scaled)
df_base['is_anomaly'] = df_base['prediction'].apply(lambda x: 1 if x == -1 else 0)

# CLAVE: Transferir la columna de etiquetas al DataFrame de plot para el Gráfico 4
df_plot['is_anomaly'] = df_base['is_anomaly']

anomalies_count = df_base['is_anomaly'].sum()
total_records = len(df_base)

print(f"Resultados de detección de anomalías para Sensor {SENSOR_ID}")
print(f"Anomalías detectadas: {anomalies_count} ({anomalies_count/total_records*100:.3f}%)")

# Separar los datos para la visualización de datos crudos
df_anomalies = df_base[df_base['is_anomaly'] == 1]
df_normal = df_base[df_base['is_anomaly'] == 0]

plt.figure(figsize=(10, 6))

# Puntos normales en azul (Gráfico 3)
plt.scatter(df_normal[T_PROMEDIO_COLUMN], df_normal[ERROR_COLUMN], c='blue', s=10, label='Comportamiento normal', alpha=0.6)

# Puntos anómalos en rojo
plt.scatter(df_anomalies[T_PROMEDIO_COLUMN], df_anomalies[ERROR_COLUMN], c='red', s=60, label='Anomalía detectada', marker='x', linewidths=2, alpha=0.4)

plt.title(f'Gráfico 3: Anomalías detectadas (Sensor {SENSOR_ID})')
plt.xlabel(T_PROMEDIO_COLUMN)
plt.ylabel(ERROR_COLUMN)
plt.legend()
plt.grid(True, linestyle='--')
plt.show()

# -------- CONTORNO DE DECISIÓN --------
# Crear una malla de puntos para evaluar el modelo en todo el espacio
xx, yy = np.meshgrid(np.linspace(df_plot[T_SCALED_COLUMN].min() - 0.1, df_plot[T_SCALED_COLUMN].max() + 0.1, 100),
                     np.linspace(df_plot[ERROR_SCALED_COLUMN].min() - 0.1, df_plot[ERROR_SCALED_COLUMN].max() + 0.1, 100))

# Evaluar la función de decisión del modelo en cada punto de la malla (Z)
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))

# Rellenar las áreas que el modelo clasifica como anómalas (valores de Z < 0)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Reds', alpha=0.4)

# Trazar la línea de decisión (donde Z = 0)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')

# Re-graficar los datos estandarizados para contexto
plt.scatter(df_plot[df_plot['is_anomaly'] == 0][T_SCALED_COLUMN], df_plot[df_plot['is_anomaly'] == 0][ERROR_SCALED_COLUMN],
            c='blue', s=10, label='Normal', alpha=0.6)
plt.scatter(df_plot[df_plot['is_anomaly'] == 1][T_SCALED_COLUMN], df_plot[df_plot['is_anomaly'] == 1][ERROR_SCALED_COLUMN],
            c='red', s=60, label='Anomalía', marker='x', linewidths=2)

plt.title(f'Gráfico 4: Límite de decisión del OC-SVM (Sensor {SENSOR_ID})')
plt.xlabel(f'{T_PROMEDIO_COLUMN} (Estandarizado)')
plt.ylabel(f'{ERROR_COLUMN} (Estandarizado)')
plt.legend()
plt.show()