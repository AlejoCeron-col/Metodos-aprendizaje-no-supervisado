
# ============================================================
# MODELO DE APRENDIZAJE NO SUPERVISADO
# Sistema de transporte masivo
# Algoritmo: K-Means
# ============================================================

# ============================================================
# 1. IMPORTAR LAS LIBRERIAS NECESARIAS MARCO
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ============================================================
# 2. CREAR EL DATASET MARCO
# ============================================================


datos = {
    "Estacion": [
        "Terminal",
        "Centro",
        "Parque",
        "Universidad",
        "Estadio",
        "Hospital",
        "Aeropuerto"
    ],

    "Conexiones": [3, 5, 4, 3, 2, 4, 1],

    "Tiempo_promedio": [12, 8, 10, 9, 14, 7, 18],

    "Pasajeros": [900, 1500, 1100, 1300, 700, 1400, 400],

    "Rutas": [3, 4, 3, 2, 2, 3, 1]
}

# Convertir los datos en una tabla de pandas
df = pd.DataFrame(datos)

# Mostrar el dataset original
print("DATASET DEL TRANSPORTE MASIVO")
print(df.to_string(index=False))

# ============================================================
# 3. SELECCIONAR LAS VARIABLES NUMERICAS MARCO
# ============================================================

# No incluimos el nombre de la estacion porque es texto.

variables = [
    "Conexiones",
    "Tiempo_promedio",
    "Pasajeros",
    "Rutas"
]

X = df[variables]
# ============================================================
# 4. NORMALIZAR LOS DATOS
# ============================================================

# Esto evita que una variable con valores grandes,
# como pasajeros, domine el calculo de distancias.

escalador = StandardScaler()
X_escalado = escalador.fit_transform(X)

# ============================================================
# 5. CREAR Y ENTRENAR EL MODELO DE K-MEANS 
# ============================================================

# Se buscan tres grupos de estaciones.

modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# El algoritmo identifica los grupos automaticamente.
df["Grupo"] = modelo.fit_predict(X_escalado)


# ============================================================
# 6. ASIGNAR NOMBRES DESCRIPTIVOS A LOS GRUPOS 
# ============================================================

# Variables que representan la actividad de una estacion.
# El tiempo de viaje se analiza por separado, porque un
# tiempo alto no siempre significa que una estacion sea mala.

variables_actividad = [
    "Conexiones",
    "Pasajeros",
    "Rutas"
]

# Calcular las caracteristicas promedio de cada grupo.
perfiles = df.groupby("Grupo")[variables_actividad].mean()

# Normalizar los promedios para que pasajeros no domine
# el calculo solo por tener valores numericos mas grandes.
escalador_perfiles = StandardScaler()
perfiles_escalados = escalador_perfiles.fit_transform(perfiles)

# Calcular un indicador relativo de actividad por grupo.
# Las tres variables tienen la misma importancia.
perfiles["Indice_actividad"] = perfiles_escalados.mean(axis=1)

# Ordenar los grupos desde menor hasta mayor actividad.
grupos_ordenados = perfiles["Indice_actividad"].sort_values().index

# Asignar nombres segun su posicion relativa.
nombres_grupos = {}

etiquetas = [
    "Baja conectividad y demanda",
    "Conectividad y demanda intermedia",
    "Alta conectividad y demanda"
]

for posicion, grupo in enumerate(grupos_ordenados):
    if len(grupos_ordenados) == 3:
        nombres_grupos[grupo] = etiquetas[posicion]
    else:
        nombres_grupos[grupo] = f"Perfil de actividad {posicion + 1}"

# Agregar el nombre descriptivo a las estaciones originales.
df["Perfil"] = df["Grupo"].map(nombres_grupos)

# Mostrar el perfil de cada grupo.
print("\nPERFILES DE LAS ESTACIONES")
print(
    df[["Estacion", "Grupo", "Perfil"]]
    .to_string(index=False)
)

# Mostrar las caracteristicas promedio de cada grupo.
print("\nCARACTERISTICAS DE CADA PERFIL")
print(
    df.groupby("Perfil")[variables + ["Grupo"]]
    .mean(numeric_only=True)
    .round(2)
)

# ============================================================
# 7. MOSTRAR LOS RESULTADOS DEL AGRUPAMIENTO
# ============================================================

print("\nESTACIONES AGRUPADAS")
print(
    df[["Estacion", "Conexiones",
        "Tiempo_promedio", "Pasajeros",
        "Rutas", "Grupo"]].to_string(index=False)
)

# ============================================================
# 8. CALCULAR LOS PERFILES PROMEDIO DE CADA GRUPO
# ============================================================

print("\nCARACTERISTICAS PROMEDIO DE CADA GRUPO")
print(
    df.groupby("Grupo")[variables]
    .mean()
    .round(2)
)

# ============================================================
# 8. CREAR UN GRAFICO DE DISPERSION PARA VISUALIZAR LOS GRUPOS
# ============================================================


# Eje X: cantidad de conexiones.
# Eje Y: cantidad estimada de pasajeros.
# El color representa el grupo encontrado.

plt.figure(figsize=(10, 6))

for grupo in sorted(df["Grupo"].unique()):
    estaciones = df[df["Grupo"] == grupo]

    plt.scatter(
        estaciones["Conexiones"],
        estaciones["Pasajeros"],
        s=100,
        label=nombres_grupos[grupo]
    )

    # Escribir el nombre de cada estacion en el grafico
    for _, fila in estaciones.iterrows():
        plt.annotate(
            fila["Estacion"],
            (fila["Conexiones"], fila["Pasajeros"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

plt.title("Agrupacion de estaciones mediante K-Means")
plt.xlabel("Cantidad de conexiones")
plt.ylabel("Pasajeros estimados")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Mostrar el grafico
plt.show()

# ============================================================
# 9. PRUEBA CON ESTACIONES NUEVAS
# ============================================================

# Creamos tres estaciones que NO estaban en el dataset
# utilizado para entrenar el modelo.

datos_nuevos = pd.DataFrame({
    "Estacion": ["Mercado", "Coliseo", "Zona Rural"],
    "Conexiones": [5, 2, 1],
    "Tiempo_promedio": [7, 13, 20],
    "Pasajeros": [1600, 750, 300],
    "Rutas": [4, 2, 1]
})

print("\nPRUEBA CON DATOS NUEVOS")
print(datos_nuevos.to_string(index=False))

# Seleccionamos las mismas variables del entrenamiento.
X_nuevo = datos_nuevos[variables]

# Aplicamos el MISMO escalador del entrenamiento.
# No debemos entrenar otro escalador con estos datos.
X_nuevo_escalado = escalador.transform(X_nuevo)

# El modelo asigna cada estación al grupo más cercano.
grupos_nuevos = modelo.predict(X_nuevo_escalado)

# Guardamos las predicciones en la tabla.
datos_nuevos["Grupo asignado"] = grupos_nuevos

# Convertir el numero del grupo en un nombre descriptivo.
datos_nuevos["Perfil"] = (
    datos_nuevos["Grupo asignado"].map(nombres_grupos)
)

print("\nRESULTADOS CON NOMBRES DESCRIPTIVOS")

print(
    datos_nuevos[
        ["Estacion", "Conexiones", "Tiempo_promedio",
         "Pasajeros", "Rutas", "Perfil"]
    ].to_string(index=False)
)

# Mostramos los resultados de la prueba.
print("\nRESULTADOS DE LAS ESTACIONES NUEVAS")

print(
    datos_nuevos[
        ["Estacion", "Conexiones",
         "Tiempo_promedio", "Pasajeros",
         "Rutas", "Grupo asignado"]
    ].to_string(index=False)
)

# Comprobar que el modelo generó una predicción
# para cada una de las estaciones nuevas.

if len(grupos_nuevos) == len(datos_nuevos):
    print("\nPRUEBA SUPERADA:")
    print("El modelo proceso todas las estaciones nuevas.")
else:
    print("\nPRUEBA FALLIDA:")
    print("No se procesaron todas las estaciones.")

# ============================================================
# 10. COMPARAR ESTACIONES NUEVAS CON LAS ORIGINALES
# ============================================================

print("\nCOMPARACION DE ESTACIONES POR GRUPO")

for _, nueva in datos_nuevos.iterrows():

    grupo = nueva["Grupo asignado"]

    # Buscar estaciones originales del mismo grupo.
    similares = df[df["Grupo"] == grupo]["Estacion"].tolist()

    print(f"\nEstacion nueva: {nueva['Estacion']}")
    print(f"Grupo asignado: {grupo}")
    print(f"Estaciones originales del grupo: {similares}")
