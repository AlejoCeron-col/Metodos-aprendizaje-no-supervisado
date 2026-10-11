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
