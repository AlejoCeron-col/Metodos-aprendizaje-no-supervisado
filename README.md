# Sistema de Transporte Masivo - Aprendizaje No Supervisado

## Descripción

Este proyecto utiliza técnicas de aprendizaje automático no supervisado para analizar y agrupar estaciones de un sistema de transporte masivo según sus características. Se emplea el algoritmo **K-Means** para identificar patrones relacionados con las conexiones, los tiempos de desplazamiento, la cantidad estimada de pasajeros y las rutas.

El proyecto utiliza un dataset simulado con fines académicos.

## ¿Qué hace el proyecto?

- Crea un conjunto de datos con estaciones y características del transporte.
- Agrupa las estaciones según sus similitudes mediante K-Means.
- Asigna nombres descriptivos a los grupos según sus niveles relativos de conectividad y demanda.
- Prueba el modelo con estaciones nuevas que no participaron en el entrenamiento.
- Genera un gráfico de dispersión para visualizar los grupos encontrados.

## Tecnologías utilizadas

- **Python:** lenguaje de programación.
- **Pandas:** creación y manipulación de datos.
- **Scikit-learn:** normalización de datos y algoritmo K-Means.
- **Matplotlib:** visualización de los resultados.

## Instalación

1. Instala Python 3.

2. Instala las dependencias:

   ```bash
   python -m pip install pandas scikit-learn matplotlib
   ```

## Ejecución

Ejecuta el archivo principal desde la terminal o Vscode:

```bash
python actividad 4.py
```

El programa mostrará las estaciones agrupadas, los perfiles descriptivos, los resultados de la prueba con datos nuevos y un gráfico de dispersión.

## Funcionamiento del modelo

1. Se seleccionan las variables numéricas de las estaciones.
2. Se normalizan los datos mediante `StandardScaler`.
3. K-Means identifica tres grupos de estaciones según sus similitudes.
4. Se calculan los perfiles promedio para asignar nombres descriptivos a los grupos.
5. Se incorporan estaciones nuevas para comprobar cómo las clasifica el modelo.
