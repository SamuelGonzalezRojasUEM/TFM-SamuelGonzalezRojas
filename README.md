---

# VIGÍA Predictivo - Modelo Predictivo

## Descripción del Proyecto

Este repositorio contiene el código, configuraciones y resultados relacionados con el proyecto **VIGÍA Predictivo**. El objetivo principal de este proyecto es desarrollar un modelo predictivo capaz de anticipar fallos en las CTOs (Caja Terminal Óptica) mediante el análisis de datos históricos y características técnicas relevantes.

## Estructura del Repositorio

- **`logs/`**: Contiene los archivos de logs generados durante la ejecución del código. Estos logs documentan el progreso y cualquier incidencia ocurrida durante los procesos de entrenamiento y predicción.
  
- **`resultados/VIGIA_PREDICTIVO/`**: En esta carpeta se encuentran los resultados obtenidos del modelo predictivo, incluyendo gráficos, reportes y otros archivos generados después de la ejecución.

- **`db_conf.py`**: Archivo de configuración para la conexión a la base de datos. Contiene los parámetros necesarios para acceder a los datos que alimentan el modelo.

- **`entrenamiento_trees.py`**: Script encargado del entrenamiento del modelo predictivo utilizando árboles de decisión. Este script realiza la preparación de los datos, entrena el modelo y guarda el modelo entrenado.

- **`predicciones_produccion.py`**: Script utilizado para generar predicciones en el entorno de producción. Este archivo es una versión refinada y optimizada para la ejecución en un entorno productivo.

- **`predicciones_produccion - Antes de incluir clientes.py`**: Versión anterior del script de predicciones en producción. Este archivo se mantiene para referencia histórica y comparativa.

- **`predicciones_produccion - PRUEBA.py`**: Script de prueba para predicciones en producción. Este archivo se utiliza para pruebas específicas y no se utiliza en el entorno de producción final.

- **`predicciones_train_test.py`**: Script utilizado para generar predicciones durante la fase de entrenamiento y prueba. Este archivo es clave para validar el rendimiento del modelo antes de su despliegue.

- **`preprocesado.py`**: Script encargado del preprocesamiento de los datos. Realiza tareas como la limpieza de datos, generación de nuevas características y cualquier otra transformación necesaria antes del entrenamiento.

- **`settings.config`**: Archivo de configuración general que incluye parámetros utilizados en distintos scripts del proyecto. Este archivo centraliza configuraciones como rutas de acceso a archivos, parámetros de modelos, entre otros.

- **`sql_sentences_VIGIA.py`**: Contiene las sentencias SQL utilizadas para extraer datos de la base de datos. Este script automatiza la recuperación de datos necesarios para el entrenamiento y predicciones del modelo.

## Cómo Utilizar este Repositorio

### Requisitos Previos

Para ejecutar los scripts y reproducir los resultados, es necesario tener instalado:

- Python 3.x
- Las librerías especificadas en el archivo `requirements.txt`

### Instalación

1. Clona el repositorio en tu máquina local:

   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

### Ejecución de Scripts

1. **Entrenamiento del Modelo:**

   Para entrenar el modelo predictivo, ejecuta el siguiente comando:

   ```bash
   python entrenamiento_trees.py
   ```

2. **Generación de Predicciones:**

   Para generar predicciones utilizando el modelo entrenado:

   ```bash
   python predicciones_produccion.py
   ```

### Resultados

Los resultados generados se almacenan en la carpeta `resultados/VIGIA_PREDICTIVO/`. Aquí puedes encontrar gráficos, reportes y otros archivos que muestran el rendimiento del modelo.

## Contribuciones

Las contribuciones al proyecto son bienvenidas. Si deseas mejorar algún aspecto o agregar nuevas funcionalidades, por favor, abre un issue o crea un pull request.

## Contacto

Si tienes alguna pregunta o sugerencia, no dudes en contactar a través de [tu correo electrónico] o abrir un issue en el repositorio.

---
