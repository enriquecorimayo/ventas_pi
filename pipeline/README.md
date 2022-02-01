### PIPELINE

- En el archivo pipeline.py se encuentra el script para la creación del modelo "pipe.pkl".
- En el archivo produccion.py se aplica este modelo para predecir el json "example", el resultado dió:1979.83.

### FURTHER WORK

- La función ft en pipeline.py es muy larga, se debería emprolijar creando funciones más pequeñas.
- La función ft no está preparada para lidiar con valores nulos más allá de las columnas tratadas en el notebook.
- Agregar Doctrings en la función.
- Hacer el análisis por grid search cv, Random search, etc.
- Agregar código "expanding window" y "rolling window" y evaluar los distintos resultados para detectar Data Drift, Data Concept en producción.
- Ordenar las direcciones de los archivos en el repositorio, sobre todo los pkl.


### NEXT STEPS

- Guardar los datos de train y test en alguna base de datos. (MySQL, Postgresql, etc)
- Desplegar la base en la nube (AWS RDS, Azure DB for MySQL, etc)
- Crear codigo ORM (SQL Alchemy) para conectarse a la base.
- Crear un DAG para re-entrenar el pipe cada un periodo determinado.(Airflow)
- Crear API (Django, Fast API, Flask).
- Conteinizar (Docker).
- Desplegar (Ej. AWS EC2). 
- CI/CD pipeline (Jenkins, GitLab, AWS SageMaker).
- Creación de tablero para monitoreo del modelo (PyCaret, Mlflow).
- Monitoreo de la data (DVC).

De todas maneras, el modelo es sencillo, podría desplegarse tranquilamente en Heroku
