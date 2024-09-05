import os
import sys
import pandas as pd
import warnings
import logging
from datetime import datetime
from sqlalchemy import MetaData, text
from flask_sqlalchemy import SQLAlchemy
from joblib import load

# Agregar rutas necesarias al path
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "MD_LIB"))

# Importar configuraciones y módulos personalizados
from settings import *
from sql_sentences_VIGIA import *
from preprocesado import *

# Inicializar metadatos y conexión a la base de datos
md_meta = MetaData()
db = SQLAlchemy()

# Ignorar advertencias
warnings.filterwarnings('ignore')

# Función para ejecutar sentencias SQL
def run_sql(sql_id, cnx=None):
    cnx = cnx or DB_CNX.engine
    sql_sentence = sql_id.strip()

    if not sql_sentence.startswith("SELECT"):
        logging.warning("Sentencia SQL no válida")
        return None, BAD_REQUEST

    try:
        df = pd.read_sql_query(text(sql_sentence), cnx)
        logging.info(f"Operación realizada correctamente. {df.shape[0]} filas cargadas")
        return df, OK
    except Exception as e:
        logging.warning(f"Error en sentencia SQL. {str(e)}")
        return None, NOT_FOUND

# Función para obtener y procesar el dataset
def get_dataset():
    dataset, status = run_sql(sql_predictivo_produccion)
    return limpieza_final(dataset), dataset.copy()

# Clase para gestionar predicciones
class Predicciones:
    def __init__(self, **kwargs):
        self.df = kwargs.get('df')
        self.df_completo = kwargs.get('df_completo')
        self.model_name = kwargs.get('model_name')
        self.fpath = settings['file_download_path']
        self.fpath_modelo = f"{self.fpath}/{self.model_name}" if self.model_name else None
        self.ruta_anotaciones = f"{self.fpath}/{self.model_name}/Anotaciones.txt" if self.model_name else None

        logging.info('--------- Lanzando Predicciones --------- ')
        self.modelo = load(f"{self.fpath_modelo}/modelo_entrenado.joblib")
        self.obtener_prediccion()

    def obtener_prediccion(self):
        self.X = self.df

        if not hasattr(self.modelo, "predict_proba"):
            raise AttributeError("El modelo no tiene el método predict_proba.")

        self.y_pred = self.modelo.predict(self.X)
        prob_class_1 = self.modelo.predict_proba(self.X)[:, 1]
        
        results = pd.DataFrame({
            'ID': self.X.index,
            'predicted_value': self.y_pred,
            'PROBABILIDAD': prob_class_1
        }).query('predicted_value == 1')

        results = results.merge(self.df_completo[['ID', 'PON', 'CTO', 'CLIENTES']], on='ID')
        results['FECHA_PREDICCION'] = datetime.now().replace(hour=9, minute=0, second=0)
        results = results.sort_values(by='PROBABILIDAD', ascending=False)

        grouped_results = results.groupby('PON', as_index=False).agg({'PROBABILIDAD': 'mean', 'CLIENTES': 'mean'})
        grouped_results = grouped_results.query('PROBABILIDAD > 0.900 & CLIENTES >= 4').nlargest(30, 'PROBABILIDAD')
        pones_top = set(grouped_results['PON'])
        results = results[results['PON'].isin(pones_top)]

        if results.empty:
            logging.info("No hay predicciones por encima del 90% de confianza")
            return

        # Preprocesar para inserción en BBDD
        results.fillna({'categoricas': 'NoDisponible', 'numericas': 0}, inplace=True)
        predicciones_activas, _ = run_sql(sql_predicciones_5dias)
        results = results[~results['CTO'].isin(predicciones_activas['CTO'])]

        if results.empty:
            logging.info("No hay predicciones nuevas en el dia de hoy")
            return

        self.insertar_en_bbdd(results, 'vigia_mp_predictivo')
        results.drop(columns=['PROBABILIDAD', 'ID_PRED'], inplace=True)
        self.insertar_en_bbdd(results, 'BO_ms', adicional=True)

    def insertar_en_bbdd(self, results, tabla, adicional=False):
        if adicional:
            results.rename(columns={'CTO': 'ELEMENTO_ALARMADO', 'FECHA_PREDICCION': 'FECHA'}, inplace=True)
            results['FECHA_ALTA'] = results['FECHA']
            results['KPI'] = 'KPI#1#5'
            results['ACCION_EN_CURSO'] = 'PREDICTIVO'
            results['ESTADO'] = 'PENDIENTE OID'

        columns = ', '.join(results.columns)
        values_str = ', '.join(
            f"('{v.strftime('%Y-%m-%d %H:%M:%S')}' if isinstance(v, datetime) else f'\"{v}\"')" 
            for v in results.itertuples(index=False, name=None)
        )
        sql_insert = f"INSERT INTO VIGIA.dbo.{tabla} ({columns}) VALUES {values_str}"

        conn = DB_CNX.session
        try:
            conn.execute(text(sql_insert))
            conn.commit()
            logging.info(f"Completado proceso de insert en {tabla}")
        except Exception as e:
            logging.error(f"Error en inserción en {tabla}: {str(e)}")

if __name__ == "__main__":
    settings = set_context(path=os.path.abspath(__file__))
    db_settings = settings.get("dsn", {}).get("VIGIA")
    if db_settings is None:
        raise Exception("Configuración incorrecta (DB Settings)")

    DB_CNX = set_meta_db(db_settings, md_meta)

    logging.info('--------------   INICIO DEL PROCESO   -------------------')

    dataset, dataset_completo = get_dataset()
    Predicciones(df=dataset, df_completo=dataset_completo, model_name='RANDOM_FOREST')

    # Para ejecutar con varios modelos
    # for model_name in ('RANDOM_FOREST', 'EXTRA_TREES'):
    #     Predicciones(df=dataset, df_completo=dataset_completo, model_name=model_name)