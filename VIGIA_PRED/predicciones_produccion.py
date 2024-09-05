import os, sys, pandas as pd
import warnings
from datetime import *

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "MD_LIB"))
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "MD_LIB"))

from settings import *
from sqlalchemy import MetaData, text
from flask_sqlalchemy import SQLAlchemy
import logging

md_meta = MetaData()
db = SQLAlchemy()

from sql_sentences_VIGIA import *
from preprocesado import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.tree import export_graphviz
# import xgboost as xgb
# import graphviz
from joblib import dump, load

# from sklearn.tree import plot_tree
# import seaborn as sns
# import matplotlib.pyplot as plt


'''
###################################################################################################################
# Esta función define el funcionamiento de las queries SQL
# y nos da información de como fué el proceso con cada query.
###################################################################################################################
'''


def run_sql(sql_id, cnx=None):
    cnx = cnx or DB_CNX.engine
    sql_sentence = sql_id.strip()

    # solo se admiten sentencias SQL de consulta
    if not sql_sentence.startswith("SELECT"):
        logging.warning("Sentencia SQL no válida")
        return None, BAD_REQUEST
    try:
        with DB_CNX.engine.connect() as conn:
            df = pd.read_sql_query(text(sql_sentence), conn)
            logging.info(f"Operacion realizada correctamente. {df.shape[0]} filas cargadas")
            return df, OK
    except Exception as e:
        logging.warning(f"Error en sentencia SQL. {str(e)}")
        return None, NOT_FOUND




'''
###################################################################################################################
# get_dataset() extrae el dataset y llama a la función de limpieza final
###################################################################################################################
'''


def get_dataset():

    dataset, status = run_sql(sql_predictivo_produccion)

    df_completo = dataset.copy()

    df = limpieza_final(dataset)


    return df, df_completo





'''
###################################################################################################################
# LANZAMIENTO DE PREDICCIONES
###################################################################################################################
'''
class Predicciones:
    def __init__(self, **kwargs):
        self.df = kwargs.get('df', None)
        self.df_completo = kwargs.get('df_completo', None)
        self.model_name = kwargs.get('model_name', None)
        self.fpath = settings['file_download_path']
        self.fpath_modelo = f"{self.fpath}/{self.model_name}" if self.model_name else None
        self.num_registros = len(self.df) if self.df is not None else None
        self.ruta_anotaciones = f"{self.fpath}/{self.model_name}/Anotaciones.txt" if self.model_name else None



        logging.info('--------- Lanzando Predicciones --------- ')


        # Cargar el modelo
        self.modelo = load(f"{self.fpath_modelo}/modelo_entrenado.joblib")
        self.obtener_prediccion()



    def obtener_prediccion(self):

        ###########################################################################################################
        # LANZAMIENTO DE PREDICCIONES CON ALGORITMOS ENTRENADOS PREVIAMENTE
        ###########################################################################################################

        self.X = self.df

        # Verifica si el modelo tiene el método predict_proba
        if hasattr(self.modelo, "predict_proba"):
            self.probabilidades_predichas = self.modelo.predict_proba(self.X)
        else:
            raise AttributeError("El modelo no tiene el método predict_proba.")

        # También puedes obtener las etiquetas predichas
        self.y_pred = self.modelo.predict(self.X)

        predictions = pd.Series(data=self.y_pred, index=self.X.index, name='predicted_value')

        # Creación de un DataFrame de probabilidades solo para la clase 1
        target_map = {v: k for k, v in enumerate(self.modelo.classes_)}
        prob_class_1 = self.probabilidades_predichas[:, target_map[1]]  # Probabilidad para la clase 1
        probabilities = pd.DataFrame(data=prob_class_1, index=self.X.index, columns=['probability_for_class_1'])



        ###########################################################################################################
        # CONSTRUCCIÓN DEL DATAFRAME CON PREDICCIONES
        ###########################################################################################################

        probabilities = probabilities.reset_index().rename(columns={'index': 'ID'})
        results = probabilities.join(predictions, how='left')


        # Reordenar las columnas para tener las variables de interés al principio
        column_order = ['ID', 'predicted_value', 'probability_for_class_1']
        results = results[column_order]

        # Filtrar el DataFrame para quedarse solo con las filas donde 'probability_for_class_1' = 1
        results = results.loc[results['predicted_value'] == 1]

        # Realiza el merge para añadir las columnas 'PON' y 'CTO' de 'df_completo' a 'results'
        results = results.merge(self.df_completo[['ID', 'PON', 'CTO', 'CLIENTES']], on='ID', how='left')
        results = results.sort_values(by='probability_for_class_1', ascending=False)
        results.rename(columns={'probability_for_class_1': 'PROBABILIDAD'}, inplace=True)


        ########################     CAMPO FECHA     ##############################

        # Obtener la fecha actual y ajustar la hora a 09:00:00
        now = datetime.now()
        fecha_prediccion = datetime(now.year, now.month, now.day, 9, 0, 0)
        results['FECHA_PREDICCION'] = fecha_prediccion

        ###########################################################################

        # Reordenar las columnas de 'results'
        column_order = ['FECHA_PREDICCION', 'CTO', 'PON'] + [col for col in results.columns if
                                                            col not in ['FECHA_PREDICCION', 'CTO', 'PON']]
        results = results[column_order]

        results.drop(columns=['predicted_value'], inplace=True)
        results.rename(columns={'ID': 'ID_PRED'}, inplace=True)
        results = results[results['CLIENTES'] >= 4]


        ###########################################################################################################
        # SELECCIÓN DE LOS 30 PONES CON MAYOR PROBABILIDAD AGREGADA
        ###########################################################################################################

        grouped_results = results.groupby('PON', as_index=False).agg({'PROBABILIDAD': 'mean'})
        grouped_results = grouped_results.sort_values(by='PROBABILIDAD', ascending=False)
        grouped_results = grouped_results[grouped_results['PROBABILIDAD'] > 0.900]
        grouped_results = grouped_results.nlargest(30, 'PROBABILIDAD')


        # Crear un conjunto con los PONES con mayor probabilidad agregada
        pones_top = set(grouped_results['PON'])
        results = results[results['PON'].isin(pones_top)]

        # results = results[results['PROBABILIDAD'] > 0.900]

        if results.empty:
            logging.info(f"No hay predicciones por encima del 90% de confianza")
            logging.info(f"Proceso detenido")
            return

        ###########################################################################################################
        # PREPROCESADO PARA INSERCIÓN EN BBDD
        ###########################################################################################################

        # Obtener todas las columnas de tipo 'object'
        categoricas = results.select_dtypes(include=['object']).columns
        results[categoricas] = results[categoricas].fillna('NoDisponible')

        # Obtener todas las columnas de tipo numérico y rellenar los valores nulos con 0
        numericas = results.select_dtypes(include=['number']).columns
        results[numericas] = results[numericas].fillna(0)



        '''
        ###########################################################################################################
        # COMPROBACIÓN PREDICCIONES ÚLTIMOS 5 DIAS
        ###########################################################################################################
        '''

        predicciones_activas, status = run_sql(sql_predicciones_5dias)

        # Crear un conjunto con las CTOs de predicciones_activas
        ctos_activas = set(predicciones_activas['CTO'])

        # Filtrar 'results' para eliminar las filas con CTOs que ya están en 'predicciones_activas'
        results = results[~results['CTO'].isin(ctos_activas)]

        # Si tras la comprobación no hay CTOS nuevas se para el proceso
        if results.empty:
            logging.info(f"No hay predicciones nuevas en el dia de hoy")
            logging.info(f"Proceso detenido")
            return


        '''
        ###########################################################################################################
        # INSERCIÓN EN BBDD: vigia_mp_predictivo
        ###########################################################################################################
        '''


        # Convertir los datos del DataFrame en una lista de tuplas
        values = [tuple(x) for x in results.itertuples(index=False, name=None)]

        # Convertir la fecha a una cadena en el formato adecuado
        values_str = ', '.join(
            f"('{v[0].strftime('%Y-%m-%d %H:%M:%S')}', '{v[1]}', '{v[2]}', '{v[3]}', '{v[4]}', '{v[5]}')"
            for v in values
        )


        # Definir la estructura de la query de inserción
        columns = ', '.join(results.columns)
        sql_insert = f"INSERT INTO VIGIA.dbo.vigia_mp_predictivo ({columns}) VALUES {values_str}"
        logging.info(f"INSERT CONTRA vigia_mp_predictivo:")
        logging.info(f"{sql_insert}")

        conn = DB_CNX.session
        executionError = False

        try:
            conn.execute(text(sql_insert))
            conn.commit()
            logging.info(f"Completado proceso de insert")

        except Exception as e:
            logging.error(f"Error en insercion")
            executionError = True

        '''
        ###########################################################################################################
        # INSERCIÓN EN BBDD: BO_ms
        ###########################################################################################################
        '''


        # Campos para BO_ms
        results.drop(columns=['PROBABILIDAD', 'ID_PRED'], inplace=True)
        results.rename(columns={'CTO': 'ELEMENTO_ALARMADO'}, inplace=True)
        results.rename(columns={'FECHA_PREDICCION': 'FECHA'}, inplace=True)
        results['FECHA_ALTA'] = results['FECHA']
        results['KPI'] = 'KPI#1#5'
        results['ACCION_EN_CURSO'] = 'PREDICTIVO'
        results['ESTADO'] = 'PENDIENTE OID'

        # Reordenar las columnas de 'results'
        column_order = ['FECHA', 'FECHA_ALTA'] + [col for col in results.columns if col not in ['FECHA', 'FECHA_ALTA']]
        results = results[column_order]


        # Convertir los datos del DataFrame en una lista de tuplas
        values = [tuple(x) for x in results.itertuples(index=False, name=None)]

        # Convertir la fecha a una cadena en el formato adecuado
        values_str = ', '.join(
            f"('{v[0].strftime('%Y-%m-%d %H:%M:%S')}', '{v[1].strftime('%Y-%m-%d %H:%M:%S')}', '{v[2]}', '{v[3]}', '{v[4]}', '{v[5]}', '{v[6]}', '{v[7]}')"
            for v in values
        )


        # Definir la estructura de la query de inserción
        columns = ', '.join(results.columns)
        sql_insert = f"INSERT INTO VIGIA.dbo.BO_ms ({columns}) VALUES {values_str}"
        logging.info(f"INSERT CONTRA BO_ms:")
        logging.info(f"{sql_insert}")

        conn = DB_CNX.session
        executionError = False

        try:
            conn.execute(text(sql_insert))
            conn.commit()
            logging.info(f"Completado proceso de insert")

        except Exception as e:
            logging.error(f"Error en insercion")
            executionError = True





if __name__ == "__main__":
    # Settings the warnings to be ignored
    warnings.filterwarnings('ignore')

    settings = set_context(path=os.path.abspath(__file__))
    URL_HOST = settings.get("host", "localhost")
    URL_PORT = settings.get("port", 5000)
    db_settings = settings.get("dsn", {}).get("VIGIA")
    if db_settings == None:
        raise Exception("Configuración incorrecta (DB Settings)")

    #configura la conexión al servidor de Base de Datos.
    #La cadena de conexión se describe en la clave "dsn.[DSN_NAME]" del archivo settings.config
    DB_CNX = set_meta_db(db_settings, md_meta)





    logging.info('--------------   INICIO DEL PROCESO   -------------------')


    ###################################################################################################################
    # EXTRACCIÓN DEL DATASET Y PREPROCESADO
    ###################################################################################################################

    dataset, dataset_completo = get_dataset()


    ###################################################################################################################
    # LANZAMIENTO DE PREDICCIONES CON MODELOS YA ENTRENADOS
    ###################################################################################################################

    model_name = 'RANDOM_FOREST'
    Predicciones (df=dataset, df_completo=dataset_completo, model_name=model_name)


    # POR SI SE QUIEREN EJECUTAR 2 MODELOS DISTINTOS
    # >>>>> Esta opción insertaría 2 veces en vigia_mp_predictivo y en BO_ms
    '''
    lista_modelos = ('RANDOM_FOREST', 'EXTRA_TREES')
    for model_name in lista_modelos:
        Predicciones (df=dataset, df_completo= dataset_completo, model_name=model_name)
    '''




