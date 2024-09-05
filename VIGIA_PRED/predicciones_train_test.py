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
import xgboost as xgb
import graphviz
from joblib import dump, load

from sklearn.tree import plot_tree
import seaborn as sns
import matplotlib.pyplot as plt


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

    dataset_con_av, status = run_sql(sql_pred_con_PREDECIR)
    dataset_sin_av, status = run_sql(sql_pred_sin_PREDECIR)

    df = pd.concat([dataset_con_av, dataset_sin_av])

    df_completo = df.copy()

    df = limpieza_final(df)


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
        self.columna_objetivo = kwargs.get('columna_objetivo', None)


        logging.info('--------- Lanzando Predicciones --------- ')


        # Cargar el modelo
        self.modelo = load(f"{self.fpath_modelo}/modelo_entrenado.joblib")
        self.obtener_prediccion()



    def obtener_prediccion(self):

        ###########################################################################################################
        # LANZAMIENTO DE PREDICCIONES CON ALGORITMOS ENTRENADOS PREVIAMENTE
        ###########################################################################################################

        # Definición de las características y la variable objetivo
        self.X = self.df.drop(columns=[self.columna_objetivo])
        self.y = self.df[self.columna_objetivo]

        # Verifica si el modelo tiene el método predict_proba
        if hasattr(self.modelo, "predict_proba"):
            self.probabilidades_predichas = self.modelo.predict_proba(self.X)
        else:
            raise AttributeError("El modelo no tiene el método predict_proba.")

        # También puedes obtener las etiquetas predichas
        self.y_pred = self.modelo.predict(self.X)

        # Creación de una Serie de predicciones
        predictions = pd.Series(data=self.y_pred, index=self.X.index, name='predicted_value')

        # Creación de un DataFrame de probabilidades solo para la clase 1
        target_map = {v: k for k, v in enumerate(self.modelo.classes_)}
        prob_class_1 = self.probabilidades_predichas[:, target_map[1]]  # Probabilidad para la clase 1
        probabilities = pd.DataFrame(data=prob_class_1, index=self.X.index, columns=['probability_for_class_1'])

        # Leer las variables importantes desde el archivo
        with open(f"{self.fpath_modelo}/variables_importantes.txt", "r") as f:
            variables_importantes = [line.strip() for line in f.readlines() if line.strip()]



        ###########################################################################################################
        # CONSTRUCCIÓN DEL DATAFRAME CON PREDICCIONES Y ALGUNAS VARIABLES DE INTERÉS
        ###########################################################################################################

        # variables_importantes = ['PORCEN_OCUPACION', 'CLIENTES', 'TOTAL_GESCAL37',
        #                          'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
        #                          'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']

        variables_importantes = ['PORCEN_OCUPACION', 'CLIENTES', 'TOTAL_GESCAL37',
                                 'P1', 'P2', 'P3', 'P4', 'P5']

        # Filtrar el DataFrame self.X con las variables importantes
        self.X = self.X[variables_importantes]

        # Construcción del DataFrame con los resultados
        results = self.X.join(predictions, how='left')
        results = results.join(probabilities, how='left')
        results = results.join(self.y, how='left')
        results = results.rename(columns={self.columna_objetivo: 'AVERIAS_PEX_OBJETIVO'})


        # Reordenar las columnas para tener las variables de interés al principio
        column_order = ['predicted_value', 'probability_for_class_1', 'AVERIAS_PEX_OBJETIVO'] + list(self.X.columns)
        results = results[column_order]

        # Filtrar el DataFrame para quedarse solo con las filas donde 'probability_for_class_1' = 1
        results = results.loc[results['predicted_value'] == 1]


        # Asegúrate de que el índice de 'results' sea una columna normal llamada 'ID'
        results.reset_index(inplace=True)

        # Realiza el merge para añadir las columnas 'PON' y 'CTO' de 'df_completo' a 'results'
        results = results.merge(self.df_completo[['ID', 'PON', 'CTO']], on='ID', how='left')

        # Ordenar el DataFrame 'results' por 'probability_for_class_1' en orden ascendente
        results = results.sort_values(by='probability_for_class_1', ascending=False)

        results.rename(columns={'probability_for_class_1': 'PROBABILIDAD'}, inplace=True)

        results['FECHA_PREDICCION'] = datetime.now().date()

        # Reordenar las columnas de 'results'
        column_order = ['FECHA_PREDICCION', 'CTO', 'PON'] + [col for col in results.columns if
                                                            col not in ['FECHA_PREDICCION', 'CTO', 'PON']]
        results = results[column_order]

        # Eliminar las columnas 'AVERIAS_PEX_OBJETIVO' y 'predicted_value'
        results.drop(columns=['AVERIAS_PEX_OBJETIVO', 'predicted_value'], inplace=True)

        # Si deseas volver a establecer 'ID' como índice
        results.rename(columns={'ID': 'ID_VIGIA_PRED'}, inplace=True)

        # results.set_index('ID', inplace=True)



        ###########################################################################################################
        # INSERCIÓN EN BBDD
        ###########################################################################################################


        # Convertir los datos del DataFrame en una lista de tuplas
        values = [tuple(x) for x in results.itertuples(index=False, name=None)]

        # Convertir la fecha a una cadena en el formato adecuado
        values_str = ', '.join(
            f"('{v[0].strftime('%Y-%m-%d')}', '{v[1]}', '{v[2]}', {v[3]}, {v[4]}, {v[5]}, {v[6]}, {v[7]}, {v[8]}, {v[9]}, {v[10]}, {v[11]}, {v[12]})"
            for v in values
        )

        # Definir la estructura de la query de inserción
        columns = ', '.join(results.columns)
        sql_insert = f"INSERT INTO VIGIA.dbo.vigia_mp_predictivo ({columns}) VALUES {values_str}"


        conn = DB_CNX.session
        executionError = False

        try:
            conn.execute(text(sql_insert))
            conn.commit()
            logging.info(f"Completado proceso de insert")

        except Exception as e:
            logging.error(f"Error en insercion")
            executionError = True


        # Guardar los resultados en CSV con separador ;
        # results.to_csv(f"{self.fpath_modelo}/resultados.csv", index_label='ID', sep=';')




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

    lista_modelos = ('RANDOM_FOREST', 'EXTRA_TREES')
    columna_objetivo = 'AVERIAS_PEX_SIN_NIVEL_SUP'
    for model_name in lista_modelos:
        Predicciones (df=dataset, df_completo= dataset_completo, model_name=model_name, columna_objetivo=columna_objetivo)


    # OPCIÓN PARA PROBAR UN SOLO MODELO
    '''
    columna_objetivo = 'AVERIAS_PEX_SIN_NIVEL_SUP'
    Predicciones(df=dataset, model_name='RANDOM_FOREST', columna_objetivo=columna_objetivo)
    '''








