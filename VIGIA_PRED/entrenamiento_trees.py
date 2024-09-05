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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, make_scorer
from sklearn.tree import export_graphviz
import xgboost as xgb
# import lightgbm as lgb
# Case REGLOGISTICA
# from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import shap

#import graphviz
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

    dataset_con_av, status = run_sql(sql_pred_con)
    dataset_sin_av, status = run_sql(sql_pred_sin)

    # Selección aleatoria de NO AVERIAS (random_state=42 - 42 es ejemplo para reproductibilidad)
    dataset_sin_av = dataset_sin_av.sample(n=3000)


    df = pd.concat([dataset_con_av, dataset_sin_av])
    df = limpieza_final(df)


    return df



'''
###################################################################################################################
# CLASE PARA ENTRENAMIENTO DE MODELOS DE ENSAMBLE DE ÁRBOLES Y VISUALIZACIONES
###################################################################################################################
'''


class Entrenamiento:
    def __init__(self, **kwargs):
        self.df = kwargs.get('df', None)
        self.model_name = kwargs.get('model_name', None)
        self.fpath = settings['file_download_path']
        self.fpath_modelo = f"{self.fpath}/{self.model_name}" if self.model_name else None
        self.num_registros = len(self.df) if self.df is not None else None
        self.ruta_anotaciones = f"{self.fpath}/{self.model_name}/Anotaciones.txt" if self.model_name else None
        # self.class_weights = kwargs.get('class_weights', None)
        self.param_grid = kwargs.get('param_grid', None)
        self.columna_objetivo = kwargs.get('columna_objetivo', None)

        # Inicio del entrenamiento y las visualizaciones
        self.entrenar()
        self.visualizar_variables_imp()
        self.visualizar_matriz_conf()
        self.guardar_modelos_entrenados()


    ###################################################################################################################
    # ENTRENAMIENTO
    ###################################################################################################################

    def entrenar(self):


        # Definición de Predictoras y variable objetivo
        self.X = self.df.drop(columns=[self.columna_objetivo])
        self.y = self.df[self.columna_objetivo]

        # División de los datos en conjuntos de entrenamiento y prueba (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Distintos modelos de ensamble de árboles
        if self.model_name == 'RANDOM_FOREST':
            model_classifier = RandomForestClassifier()
        elif self.model_name == 'EXTRA_TREES':
            model_classifier = ExtraTreesClassifier()


        with open(self.ruta_anotaciones, "w") as f:
            f.write(f"\n--------------   INICIO DEL ENTRENAMIENTO CON {self.model_name}   -------------------\n")
            f.write(f"Hyperparámetros probados: {self.param_grid} \n")

        grid_search = GridSearchCV(estimator=model_classifier, param_grid=self.param_grid, scoring='precision')

        fecha_y_hora_inicio_train = datetime.now().replace(microsecond=0)

        logging.info(f"------------- Iniciando Entrenamiento con {self.model_name} -----------------")

        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_

        logging.info(f"Mejores hiperparametros: {best_params}")
        with open(self.ruta_anotaciones, "a") as f:
            f.write(f"Mejores hiperparametros: {best_params}\n")

        self.best_model = grid_search.best_estimator_

        fecha_y_hora_fin_train = datetime.now().replace(microsecond=0)

        with open(self.ruta_anotaciones, "a") as f:
            f.write(f"INICIO DEL ENTRENAMIENTO: {fecha_y_hora_inicio_train}\n")
            f.write(f"FIN DEL ENTRENAMIENTO: {fecha_y_hora_fin_train}\n")


    ###################################################################################################################
    # VISUALIZACION VARIABLES MÁS IMPORTANTES
    ###################################################################################################################

    def visualizar_variables_imp(self):
        importances = self.best_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices[:20]], y=self.X.columns[indices[:20]])
        plt.title(f"{self.model_name}: Las 20 variables más importantes")
        plt.xlabel("Importancia")
        plt.ylabel("Variable")

        ruta_VarImport = f"{self.fpath_modelo}/ImportanciaVariables.png"
        plt.savefig(ruta_VarImport, format='png')
        plt.close()
        logging.info(f"Grafico de variables importantes generado.")

        # Guardar las variables más importantes en un archivo
        top_variables = self.X.columns[indices[:20]]
        with open(f"{self.fpath_modelo}/variables_importantes.txt", "w") as f:
            for variable in top_variables:
                f.write(f"{variable}\n")


    ###################################################################################################################
    # VISUALIZACION MATRIZ DE CONFUSION
    ###################################################################################################################
    def visualizar_matriz_conf(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Conjunto de entrenamiento
        self.best_model.fit(self.X_train, self.y_train)
        y_train_pred = self.best_model.predict(self.X_train)
        conf_matrix_train = confusion_matrix(self.y_train, y_train_pred)

        sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.xaxis.set_label_position('top')
        ax.set_title(f"{self.model_name} - Matriz de Confusión (Train)")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor Real")

        plt.tight_layout()
        plt.savefig(f"{self.fpath_modelo}/MC_train.png", format='png')
        plt.close()
        logging.info(f"Matriz de confusion para train generada.")


    ###################################################################################################################
    # GUARDAMOS MODELOS ENTRENADOS EN DIRECTORIO
    ###################################################################################################################

    def guardar_modelos_entrenados(self):

        # Guardar el modelo en un directorio específico
        dump(self.best_model, f"{self.fpath_modelo}/modelo_entrenado.joblib")
        logging.info(f"{self.model_name}. Modelo entrenado y guardado.")


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

    dataset = get_dataset()


    ###################################################################################################################
    # DEFINICIÓN DE BÚSQUEDA DE HIPERPARÁMETROS
    ###################################################################################################################

    # PARÁMETROS REQUERIDOS PARA LOS MODELOS
    # class_weights = {0: 1, 1: 1}
    param_grid = {
        'n_estimators': [40, 50],
        'max_depth': [4, 5, 6]
    }
    columna_objetivo = 'AVERIAS_PEX_SIN_NIVEL_SUP'


    ###################################################################################################################
    # LANZAMIENTO DE MODELOS Y SU ENTRENAMIENTO
    ###################################################################################################################

    lista_modelos = ('RANDOM_FOREST', 'EXTRA_TREES')
    
    for model_name in lista_modelos:
        Entrenamiento (df=dataset, model_name=model_name, columna_objetivo=columna_objetivo, param_grid=param_grid)


    # OPCIÓN PARA PROBAR UN SOLO MODELO
    '''
    Entrenamiento (df=dataset, model_name='REG_LOG', columna_objetivo=columna_objetivo, param_grid=param_grid)
    '''







