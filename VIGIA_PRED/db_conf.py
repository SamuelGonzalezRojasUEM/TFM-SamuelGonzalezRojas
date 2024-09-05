"""
db_conf: carga los parámetros de trabajo de la aplicación desde el archivo de configuración
    indicado como argumento en línea de comandos "-conf". default: "settings.config"

"""
import os, sys, pymysql

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "MD_LIB"))
from settings import *


# inicialización de la conexión a BD.
settings = set_context(path=os.path.abspath(__file__))
URL_HOST = settings.get("host", "localhost")
URL_PORT = settings.get("port", 5000)

try:
    db_settings = settings["dsn"]["VIGIA"]
    VIGIA = set_meta_db(db_settings)
    if VIGIA != None:
        logging.info(f"Database backend: {VIGIA.engine.url.host}.VIGIA")
    else:
        logging.error(
            "Error en settings.config: configuración incorrecta de dsn 'VIGIA'"
        )
        exit(-1)
except:
    logging.error("Error en conexión a BD. El programa no puede ejecutarse")
    exit(-2)


