import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler



'''
###################################################################################################################
# limpieza_final(df)    realiza el último tratamiento de limpieza de valores nulos para entrenamiento posterior
###################################################################################################################
'''


def limpieza_final(df):

    ###################################################################################################################
    # VARIABLE OBJETIVO
    ###################################################################################################################

    # ELIMINAMOS VARIABLES QUE DE MOMENTO NO SE VAN A USAR
    columnas_a_eliminar = ['AVERIAS_PEX_OBJETIVO']

    # 'AVERIAS_PEX_SIN_NIVEL_SUP',
    # 'AVERIAS_PEX_OBJETIVO'

    # Eliminar las columnas especificadas
    try:
        df = df.drop(columns=columnas_a_eliminar)
    except:
        pass



    ###################################################################################################################
    # TRATAMIENTO DE ''''''''''NULL''''''''''''''''
    ###################################################################################################################

    # Obtener todas las columnas de tipo 'object'
    categoricas = df.select_dtypes(include=['object']).columns

    # Rellenar los valores nulos en las columnas de tipo 'object' con 'DESCONOCIDO'
    df[categoricas] = df[categoricas].fillna('DESCONOCIDO')

    ###################################################################################################################

    # Obtener todas las columnas de tipo 'object'
    numericas = df.select_dtypes(include=['float']).columns

    # Rellenar los valores nulos en las columnas numéricas con la media de esa columna
    for col in numericas:
        mean_value = df[col].mean()
        df[col] = df[col].fillna(mean_value)

    # Crear un objeto MinMaxScaler
    scaler = MinMaxScaler()

    # Ajustar y transformar las columnas numéricas
    df[numericas] = scaler.fit_transform(df[numericas])



    ###################################################################################################################
    # ELIMINAMOS VARIABLES QUE DE MOMENTO NO SE VAN A USAR

    columnas_a_eliminar = ['PON', 'FECHA_DATOS', 'FECHA_AVERIA', 'CTO',
                           'PROVINCIA', 'CENTRAL', 'POBLACION',
                           'HUELLA', 'SFP_TYPE', 'FECHA_ULTIMA_MODPEX',
                           'FECHA_CARGA_CTO', 'FECHA_ULTIMA_AVERIA_IM',
                           'FECHA_ULTIMO_ACCESO', 'FECHA_ULTIMO_TP', 'FECHA_ULTIMA_AMPLIACION']

    # Eliminar las columnas especificadas
    df = df.drop(columns=columnas_a_eliminar)

    # Eliminar si existe
    columnas_a_eliminar = ['FECHA_ULT_AVER_PEX']

    try:
        df = df.drop(columns=columnas_a_eliminar)
    except:
        pass

    ###################################################################################################################
    # AGLUTINAMOS MANIPULACIONES SOBRE LAS CTO'S

    columns_to_check = ['N_ACCESOS_CR', 'N_TPS', 'N_MODIFICACIONES_PEX', 'N_AMPLIACIONES']
    df['MANIPULACIONES_CTO'] = (df[columns_to_check] > 0).any(axis=1).astype(int)


    ###################################################################################################################


    # ELIMINAMOS VARIABLES  PARA PROBAR MODELO SIMPLE -------- FUTURAS NO ELIMINADAS
    columnas_a_eliminar = ['TIPO_CTO_INT_EXT', 'TIPO_CTO', 'UBICACION_CTO_CATEG',
                           'N_ACCESOS_CR', 'N_TPS', 'N_MODIFICACIONES_PEX', 'N_AMPLIACIONES' ]


    # 'AVERIAS_PEX_LAST30'
    # 'ANTIGUEDAD_CTO',
    # 'N_AVERIAS_IM',
    # 'UBICACION_CTO_CATEG',
    # 'TIPO_CTO',

    # Eliminar las columnas especificadas
    df = df.drop(columns=columnas_a_eliminar)


    ###################################################################################################################

    # redefinimos el índice como ID
    df.set_index('ID', inplace=True)

    ###################################################################################################################

    # Seleccionar las columnas categóricas
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    # Aplicar one-hot encoding a las columnas categóricas
    df = pd.get_dummies(df, columns=columnas_categoricas)
    # COMPROBACION NULL
    desglose_null = df.isnull().sum()


    ###################################################################################################################
    # FILTRO REGISTROS DONDE LA ATENUACIÓN, LA POTENCIA O EL PORCENTAJE DE OCUPACIÓN SEA 0
    ###################################################################################################################

    # Definir una lista con los nombres de las columnas de interés (de A1 a A30)
    columnas_a_filtrar = ['A' + str(i) for i in range(1, 28)]

    # Aplicar la operación de filtrado para las columnas especificadas
    for col in columnas_a_filtrar:
        df = df[df[col] != 0]

    # Definir una lista con los nombres de las columnas de interés (de A1 a A30)
    columnas_a_filtrar = ['P' + str(i) for i in range(1, 28)]

    # Aplicar la operación de filtrado para las columnas especificadas
    for col in columnas_a_filtrar:
        df = df[df[col] != 0]

    # Definir una lista con los nombres de las columnas de interés (de A1 a A30)
    columnas_a_filtrar = ['PORCEN_OCUPACION', 'CLIENTES', 'TOTAL_GESCAL37' ]

    # Aplicar la operación de filtrado para las columnas especificadas
    for col in columnas_a_filtrar:
        df = df[df[col] != 0]

    ###################################################################################################################
    # Columnas calculadas: RESTAS, SUMAS Y DIVISIONES CONTRA A1 Y P1
    ###################################################################################################################

    # Columnas de potencia (P)
    potencia_cols = ['P2', 'P3', 'P5', 'P7', 'P10', 'P15', 'P20']

    # Columnas de atenuación (A)
    atenuacion_cols = ['A2', 'A3', 'A5', 'A7', 'A10', 'A15', 'A20']

    # Restas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} menos P1'] = df[potencia_cols[i]] - df['P1']
        df[f'{atenuacion_cols[i]} menos A1'] = df[atenuacion_cols[i]] - df['A1']

    # Sumas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} mas P1'] = df[potencia_cols[i]] + df['P1']
        df[f'{atenuacion_cols[i]} mas A1'] = df[atenuacion_cols[i]] + df['A1']

    # Divisiones
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} entre P1'] = df[potencia_cols[i]] / df['P1']
        df[f'{atenuacion_cols[i]} entre A1'] = df[atenuacion_cols[i]] / df['A1']


    ###################################################################################################################
    # Columnas calculadas: RESTAS Y DIVISIONES CONTRA A2 Y P2
    ###################################################################################################################

    # Columnas de potencia (P)
    potencia_cols = ['P3', 'P4', 'P5', 'P7', 'P10', 'P15', 'P20']

    # Columnas de atenuación (A)
    atenuacion_cols = ['A3', 'A4', 'A5', 'A7', 'A10', 'A15', 'A20']

    # Restas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} menos P2'] = df[potencia_cols[i]] - df['P2']
        df[f'{atenuacion_cols[i]} menos A2'] = df[atenuacion_cols[i]] - df['A2']

    # Sumas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} mas P2'] = df[potencia_cols[i]] + df['P2']
        df[f'{atenuacion_cols[i]} mas A2'] = df[atenuacion_cols[i]] + df['A2']

    # Divisiones
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} entre P2'] = df[potencia_cols[i]] / df['P2']
        df[f'{atenuacion_cols[i]} entre A2'] = df[atenuacion_cols[i]] / df['A2']

    ###################################################################################################################
    # Columnas calculadas: RESTAS Y DIVISIONES CONTRA A3 Y P3
    ###################################################################################################################

    # Columnas de potencia (P)
    potencia_cols = ['P4', 'P5', 'P7', 'P10', 'P15', 'P20']

    # Columnas de atenuación (A)
    atenuacion_cols = ['A4', 'A5', 'A7', 'A10', 'A15', 'A20']


    # Restas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} menos P3'] = df[potencia_cols[i]] - df['P3']
        df[f'{atenuacion_cols[i]} menos A3'] = df[atenuacion_cols[i]] - df['A3']

    # Sumas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} mas P3'] = df[potencia_cols[i]] + df['P3']
        df[f'{atenuacion_cols[i]} mas A3'] = df[atenuacion_cols[i]] + df['A3']

    # Divisiones
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} entre P3'] = df[potencia_cols[i]] / df['P3']
        df[f'{atenuacion_cols[i]} entre A3'] = df[atenuacion_cols[i]] / df['A3']


    ###################################################################################################################
    # Columnas calculadas: RESTAS Y DIVISIONES CONTRA A4 Y P4
    ###################################################################################################################

    # Columnas de potencia (P)
    potencia_cols = ['P5', 'P7', 'P10', 'P15', 'P20']

    # Columnas de atenuación (A)
    atenuacion_cols = ['A5', 'A7', 'A10', 'A15', 'A20']


    # Restas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} menos P4'] = df[potencia_cols[i]] - df['P4']
        df[f'{atenuacion_cols[i]} menos A4'] = df[atenuacion_cols[i]] - df['A4']

    # Sumas
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} mas P4'] = df[potencia_cols[i]] + df['P4']
        df[f'{atenuacion_cols[i]} mas A4'] = df[atenuacion_cols[i]] + df['A4']

    # Divisiones
    for i in range(1, len(potencia_cols)):
        df[f'{potencia_cols[i]} entre P4'] = df[potencia_cols[i]] / df['P4']
        df[f'{atenuacion_cols[i]} entre A4'] = df[atenuacion_cols[i]] / df['A4']


    ###################################################################################################################
    # Columnas calculadas: AL CUADRADO
    ###################################################################################################################

    for col in ['A1', 'A2', 'A3', 'A4', 'A5']:
        new_col_name = col + '_^2'
        df[new_col_name] = df[col].apply(lambda x: x ** 2)

    for col in ['P1', 'P2', 'P3', 'P4', 'P5']:
        new_col_name = col + '_^2'
        df[new_col_name] = df[col].apply(lambda x: x ** 2)


    ###################################################################################################################
    # Columnas calculadas (CLIENTES / GESCAL / PORCEN_OCUPACION)
    ###################################################################################################################

    df['GESCAL_entre_PorcenOCUP'] = df['TOTAL_GESCAL37'].astype(float) / df['PORCEN_OCUPACION'].astype(float)

    df['CLIENTES_entre_PorcenOCUP'] = df['CLIENTES'].astype(float) / df['PORCEN_OCUPACION'].astype(float)

    df['CLIENTES_entre_GESCAL'] = df['CLIENTES'].astype(float) / df['TOTAL_GESCAL37'].astype(float)


    df['GESCAL_entre_PorcenOCUP'] = df['GESCAL_entre_PorcenOCUP'].astype(float)

    df['CLIENTES_entre_PorcenOCUP'] = df['CLIENTES_entre_PorcenOCUP'].astype(float)

    df['CLIENTES_entre_GESCAL'] = df['CLIENTES_entre_GESCAL'].astype(float)

    # Clientes menos Gescal

    df['CLIENTES_menos_GESCAL'] = df['CLIENTES'].astype(float) - df['TOTAL_GESCAL37'].astype(float)

    df['CLIENTES_menos_GESCAL'] = df['CLIENTES_menos_GESCAL'].astype(float)




    # Bloque de verificación para identificar Null - problematica con calculadas
    '''
    
    # Verificar en qué columnas hay valores NaN
    columnas_con_nan = df.isna().any()

    # Filtrar las columnas que tienen NaN
    columnas_con_nan = columnas_con_nan[columnas_con_nan].index.tolist()

    # Encontrar las posiciones de los NaN en el DataFrame
    nan_positions = df.isna()

    # Convertir las posiciones en un formato más legible
    nan_positions = nan_positions.stack()[nan_positions.stack()]

    # Reiniciar el índice para obtener un DataFrame con filas y columnas con NaN
    nan_positions = nan_positions.reset_index()

    # Renombrar las columnas para mayor claridad
    nan_positions.columns = ['Fila', 'Columna', 'EsNaN']

    # Obtenemos las filas únicas que contienen NaN
    filas_con_nan = nan_positions['Fila'].unique()

    # Filtrar el DataFrame original para incluir solo estas filas
    df_con_nan = df.loc[filas_con_nan]

    '''


    df = df[sorted(df.columns)]

    logging.info('Columnas calculadas.')

    logging.info('Dataset preparado.')


    return df





