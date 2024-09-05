

# AVERIAS_PEX_OBJETIVO
# AVERIAS_PEX_SIN_NIVEL_SUP


###################################################################################################################
    # QUERIES PARA ENTRENAMIENTO
###################################################################################################################

sql_pred_con = '''
SELECT * 
  FROM [VIGIA].[dbo].[vigia_mp_dataset_historico] 
    WHERE AVERIAS_PEX_SIN_NIVEL_SUP = 1
    AND FECHA_DATOS < DATEADD(DAY, DATEDIFF(DAY, 0, GETDATE())-2, 0)
  '''

sql_pred_sin = '''
 SELECT TOP 30000 * 
  FROM [VIGIA].[dbo].[vigia_mp_dataset_historico] 
    WHERE AVERIAS_PEX_SIN_NIVEL_SUP = 0
    AND FECHA_DATOS < DATEADD(DAY, DATEDIFF(DAY, 0, GETDATE())-2, 0)
'''





###################################################################################################################
    # QUERIES PARA LANZAMIENTO DE PREDICCIONES TRAIN-TEST
###################################################################################################################

sql_pred_con_PREDECIR = '''
SELECT * 
  FROM [VIGIA].[dbo].[vigia_mp_dataset_historico] 
    WHERE AVERIAS_PEX_SIN_NIVEL_SUP = 1
    AND FECHA_DATOS = DATEADD(DAY, DATEDIFF(DAY, 0, GETDATE())-1, 0)
  '''

sql_pred_sin_PREDECIR = '''
SELECT top 500 * 
  FROM [VIGIA].[dbo].[vigia_mp_dataset_historico] 
    WHERE AVERIAS_PEX_OBJETIVO = 0
    AND FECHA_DATOS = DATEADD(DAY, DATEDIFF(DAY, 0, GETDATE())-1, 0)
'''




###################################################################################################################
    # QUERY PARA:          >>>>>>     PREDICCIONES EN PRODUCCIÓN    <<<<<<<
###################################################################################################################

sql_predictivo_produccion = '''
SELECT * 
  FROM [VIGIA].[dbo].[vigia_mp_dataset_produccion] 
  '''






###################################################################################################################
    # RECUPERACIÓN DE PREDICCIONES DE LOS ÚLTIMOS 5 DÍAS
###################################################################################################################

sql_predicciones_5dias = '''
SELECT CTO 
FROM [VIGIA].[dbo].[vigia_mp_predictivo]
WHERE FECHA_PREDICCION >= DATEADD(DAY, DATEDIFF(DAY, 0, GETDATE()) - 5, 0) 
'''




