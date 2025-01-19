import pandas as pd
import pickle

def predecir_alquiler(df_path, transformer_paths):
    """
    Carga los archivos necesarios, procesa los datos y predice la columna 'alquiler_predicho'.

    Parámetros:
        df_path (str): Ruta del archivo pickle que contiene el DataFrame de entrada.
        transformer_paths (list): Lista con las rutas de los transformadores [scaler, encoder, modelo].

    Retorna:
        pd.DataFrame: El DataFrame con la columna 'alquiler_predicho' añadida.
    """
    # Diccionario para mapear valores en la columna 'planta'
    dicc_sust_planta = {'st': -3, 'ss': -2, 'bj': -1, 'en': 0.5, 'ND': 0}

    # Lista de columnas booleanas
    lista_col_bools = ["ascensor", "exterior", "aire_acondicionado", "trastero", "terraza", "patio"]

    # Cargar el DataFrame
    df = pd.read_pickle(df_path)
    
    # Cargar transformadores y modelo desde las rutas proporcionadas
    encoder_path, scaler_path, model_path = transformer_paths
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Seleccionar características
    features = df[list(encoder.get_feature_names())]
    
    # Aplicar codificación
    df_encoded = encoder.transform(features)
    
    # Mapear valores de 'planta'
    df_encoded['planta'] = features['planta'].map(dicc_sust_planta).fillna(features['planta'])
    
    # Procesar columnas booleanas
    for columna in lista_col_bools:
        df_encoded[columna] = features[columna].fillna('ND').astype(str).replace({"True": 1, "False": -1, "ND": 0})
    
    # Escalar la columna 'tamanio'
    df_encoded['tamanio'] = scaler.transform(features[['tamanio']])
    
    # Predecir alquiler y añadir la columna al DataFrame original
    df["alquiler_predicho"] = model.predict(df_encoded)
    
    return df

def calcular_rentabilidad_inmobiliaria(porcentaje_entrada, coste_compra, coste_reformas, comision_agencia, 
                                       porcentaje_coste_notario, coste_impuestos, alquiler_mensual, anios, tin, 
                                       gastos_mensuales, deducciones):
    """
    Función para calcular las métricas de rentabilidad inmobiliaria basadas en los datos proporcionados.
    
    Parámetros:
    - porcentaje_entrada: Porcentaje del coste total cubierto por el pago inicial.
    - coste_compra: Coste total de la compra de la propiedad.
    - coste_reformas: Costes asociados con reformas y reparaciones.
    - comision_agencia: Comisión de la agencia o PSI.
    - porcentaje_coste_notario: Porcentaje del coste de compra para notario y registro.
    - coste_impuestos: Impuestos aplicables a la compra de la propiedad.
    - alquiler_mensual: Ingresos mensuales esperados por alquiler.
    - anios: Duración de la hipoteca en años.
    - tin: Tasa de interés nominal fija anual de la hipoteca.
    - gastos_mensuales: Gastos fijos mensuales asociados con el inmueble.
    - deducciones: Porcentaje estimado de deducciones sobre el ingreso bruto anual.

    Devuelve:
    - Un diccionario con las métricas financieras calculadas.
    """
    # Coste total
    coste_notario = porcentaje_coste_notario * coste_compra
    coste_total = coste_compra + coste_reformas + comision_agencia + coste_notario + coste_impuestos

    # Pago inicial (inversión inicial)
    pago_entrada = porcentaje_entrada * coste_compra

    # Cash necesario para compra y reforma
    cash_necesario_compra = pago_entrada + comision_agencia + coste_notario + coste_impuestos
    cash_total_compra_reforma = cash_necesario_compra + coste_reformas

    # Monto del préstamo
    monto_prestamo = coste_total - pago_entrada

    # Pagos mensuales y anuales de la hipoteca (asumiendo interés fijo, usando fórmula de anualidad)
    tasa_interes_mensual = tin / 12
    numero_pagos = anios * 12
    if tasa_interes_mensual > 0:
        hipoteca_mensual = monto_prestamo * (tasa_interes_mensual * (1 + tasa_interes_mensual)**numero_pagos) / \
                           ((1 + tasa_interes_mensual)**numero_pagos - 1)
    else:
        hipoteca_mensual = monto_prestamo / numero_pagos

    # Interés total pagado durante el período del préstamo
    interes_total = hipoteca_mensual * numero_pagos - monto_prestamo

    # Ingresos anuales por alquiler
    alquiler_anual = alquiler_mensual * 12

    # Gastos anuales
    gastos_anuales = gastos_mensuales * 12 + hipoteca_mensual * 12

    # Ingreso neto anual después de deducciones
    ingresos_netos_antes_deducciones = alquiler_anual - gastos_anuales
    ingresos_netos_despues_deducciones = ingresos_netos_antes_deducciones * (1 - deducciones)

    # Cashflow anual y mensual (antes y después de impuestos/deducciones)
    cashflow_ai = ingresos_netos_antes_deducciones  # Antes de impuestos/deducciones
    cashflow_di = ingresos_netos_despues_deducciones  # Después de impuestos/deducciones

    # Rentabilidad sobre el capital invertido (ROCE)
    roce = (ingresos_netos_antes_deducciones / cash_total_compra_reforma) * 100

    # Años para recuperar la inversión inicial
    roce_anios = cash_total_compra_reforma / ingresos_netos_antes_deducciones if ingresos_netos_antes_deducciones > 0 else None

    # Cash-on-Cash Return (COCR)
    cocr = (cashflow_di / cash_total_compra_reforma) * 100

    return {
        "Coste Total": coste_total,
        "Pago Entrada": pago_entrada,
        "Monto Préstamo": monto_prestamo,
        "Hipoteca Mensual": hipoteca_mensual,
        "Interés Total": interes_total,
        "Alquiler Anual": alquiler_anual,
        "Ingresos Netos Antes Deducciones": ingresos_netos_antes_deducciones,
        "Ingresos Netos Después Deducciones": ingresos_netos_despues_deducciones,
        "Cash Necesario para Compra": cash_necesario_compra,
        "Cash Total Compra y Reforma": cash_total_compra_reforma,
        "Cashflow AI": cashflow_ai,
        "Cashflow DI": cashflow_di,
        "ROCE": roce,
        "ROCE Años": roce_anios,
        "COCR": cocr
    }
