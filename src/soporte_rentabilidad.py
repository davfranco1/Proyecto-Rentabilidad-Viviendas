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
                                      alquiler_mensual, anios, tin, gastos_anuales, porcentaje_irpf, porcentaje_amortizacion):
    """
    Función para calcular las métricas de rentabilidad inmobiliaria basadas en los datos proporcionados.

    Parámetros:
    - porcentaje_entrada: Porcentaje del coste total cubierto por el pago inicial.
    - coste_compra: Coste total de la compra de la propiedad.
    - coste_reformas: Costes asociados con reformas y reparaciones.
    - comision_agencia: Comisión de la agencia o PSI.
    - alquiler_mensual: Ingresos mensuales esperados por alquiler.
    - anios: Duración de la hipoteca en años.
    - tin: Tasa de interés nominal fija anual de la hipoteca.
    - gastos_anuales: Gastos fijos anuales asociados con el inmueble.
    - porcentaje_irpf: Porcentaje aplicado para calcular el IRPF.
    - porcentaje_amortizacion: Porcentaje anual aplicado para amortización.

    Devuelve:
    - Un diccionario con las métricas financieras calculadas.
    """
    # Cálculo del ITP (8%) y coste notario (2%)
    coste_itp = coste_compra * 0.08
    coste_notario = coste_compra * 0.02

    # Coste total
    coste_total = coste_compra + coste_reformas + comision_agencia + coste_notario + coste_itp

    # Pago inicial (inversión inicial)
    pago_entrada = porcentaje_entrada * coste_compra

    # Cash necesario para compra y reforma
    cash_necesario_compra = pago_entrada + comision_agencia + coste_notario + coste_itp
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

    # Gastos anuales (incluyen hipoteca anual)
    gastos_totales_anuales = gastos_anuales + hipoteca_mensual * 12

    # Beneficio antes de impuestos
    beneficio_antes_impuestos = alquiler_anual - gastos_totales_anuales

    # IRPF aplicado a larga duración
    irpf_larga_duracion = beneficio_antes_impuestos * (porcentaje_irpf / 100)

    # Deducción por larga duración
    deduccion_larga_duracion = (beneficio_antes_impuestos - irpf_larga_duracion) * 0.04

    # Rentabilidad bruta
    rentabilidad_bruta = alquiler_anual / coste_total

    # Amortización anual
    amortizacion_anual = coste_total * (porcentaje_amortizacion / 100)

    return {
        "Coste Total": coste_total,
        "Pago Entrada": pago_entrada,
        "Monto Préstamo": monto_prestamo,
        "Hipoteca Mensual": hipoteca_mensual,
        "Interés Total": interes_total,
        "Alquiler Anual": alquiler_anual,
        "Gastos Anuales Totales": gastos_totales_anuales,
        "Beneficio Antes de Impuestos": beneficio_antes_impuestos,
        "IRPF Larga Duración": irpf_larga_duracion,
        "Deducción Larga Duración": deduccion_larga_duracion,
        "Rentabilidad Bruta": rentabilidad_bruta,
        "Amortización Anual": amortizacion_anual
    }
