import pandas as pd
import numpy as np
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
    df["alquiler_predicho"] = np.round(model.predict(df_encoded), 0)
    
    return df

def calcular_beneficio(precio_vivienda, ingresos_anuales, seguro_vida, intereses_hipoteca):
    """
    Calcula el beneficio antes de impuestos para una vivienda en alquiler.

    Args:
    precio_vivienda (float): Precio de la vivienda.
    ingresos_anuales (float): Ingresos anuales por alquiler.
    seguro_vida (float): Costo del seguro de vida.
    intereses_hipoteca (float): Intereses anuales de la hipoteca.

    Returns:
    float: Beneficio antes de impuestos.
    """
    # Seguro impago = 4% * ingresos
    # corresponde al 4% de la renta anual
    seguro_impago = 0.04 * ingresos_anuales

    # Seguro hogar = 176,29
    # Fuente: https://selectra.es/seguros/seguros-hogar/precios-seguros-hogar
    seguro_hogar = 176.29

    # IBI = precio_vivienda * 0,4047%
    ibi = precio_vivienda * 0.004047

    # Impuesto basuras = 283
    impuesto_basuras = 283

    # Mantenimiento y comunidad = ingresos_anuales * 10%
    # incluye la comunidad de vecinos. Fuente: https://www.donpiso.com/blog/mantener-piso-vacio-cuesta-2-300-euros-al-ano/
    mantenimiento_comunidad = ingresos_anuales * 0.10

    # Periodos vacío = ingresos_anuales * 5%
    periodos_vacios = ingresos_anuales * 0.05

    # Beneficio = ingresos - seguro impago - seguro basuras - seguro hogar 
    # - seguro vida - IBI - mantenimiento - periodos vacío - intereses hipoteca
    beneficio = (ingresos_anuales - seguro_impago - seguro_hogar - seguro_vida - 
                 ibi - impuesto_basuras - mantenimiento_comunidad - 
                 periodos_vacios - intereses_hipoteca)

    return beneficio


def calcular_rentabilidad_inmobiliaria(porcentaje_entrada, coste_compra, coste_reformas, comision_agencia, 
                                       alquiler_mensual, anios, tin, gastos_anuales, porcentaje_irpf, 
                                       porcentaje_amortizacion):
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
    - Diccionario con las métricas calculadas.
    """
    # Cálculo del ITP (8%) y coste notario (2%)
    coste_itp = coste_compra * 0.08
    coste_notario = coste_compra * 0.02

    # Coste total
    coste_total = coste_compra + coste_reformas + comision_agencia + coste_notario + coste_itp

    # Pago inicial (inversión inicial)
    pago_entrada = porcentaje_entrada * coste_compra

    # Cash necesario para la compra y reforma
    cash_necesario_compra = pago_entrada + comision_agencia + coste_notario + coste_itp
    cash_total_compra_reforma = cash_necesario_compra + coste_reformas

    # Monto del préstamo
    monto_prestamo = coste_total - pago_entrada

    # Pagos mensuales y anuales de la hipoteca (usando fórmula de anualidad)
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

    # Cálculo del beneficio antes de impuestos usando la función calcular_beneficio
    beneficio_antes_impuestos = calcular_beneficio(
        precio_vivienda=coste_compra,
        ingresos_anuales=alquiler_anual,
        seguro_vida=gastos_anuales,
        intereses_hipoteca=hipoteca_mensual * 12
    )

    # IRPF aplicado a larga duración
    irpf_larga_duracion = beneficio_antes_impuestos * (porcentaje_irpf / 100)

    # Deducción por larga duración (60%)
    deduccion_larga_duracion = beneficio_antes_impuestos * 0.60

    # Beneficio neto
    beneficio_neto = beneficio_antes_impuestos - irpf_larga_duracion + deduccion_larga_duracion

    # Rentabilidad bruta
    rentabilidad_bruta = alquiler_anual / coste_total

    # Rentabilidad neta
    rentabilidad_neta = beneficio_neto / coste_total

    # Cashflow antes de impuestos
    cashflow_antes_impuestos = alquiler_anual - (gastos_anuales + hipoteca_mensual * 12)

    # Cashflow después de impuestos
    cashflow_despues_impuestos = cashflow_antes_impuestos - irpf_larga_duracion + deduccion_larga_duracion

    # ROCE (Return on Capital Employed)
    roce = beneficio_neto / cash_necesario_compra

    # Cash-on-Cash Return (COCR)
    cash_on_cash_return = cashflow_despues_impuestos / cash_total_compra_reforma

    # Resultados finales
    return {
        "Coste Total": coste_total,
        "Rentabilidad Bruta": rentabilidad_bruta,
        "Beneficio Antes de Impuestos": beneficio_antes_impuestos,
        "Rentabilidad Neta": rentabilidad_neta,
        "Cuota Mensual Hipoteca": hipoteca_mensual,
        "Cash Necesario Compra": cash_necesario_compra,
        "Cash Total Compra y Reforma": cash_total_compra_reforma,
        "Beneficio Neto": beneficio_neto,
        "Cashflow Antes de Impuestos": cashflow_antes_impuestos,
        "Cashflow Después de Impuestos": cashflow_despues_impuestos,
        "ROCE": roce,
        "ROCE (Años)": roce * anios,
        "Cash-on-Cash Return": cash_on_cash_return,
        "COCR (Años)": cash_on_cash_return * anios
    }