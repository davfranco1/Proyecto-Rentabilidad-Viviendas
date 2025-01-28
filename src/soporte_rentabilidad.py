import pandas as pd
import numpy as np
import numpy_financial as npf
import pickle

def predecir_alquiler(df, transformer_paths):
    """
    Carga los archivos necesarios, procesa los datos y predice la columna 'alquiler_predicho'.

    Parámetros:
        df: El DataFrame de entrada.
        transformer_paths (list): Lista con las rutas de los transformadores [scaler, encoder, modelo].

    Retorna:
        pd.DataFrame: El DataFrame con la columna 'alquiler_predicho' añadida.
    """
    # Diccionario para mapear valores en la columna 'planta'
    dicc_sust_planta = {'st': -3, 'ss': -2, 'bj': -1, 'en': 0.5, 'ND': 0}

    # Lista de columnas booleanas
    lista_col_bools = ["ascensor", "exterior", "aire_acondicionado", "trastero", "terraza", "patio"]
    
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
                                       alquiler_mensual, anios, tin, seguro_vida, tipo_irpf, 
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
    - seguro_vida: Seguro de vida del propietario.
    - tipo_irpf: Porcentaje aplicado para calcular el IRPF.
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
    cash_total_compra_reforma = pago_entrada + coste_reformas + coste_notario + coste_itp
    
    # Monto del préstamo
    monto_prestamo = coste_compra*(1-porcentaje_entrada)

    # Pagos mensuales y anuales de la hipoteca
    
    hipoteca_mensual = npf.pmt(tin/12, anios*12, monto_prestamo)

    total_pagado = -hipoteca_mensual * (anios*12) 
    interes_total = total_pagado - monto_prestamo 
    capital_anual = monto_prestamo/anios
    capital_mensual = capital_anual/12
    interes_anual = interes_total / anios

    # Ingresos anuales por alquiler
    alquiler_anual = alquiler_mensual * 12

    # Cálculo del beneficio antes de impuestos usando la función calcular_beneficio
    beneficio_antes_impuestos = calcular_beneficio(
        precio_vivienda=coste_compra,
        ingresos_anuales=alquiler_anual,
        seguro_vida=seguro_vida,
        intereses_hipoteca=interes_anual
    )

    # Amortización anual
    amortizacion_anual = 0.03*(porcentaje_amortizacion*coste_compra+(coste_reformas+comision_agencia+coste_notario+coste_itp))

    # Deducción por larga duración (60%)
    deduccion_larga_duracion = (beneficio_antes_impuestos - amortizacion_anual)* 0.60

    # IRPF aplicado a larga duración
    irpf = -(deduccion_larga_duracion * tipo_irpf)

    # Beneficio neto
    beneficio_neto = beneficio_antes_impuestos + irpf

    # Rentabilidad bruta
    rentabilidad_bruta = alquiler_anual / coste_total * 100

    # Rentabilidad neta
    rentabilidad_neta = (beneficio_antes_impuestos + irpf) / coste_total * 100

    # Cashflow antes de impuestos
    cashflow_antes_impuestos = beneficio_antes_impuestos - capital_anual

    # Cashflow después de impuestos
    cashflow_despues_impuestos = beneficio_neto - capital_anual

    # ROCE (Return on Capital Employed)
    roce = alquiler_anual/(pago_entrada+coste_reformas+comision_agencia+coste_notario+coste_itp) * 100

    # ROCE Años (Return on Capital Employed)
    roce_anios = pago_entrada/(pago_entrada*roce) * 100

    # Cash-on-Cash Return (COCR)
    cash_on_cash_return = cashflow_despues_impuestos/(pago_entrada+coste_reformas+comision_agencia+coste_notario+coste_itp) * 100

    # Cash-on-Cash Return Años (COCR)
    cash_on_cash_return_anios = (pago_entrada+coste_reformas+comision_agencia+coste_notario+coste_itp) / cashflow_despues_impuestos


    # Resultados finales
    return {
        "Coste Total": coste_total,
        "Rentabilidad Bruta": round(rentabilidad_bruta, 2),
        "Beneficio Antes de Impuestos": round(beneficio_antes_impuestos,2),
        "Rentabilidad Neta": round(rentabilidad_neta,2),
        "Cuota Mensual Hipoteca": round(hipoteca_mensual,2),
        "Cash Necesario Compra": round(cash_necesario_compra,2),
        "Cash Total Compra y Reforma": round(cash_total_compra_reforma,2),
        "Beneficio Neto": round(beneficio_neto,2),
        "Rentabilidad Neta": round(rentabilidad_neta,2),
        "Cashflow Antes de Impuestos": round(cashflow_antes_impuestos,2),
        "Cashflow Después de Impuestos": round(cashflow_despues_impuestos,2),
        "ROCE": round(roce,2),
        "ROCE (Años)": round(roce_anios,2),
        "Cash-on-Cash Return": round(cash_on_cash_return,2),
        "COCR (Años)": round(cash_on_cash_return_anios,2)
    }


def calcular_rentabilidad_inmobiliaria_wrapper(df, porcentaje_entrada, coste_reformas, comision_agencia,
                                               anios, tin, seguro_vida, tipo_irpf, 
                                               porcentaje_amortizacion):
    """
    Calcula la rentabilidad inmobiliaria para cada fila de un DataFrame y devuelve un DataFrame final con los resultados.

    Args:
        df (pd.DataFrame): DataFrame con los datos de entrada. Debe contener las columnas 'precio' y 'alquiler_predicho'.
        porcentaje_entrada (float): Porcentaje de entrada para la hipoteca.
        coste_reformas (float): Coste total de las reformas.
        comision_agencia (float): Comisión de la agencia.
        anios (int): Duración del préstamo hipotecario en años.
        tin (float): Tasa de interés nominal del préstamo hipotecario.
        seguro_vida (float): Coste anual del seguro de vida.
        tipo_irpf (float): Tipo impositivo del IRPF.
        porcentaje_amortizacion (float): Porcentaje de amortización aplicable.

    Returns:
        pd.DataFrame: DataFrame con las métricas financieras calculadas añadidas.
    """
    def calcular_rentabilidad_fila(row):
        """
        Calcula la rentabilidad para una fila del DataFrame.

        Args:
            row (pd.Series): Una fila del DataFrame.

        Returns:
            dict: Diccionario con las métricas financieras calculadas.
        """
        return calcular_rentabilidad_inmobiliaria(
            porcentaje_entrada=porcentaje_entrada,
            coste_compra=row['precio'],
            coste_reformas=coste_reformas,
            comision_agencia=comision_agencia,
            alquiler_mensual=row['alquiler_predicho'],
            anios=anios,
            tin=tin,
            seguro_vida=seguro_vida,
            tipo_irpf=tipo_irpf,
            porcentaje_amortizacion=porcentaje_amortizacion
        )
    
    # Aplicar la función fila por fila y obtener un DataFrame con los resultados
    df_resultados = df.apply(lambda row: pd.Series(calcular_rentabilidad_fila(row)), axis=1)
    
    # Combinar el DataFrame original con los resultados
    df_final = pd.concat([df, df_resultados], axis=1)
    df_final.sort_values(by="Rentabilidad Bruta", ascending=False, inplace=True)
    
    return df_final