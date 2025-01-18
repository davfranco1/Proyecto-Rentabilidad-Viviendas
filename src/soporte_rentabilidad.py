def calcular_rentabilidad_inmobiliaria(porcentaje_entrada, coste_compra, coste_reformas, comision_agencia, 
                                       porcentaje_coste_notario, coste_impuestos, alquiler_mensual, anios, tin):
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

    Devuelve:
    - Un diccionario con las métricas financieras calculadas.
    """
    # Coste total
    coste_notario = porcentaje_coste_notario * coste_compra
    coste_total = coste_compra + coste_reformas + comision_agencia + coste_notario + coste_impuestos
    
    # Pago inicial (inversión inicial)
    pago_entrada = porcentaje_entrada * coste_compra
    
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
    
    return {
        "Coste Total": coste_total,
        "Pago Entrada": pago_entrada,
        "Monto Préstamo": monto_prestamo,
        "Hipoteca Mensual": hipoteca_mensual,
        "Interés Total": interes_total,
        "Alquiler Anual": alquiler_anual,
        "Alquiler Mensual": alquiler_mensual
    }