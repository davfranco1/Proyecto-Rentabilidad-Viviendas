# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Feature scaling
# -----------------------------------------------------------------------
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer, StandardScaler

# Gráficos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns


def aplicar_escaladores(df, columnas, escaladores, return_scalers=False):
    """
    Aplica múltiples escaladores secuencialmente a columnas específicas y devuelve un DataFrame con todas las transformaciones.

    Parámetros:
    - df: DataFrame de entrada.
    - columnas: Lista de nombres de columnas a escalar.
    - escaladores: Lista de instancias de escaladores de sklearn.
    - return_scalers: Booleano, si es True devuelve también los escaladores utilizados y el DataFrame solo con columnas escaladas.

    Retorna:
    - df_escalado: DataFrame con columnas originales seguidas por las columnas escaladas.
      Si `return_scalers` es True, incluye todas las columnas del DataFrame original y las escaladas en el mismo orden.
    - (Opcional) escaladores_aplicados: Lista de escaladores utilizados.
    """
    # Crear una copia inicial con todas las columnas originales
    df_escalado = df.copy()

    # Diccionario para guardar los escaladores aplicados
    escaladores_aplicados = []

    # DataFrame para almacenar solo las columnas escaladas si return_scalers=True
    df_solo_escaladas = df.copy()

    # Aplicar cada escalador secuencialmente
    for escalador in escaladores:
        # Ajustar y transformar las columnas seleccionadas
        datos_escalados = escalador.fit_transform(df[columnas])

        # Añadir las columnas escaladas al DataFrame original
        columnas_escaladas = [f"{col}_{type(escalador).__name__}" for col in columnas]
        df_escalado[columnas_escaladas] = pd.DataFrame(datos_escalados, columns=columnas_escaladas, index=df.index)

        # Añadir las columnas escaladas con nombres originales al DataFrame reducido
        df_solo_escaladas[columnas] = pd.DataFrame(datos_escalados, columns=columnas, index=df.index)

        # Almacenar el escalador utilizado
        escaladores_aplicados.append(escalador)

    if return_scalers:
        return df_solo_escaladas, escaladores_aplicados
    else:
        return df_escalado



def graficar_escaladores(df, variables_originales, variables_escaladas):
    """
    Genera gráficos comparativos (boxplots e histogramas) donde cada fila corresponde a una variable original
    y cada fila tiene 5 gráficos: el gráfico original y 4 versiones escaladas.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene las variables originales y escaladas.
    - variables_originales (list): Lista de columnas originales.
    - variables_escaladas (list): Lista de columnas escaladas (deben incluir el nombre de la variable original en el nombre).

    Returns:
    - None: Muestra los gráficos directamente.
    """
    # Total de gráficos por fila (1 original + columnas escaladas asociadas)
    num_columnas = 5  # 4 escaladas + 1 original
    total_graficos = len(variables_originales) * 2 * num_columnas  # 2 tipos de gráficos: boxplot e histograma

    # Configurar figura y ejes
    filas = len(variables_originales) * 2  # 2 filas por variable original (boxplot e histograma)
    fig, axes = plt.subplots(
        nrows=filas, 
        ncols=num_columnas, 
        figsize=(num_columnas * 5, filas * 2.5)
    )
    axes = axes.flat

    # Generar gráficos para cada variable original
    current_plot = 0
    for variable in variables_originales:
        # Filtrar las variables escaladas correspondientes
        escaladas = [col for col in variables_escaladas if variable in col]

        # Asegurar que haya exactamente 4 variables escaladas para cada variable original
        if len(escaladas) < 4:
            escaladas.extend([None] * (4 - len(escaladas)))

        # Boxplots (primera fila por variable)
        for col in [variable] + escaladas[:4]:
            if col is not None:
                sns.boxplot(x=col, data=df, ax=axes[current_plot])
                axes[current_plot].set_title(f"Boxplot: {col}")
            else:
                axes[current_plot].axis('off')  # Desactivar el eje si no hay suficiente variable escalada
            current_plot += 1

        # Histogramas (segunda fila por variable)
        for col in [variable] + escaladas[:4]:
            if col is not None:
                sns.histplot(x=col, data=df, ax=axes[current_plot])
                axes[current_plot].set_title(f"Histograma: {col}")
            else:
                axes[current_plot].axis('off')  # Desactivar el eje si no hay suficiente variable escalada
            current_plot += 1

    # Ajustar diseño
    plt.tight_layout()
    plt.show()