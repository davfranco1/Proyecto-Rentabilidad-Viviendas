# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd
import pickle

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest # para hacer el ztest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from itertools import product, combinations

from scipy.stats import chi2_contingency

from tqdm import tqdm

def exploracion_datos(dataframe):

    """
    Realiza una exploración básica de los datos en el DataFrame dado e imprime varias estadísticas descriptivas.

    Parameters:
    -----------
    dataframe : pandas DataFrame. El DataFrame que se va a explorar.

    Returns:
    --------
    None

    Imprime varias estadísticas descriptivas sobre el DataFrame, incluyendo:
    - El número de filas y columnas en el DataFrame.
    - El número de valores duplicados en el DataFrame.
    - Una tabla que muestra las columnas con valores nulos y sus porcentajes.
    - Las principales estadísticas de las variables numéricas en el DataFrame.
    - Las principales estadísticas de las variables categóricas en el DataFrame.

    """

    print(f"El número de filas es {dataframe.shape[0]} y el número de columnas es {dataframe.shape[1]}")

    print("\n----------\n")

    print(f"En este conjunto de datos tenemos {dataframe.duplicated().sum()} valores duplicados")

    
    print("\n----------\n")


    print("Los columnas con valores nulos y sus porcentajes son: ")
    dataframe_nulos = dataframe.isnull().sum()

    display((dataframe_nulos[dataframe_nulos.values >0] / dataframe.shape[0]) * 100)

    print("\n----------\n")
    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)

    print("\n----------\n")
    print("Las características principales del dataframe son:")
    display(dataframe.info())

class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include = ["object", "category"])
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5), bins=20):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        _, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=bins)
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

    def plot_categoricas(self, color="grey", tamano_grafica=(40, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        dataframe_cat = self.separar_dataframes()[1]
        _, axes = plt.subplots(2, math.ceil(len(dataframe_cat.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=45)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.suptitle("Distribución de variables categóricas")

    def plot_relacion(self, vr, tamano_grafica=(40, 12), color="grey"):
        """
        Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.

        Parameters:
            - vr (str): El nombre de la variable en el eje y.
            - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (40, 12).
            - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        Returns:
            No devuelve nada    
        """
        df_numericas = self.separar_dataframes()[0].columns
        meses_ordenados = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in df_numericas:
                sns.scatterplot(x=vr, 
                                y=columna, 
                                data=self.dataframe, 
                                color=color, 
                                ax=axes[indice])
                axes[indice].set_title(columna)
                axes[indice].set(xlabel=None)
            else:
                if columna == "Month":
                    sns.barplot(x=columna, y=vr, data=self.dataframe, order=meses_ordenados, ax=axes[indice],
                                color=color)
                    axes[indice].tick_params(rotation=90)
                    axes[indice].set_title(columna)
                    axes[indice].set(xlabel=None)
                else:
                    sns.barplot(x=columna, y=vr, data=self.dataframe, ax=axes[indice], color=color)
                    axes[indice].tick_params(rotation=90)
                    axes[indice].set_title(columna)
                    axes[indice].set(xlabel=None)

        plt.tight_layout()
    
    def analisis_temporal(self, var_respuesta, var_temporal, color = "black", order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):

        """
        Realiza un análisis temporal mensual de una variable de respuesta en relación con una variable temporal. Visualiza un gráfico de líneas que muestra la relación entre la variable de respuesta y la variable temporal (mes), con la línea de la media de la variable de respuesta.

        Params:
        -----------
        dataframe : pandas DataFrame. El DataFrame que contiene los datos.
        var_respuesta : str. El nombre de la columna que contiene la variable de respuesta.
        var_temporal : str. El nombre de la columna que contiene la variable temporal (normalmente el mes).
        order : list, opcional.  El orden de los meses para representar gráficamente. Por defecto, se utiliza el orden estándar de los meses.

        Returns:
        --------
        None
        """

        plt.figure(figsize = (15, 5))

        # Convierte la columna "Month" en un tipo de datos categórico con el orden especificado
        self.dataframe[var_temporal] = pd.Categorical(self.dataframe[var_temporal], categories=order, ordered=True)

        # Trama el gráfico
        sns.lineplot(x=var_temporal, 
                     y=var_respuesta, 
                     data=self.dataframe, 
                     color = color)

        # Calcula la media de PageValues
        mean_page_values = self.dataframe[var_respuesta].mean()

        # Agrega la línea de la media
        plt.axhline(mean_page_values, 
                    color='green', 
                    linestyle='--', 
                    label='Media de PageValues')


        # quita los ejes de arriba y de la derecha
        sns.despine()

        # Rotula el eje x
        plt.xlabel("Month");


    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
    

def gestion_nulos_lof(df, col_numericas, list_neighbors, lista_contaminacion):
    
    combinaciones = list(product(list_neighbors, lista_contaminacion))
    
    for neighbors, contaminacion in tqdm(combinaciones):
        lof = LocalOutlierFactor(n_neighbors=neighbors, 
                                 contamination=contaminacion,
                                 n_jobs=-1)
        df[f"outliers_lof_{neighbors}_{contaminacion}"] = lof.fit_predict(df[col_numericas])

    return df

def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    for categoria in dataframe[columna_control].unique():
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe().T)


def separar_dataframe(dataframe):
    """
    Separa un DataFrame en dos DataFrames: uno con columnas numéricas y otro con columnas categóricas.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame original.

    Returns:
    - (pd.DataFrame, pd.DataFrame): Un DataFrame con columnas numéricas y otro con columnas categóricas.
    """
    return dataframe.select_dtypes(include=np.number), dataframe.select_dtypes(include="O")


def plot_numericas(dataframe, figsize=(15, 10), bins=50):
    """
    Genera histogramas para todas las columnas numéricas de un DataFrame.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene las columnas numéricas.

    Returns:
    - None: Muestra los histogramas directamente.
    """
    cols_numericas = dataframe.select_dtypes(include=np.number).columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        sns.histplot(x=columna, data=dataframe, ax=axes[indice], bins=bins)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def plot_cat(dataframe, paleta="mako", tamano_grafica=(15, 10)):
    """
    Genera gráficos de conteo para todas las columnas categóricas de un DataFrame.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene las columnas categóricas.
    - paleta (str): Paleta de colores para los gráficos.
    - tamano_grafica (tuple): Tamaño de la figura.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    cols_categoricas = dataframe.select_dtypes(include=["category", "object"]).columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        sns.countplot(
            x=columna,
            data=dataframe,
            ax=axes[indice],
            palette=paleta,
            order=dataframe[columna].value_counts().index
        )
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")
        axes[indice].tick_params(rotation=45)

    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def relacion_vs_cat(dataframe, variable_respuesta, paleta="mako", tamano_grafica=(15, 10)):
    """
    Muestra la relación entre una variable categórica y una respuesta usando gráficos de barras.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    - variable_respuesta (str): Nombre de la variable respuesta.
    - paleta (str): Paleta de colores para los gráficos.
    - tamano_grafica (tuple): Tamaño de la figura.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    df_cat = separar_dataframe(dataframe)[1]
    cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta, ascending=False)
        sns.barplot(
            x=columna,
            y=variable_respuesta,
            data=datos_agrupados,
            ax=axes[indice],
            palette=paleta
        )
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre {columna} y {variable_respuesta}")
    plt.tight_layout()


def relacion_vs_numericas(dataframe, variable_respuesta, paleta="mako", tamano_grafica=(15, 10)):
    """
    Muestra la relación entre variables numéricas y una respuesta usando gráficos scatter.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    - variable_respuesta (str): Nombre de la variable respuesta.
    - paleta (str): Paleta de colores para los gráficos.
    - tamano_grafica (tuple): Tamaño de la figura.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    numericas = separar_dataframe(dataframe)[0]
    cols_numericas = numericas.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(
                x=columna,
                y=variable_respuesta,
                data=numericas,
                ax=axes[indice]
            )
    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()


def matriz_correlacion(dataframe):
    """
    Muestra un mapa de calor con la matriz de correlación entre variables numéricas.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos.

    Returns:
    - None: Muestra el mapa de calor directamente.
    """
    matriz_corr = dataframe.corr(numeric_only=True)

    plt.figure(figsize=(10, 5))
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))

    sns.heatmap(
        matriz_corr,
        annot=True,
        vmin=-1,
        vmax=1,
        mask=mascara,
        cmap="coolwarm"
    )
    plt.show()


def detectar_outliers(dataframe, color="orange", tamano_grafica=(15, 10)):
    """
    Muestra gráficos de caja para detectar outliers en columnas numéricas.

    Parámetros:
    - dataframe (pd.DataFrame): El DataFrame que contiene los datos.
    - color (str): Color para los gráficos.
    - tamano_grafica (tuple): Tamaño de la figura.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    df_num = separar_dataframe(dataframe)[0]
    num_filas = math.ceil(len(df_num.columns) / 2)

    fig, axes = plt.subplots(ncols=2, nrows=num_filas, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):
        sns.boxplot(
            x=columna,
            data=df_num,
            ax=axes[indice],
            color=color,
            flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5}
        )
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")

    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.show()


def relacion_var_numericas_problem_cat(df, vr, figsize=(15,10)):
    df_num, df_cat = separar_dataframe(df)
    col_numericas = df_num.columns
    num_filas = math.ceil(len(col_numericas)/2)
    fig, axes = plt.subplots(num_filas, 2, figsize=figsize)
    axes = axes.flat
    for indice, columna in enumerate(col_numericas):
        sns.histplot(df, x=columna, ax=axes[indice], hue= vr, bins=30)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(col_numericas) % 2 ==1:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()


def relacion_var_categoricas_problem_cat(df, vr, figsize=(15,20), rotate=True):
    df_cat = separar_dataframe(df)[1]
    columnas_categoticas = df_cat.columns
    df_cat[vr] = df[vr]
    num_filas = math.ceil(len(columnas_categoticas)/2)
    fig, axes = plt.subplots(num_filas,2, figsize=figsize)
    axes = axes.flat

    for indice, columna in enumerate(columnas_categoticas):
        sns.countplot(df, 
                      x=columna, 
                      palette="Set2",
                      order = df[columna].value_counts().index,
                      hue=vr,
                      ax=axes[indice])
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")
        if rotate:
            axes[indice].tick_params(axis='x', rotation=45)

    if len(columnas_categoticas) % 2 == 1:
        fig.delaxes(axes[-1])

    plt.tight_layout()


def outliers_isolation_forest(df, niveles_contaminacion = [0.01, 0.05, 0.1], lista_estimadores = [50, 100, 200]):
    """
    Agrega columnas al DataFrame con la detección de outliers utilizando Isolation Forest
    para diferentes niveles de contaminación y números de estimadores.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada.
    - niveles_contaminacion (list): Lista de niveles de contaminación a probar (por ejemplo, [0.01, 0.05, 0.1]).
    - lista_estimadores (list): Lista de cantidades de estimadores a probar (por ejemplo, [50, 100, 200]).

    Returns:
    - pd.DataFrame: DataFrame con nuevas columnas para cada configuración de Isolation Forest.
    """
    # Seleccionar columnas numéricas
    col_numericas = df.select_dtypes(include=np.number).columns

    # Generar todas las combinaciones de niveles de contaminación y estimadores
    combinaciones = list(product(niveles_contaminacion, lista_estimadores))

    for cont, esti in combinaciones:
        # Inicializar Isolation Forest
        ifo = IsolationForest(
            n_estimators=esti,
            contamination=cont,
            n_jobs=-1  # Usar todos los núcleos disponibles
        )

        # Ajustar y predecir outliers
        df[f"outliers_ifo_{cont}_{esti}"] = ifo.fit_predict(df[col_numericas])
    
    return df


def visualizar_outliers(df, cols_numericas, figsize=(15, 10)):
    """
    Genera visualizaciones para las combinaciones de columnas numéricas utilizando las columnas
    que identifican outliers como `hue` en gráficos scatter.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos.
    - cols_numericas (list): Lista de columnas numéricas a combinar.
    - figsize (tuple): Tamaño de cada figura de visualización.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    # Crear todas las combinaciones de pares de columnas numéricas
    combinaciones_viz = list(combinations(cols_numericas, 2))
    columnas_hue = df.filter(like="outlier").columns

    for outlier in columnas_hue:
        # Calcular el número de filas y columnas necesarias para los subplots
        num_combinaciones = len(combinaciones_viz)
        ncols = min(3, num_combinaciones)  # Máximo 3 columnas por fila
        nrows = (num_combinaciones + ncols - 1) // ncols  # Calcular filas necesarias
        
        # Crear figura y ejes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if num_combinaciones > 1 else [axes]  # Asegurar array plano
        
        for indice, (x, y) in enumerate(combinaciones_viz):
            sns.scatterplot(
                x=x,
                y=y,
                ax=axes[indice],
                data=df,
                hue=outlier,
                palette="Set1",
                style=outlier,
                alpha=0.4
            )
            axes[indice].set_title(f"{x} vs {y}")

        # Eliminar ejes vacíos si hay menos gráficos que subplots
        for ax in axes[len(combinaciones_viz):]:
            ax.axis("off")

        # Configuración general de la figura
        plt.suptitle(f"Outlier Analysis: {outlier}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar espacio sin solapar el título
        plt.show()

def filtrar_outliers(dataframe, porcentaje, drop_indices=False, drop_columnas=True):
    """
    Filtra filas en un DataFrame basado en la proporción de columnas con valores atípicos y opcionalmente elimina filas y columnas.

    Parámetros:
    - dataframe: El DataFrame a filtrar.
    - porcentaje: El porcentaje de columnas con valores atípicos (-1) requerido para filtrar una fila.
    - drop_indices: Booleano, opcional. Si es True, elimina las filas filtradas del DataFrame original. Por defecto es False.
    - drop_columnas: Booleano, opcional. Si es True, elimina las columnas que contienen "outliers" en su nombre del DataFrame original. Por defecto es True.

    Retorna:
    - dataframe: El DataFrame actualizado después de eliminar filas y/o columnas, si se especifica.
    - df_filtrado: El DataFrame filtrado que contiene las filas que fueron identificadas como que tienen valores atípicos.

    Uso:
    dataframe_filtrado, filas_filtradas = filtrar_outliers(dataframe, porcentaje=0.3, drop_indices=True, drop_columnas=False)
    """
    total_columnas = len(dataframe.filter(like="outliers").columns)
    porcentaje = porcentaje
    df_filtrado = dataframe[(dataframe == -1).sum(axis=1) > total_columnas * porcentaje]
    print(f"Se han filtrado {len(df_filtrado)} filas, que representan un {round(len(df_filtrado)/len(dataframe)*100, 2)}% del dataframe original.")
    display(df_filtrado.head())

    if drop_indices:
        dataframe.drop(df_filtrado.index, inplace=True)
        print(f"Se han eliminado {len(df_filtrado)} filas ({round(len(df_filtrado)/len(dataframe)*100, 0)}%) del dataframe original.")
    else:
        print(f"Se han mantenido todas las filas del dataframe original.")

    if drop_columnas:
        dataframe.drop(columns = dataframe.filter(like="outliers").columns, inplace=True)
        df_filtrado.drop(columns = df_filtrado.filter(like="outliers").columns, inplace=True)

    return dataframe, df_filtrado


def plot_top5_num(df, col_agrupacion='Country', figsize=(15, 10)):
    """
    Plots the top 5 values for each numeric column in a DataFrame, using a grouping column (e.g., country names) as labels,
    organized in two columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing data about countries.
    - col_agrupacion (str): The name of the column containing labels (e.g., country names).
    - figsize (tuple): Figure size for the subplots.

    Returns:
    None (Displays the subplots directly).
    """
    # Filter numeric columns
    numeric_columns = df.select_dtypes(include='number').columns
    num_columns = len(numeric_columns)
    
    # Calculate rows needed for two-column layout
    nrows = (num_columns + 1) // 2  # Round up for odd numbers
    
    # Create subplots with a 2-column layout
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()  # Flatten for easy indexing

    # Loop through each column and plot the top 5
    for i, col in enumerate(numeric_columns):
        top5 = df.nlargest(5, col)  # Get the top 5 rows for the column
        axes[i].bar(top5[col_agrupacion], top5[col])  # Create traditional bar plot
        axes[i].set_title(f"Top 5 for {col}")
        axes[i].set_ylabel(col)
        axes[i].set_xlabel(col_agrupacion)
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.show()

def exportar_pickles(dicc_pickle):
    """
    Guarda múltiples objetos en archivos pickle.

    Parámetros:
    - dicc_pickle: Diccionario donde las claves son las rutas de los archivos y los valores son los objetos a guardar.

    """
    for path, variable in dicc_pickle.items():
        with open(path, 'wb') as f:
            pickle.dump(variable, f)
        print(f"Guardado: {path}")