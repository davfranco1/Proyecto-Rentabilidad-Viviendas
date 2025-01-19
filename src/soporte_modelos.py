# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
from IPython.display import display

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar el entrenamiento y la evaluación de los modelos
# -----------------------------------------------------------------------
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import pickle

import warnings
warnings.filterwarnings('ignore')


def separar_datos(dataframe, variable_respuesta, train_size=0.7, seed=42):
    """
    Divide un dataframe en conjuntos de entrenamiento y prueba, 
    devolviendo además las variables completas X e y.

    Parámetros:
    - dataframe (pd.DataFrame): El dataframe que contiene los datos a dividir.
    - variable_respuesta (str): El nombre de la columna que se usará como variable respuesta (dependiente).
    - train_size (float, opcional): Proporción de datos asignada al conjunto de entrenamiento (default=0.7).
    - semilla (int, opcional): Semilla para la generación de números aleatorios (default=42).

    Retorna:
    - X (pd.DataFrame): Matriz de características completa.
    - y (pd.DataFrame): Variable respuesta completa.
    - X_train (pd.DataFrame): Conjunto de características para entrenamiento.
    - X_test (pd.DataFrame): Conjunto de características para prueba.
    - y_train (pd.DataFrame): Variable respuesta para entrenamiento.
    - y_test (pd.DataFrame): Variable respuesta para prueba.
    """
    X = dataframe.drop(variable_respuesta, axis=1)
    y = dataframe[[variable_respuesta]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size= train_size, random_state= seed
    )
    return X, y, X_train, X_test, y_train, y_test


def metricas(y_train, y_train_pred, y_test, y_test_pred):
    metricas = {
        'train': {
            'r2_score': r2_score(y_train, y_train_pred),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'MSE': mean_squared_error(y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2_score': r2_score(y_test, y_test_pred),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'MSE': mean_squared_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred))

        }
    }
    return pd.DataFrame(metricas).T


def entrenar_modelo(X_train, y_train, X_test, y_test, params, regressor, random_state=42, modelo_final=False, X=None, y=None):
    """
    Entrena y evalúa un modelo de regresión utilizando GridSearchCV y opcionalmente 
    lo entrena con todos los datos disponibles.

    Parámetros:
        X_train (pd.DataFrame): Características del conjunto de entrenamiento.
        y_train (pd.Series): Variable objetivo del conjunto de entrenamiento.
        X_test (pd.DataFrame): Características del conjunto de prueba.
        y_test (pd.Series): Variable objetivo del conjunto de prueba.
        params (dict): Diccionario con los hiperparámetros para la búsqueda en GridSearchCV.
        regressor (str): Nombre del modelo de regresión que se va a entrenar 
                         ('RandomForest', 'LinearRegression', 'DecisionTree', 'GradientBoost', 'XGBoost').
        random_state (int, opcional): Semilla para garantizar reproducibilidad en modelos aleatorios (default=42).
        modelo_final (bool, opcional): Si es True, el modelo se entrena con todos los datos (X, y).
        X (pd.DataFrame, opcional): Características de todo el conjunto de datos.
        y (pd.DataFrame, opcional): Variable objetivo de todo el conjunto de datos.

    Retorna:
        best_model: El mejor modelo obtenido tras la búsqueda con GridSearchCV.
        df_metricas: DataFrame con las métricas del modelo para los conjuntos de entrenamiento y prueba.
        final_model (opcional): El modelo entrenado con todos los datos, si modelo_final=True.
    """
    # Seleccionar el modelo de regresión con random_state si aplica
    if regressor == "RandomForest":
        model = RandomForestRegressor(random_state=random_state)
    elif regressor == "LinearRegression":
        model = LinearRegression()
    elif regressor == "DecisionTree":
        model = DecisionTreeRegressor(random_state=random_state)
    elif regressor == "GradientBoost":
        model = GradientBoostingRegressor(random_state=random_state)
    elif regressor == "XGBoost":
        model = xgb.XGBRegressor(random_state=random_state)
    else:
        raise ValueError("Regressor no reconocido. Use 'RandomForest', 'LinearRegression', 'DecisionTree', 'GradientBoost' o 'XGBoost'.")

    # Configuración de GridSearchCV
    grid_search = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    
    # Entrenar el modelo
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Predicciones
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calcular métricas
    df_metricas = pd.DataFrame({
        "R2": [
            r2_score(y_train, y_pred_train),
            r2_score(y_test, y_pred_test)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_train, y_pred_train)),
            np.sqrt(mean_squared_error(y_test, y_pred_test))
        ],
        "MSE": [
            mean_squared_error(y_train, y_pred_train),
            mean_squared_error(y_test, y_pred_test)
        ],
        "MAE": [
            mean_absolute_error(y_train, y_pred_train),
            mean_absolute_error(y_test, y_pred_test)
        ]
    }, index=["Train", "Test"])
    
    # Imprimir los resultados
    print(f'''Los mejores parámetros para el modelo {regressor} son:
    {grid_search.best_params_}
    \n
    Y sus mejores métricas son:''')
    display(df_metricas)


    final_model = None
    if modelo_final:
        if X is None or y is None:
            raise ValueError("X y y deben proporcionarse para entrenar el modelo con todos los datos.")
        
        final_model = type(best_model)(**grid_search.best_params_)
        final_model.random_state = random_state
        final_model.fit(X, y)
        print("Modelo final entrenado con todos los datos.")

    if modelo_final:
        return best_model, df_metricas, final_model
    else:
        return best_model, df_metricas
    






class ModelosRegresion:
    def __init__(self, dataframe, variable_respuesta, train_size=0.7, random_state=42):
        """
        Inicializa la clase con el dataframe, la variable respuesta y divide los datos
        automáticamente en conjuntos de entrenamiento y prueba.

        Parámetros:
        - dataframe (pd.DataFrame): El dataframe con los datos a procesar.
        - variable_respuesta (str): El nombre de la columna de la variable respuesta.
        - train_size (float, opcional): Proporción de datos asignada al conjunto de entrenamiento (default=0.7).
        - random_state (int, opcional): Semilla para reproducibilidad en modelos aleatorios (default=42).
        """
        self.random_state = random_state
        self.train_size = train_size
        self.variable_respuesta = variable_respuesta

        # Separar datos al inicializar la clase
        self.X = dataframe.drop(variable_respuesta, axis=1)
        self.y = dataframe[[variable_respuesta]]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=train_size, random_state=random_state
        )
        print("Datos separados automáticamente al inicializar la clase.")

        # Atributo para el modelo final
        self.final_model = None

    def entrenar_modelo(self, params, regressor, modelo_final=False):
        """
        Entrena y evalúa un modelo de regresión utilizando GridSearchCV y opcionalmente
        lo entrena con todos los datos disponibles.

        Parámetros:
        - params (dict): Diccionario con los hiperparámetros para la búsqueda en GridSearchCV.
        - regressor (str): Nombre del modelo de regresión que se va a entrenar 
                           ('RandomForest', 'LinearRegression', 'DecisionTree', 'GradientBoost', 'XGBoost').
        - modelo_final (bool, opcional): Si es True, el modelo se entrena con todos los datos (X, y).

        Retorna:
        - best_model: El mejor modelo obtenido tras la búsqueda con GridSearchCV.
        - df_metricas: DataFrame con las métricas del modelo para los conjuntos de entrenamiento y prueba.
        - final_model (opcional): El modelo entrenado con todos los datos, si modelo_final=True.
        """
        # Seleccionar el modelo de regresión con random_state si aplica
        if regressor == "RandomForest":
            model = RandomForestRegressor(random_state=self.random_state)
        elif regressor == "LinearRegression":
            model = LinearRegression()
        elif regressor == "DecisionTree":
            model = DecisionTreeRegressor(random_state=self.random_state)
        elif regressor == "GradientBoost":
            model = GradientBoostingRegressor(random_state=self.random_state)
        elif regressor == "XGBoost":
            model = XGBRegressor(random_state=self.random_state)
        else:
            raise ValueError("Regressor no reconocido. Use 'RandomForest', 'LinearRegression', 'DecisionTree', 'GradientBoost' o 'XGBoost'.")

        # Configuración de GridSearchCV
        grid_search = GridSearchCV(
            model,
            params,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        
        # Entrenar el modelo
        grid_search.fit(self.X_train, self.y_train)
        
        # Obtener el mejor modelo
        best_model = grid_search.best_estimator_
        
        # Predicciones
        y_pred_train = best_model.predict(self.X_train)
        y_pred_test = best_model.predict(self.X_test)
        
        # Calcular métricas
        df_metricas = pd.DataFrame({
            "R2": [
                r2_score(self.y_train, y_pred_train),
                r2_score(self.y_test, y_pred_test)
            ],
            "RMSE": [
                np.sqrt(mean_squared_error(self.y_train, y_pred_train)),
                np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            ],
            "MSE": [
                mean_squared_error(self.y_train, y_pred_train),
                mean_squared_error(self.y_test, y_pred_test)
            ],
            "MAE": [
                mean_absolute_error(self.y_train, y_pred_train),
                mean_absolute_error(self.y_test, y_pred_test)
            ]
        }, index=["Train", "Test"])
        
        # Imprimir los resultados
        print(f'''Los mejores parámetros para el modelo {regressor} son:
        {grid_search.best_params_}
        \n
        Y sus mejores métricas son:''')
        display(df_metricas)

        # Entrenar con todos los datos si es necesario
        if modelo_final:
            if self.X is None or self.y is None:
                raise ValueError("Los datos completos (X, y) deben estar definidos para entrenar el modelo final.")
            
            self.final_model = type(best_model)(**grid_search.best_params_)
            self.final_model.random_state = self.random_state
            self.final_model.fit(self.X, self.y)
            print("Modelo final entrenado con todos los datos.")

        if modelo_final:
            return best_model, df_metricas, self.final_model
        else:
            return best_model, df_metricas

    def guardar_modelo(self, ruta):
        """
        Guarda el modelo final en un archivo especificado.

        Parámetros:
        - ruta (str): Ruta del archivo donde se guardará el modelo.
        """
        if self.final_model is None:
            raise ValueError("No hay un modelo final para guardar. Asegúrate de entrenar el modelo final primero.")
        
        with open(ruta, 'wb') as f:
            pickle.dump(self.final_model, f)
        print(f"Modelo guardado en {ruta}.")