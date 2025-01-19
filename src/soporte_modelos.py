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
from xgboost import XGBRegressor as xgb

from tqdm import tqdm

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



def entrenar_modelo(X_train, y_train, X_test, y_test, params, regressor, modelo_final=False):
    """
    Entrena y evalúa un modelo de regresión utilizando GridSearchCV.

    Parámetros:
        X_train (pd.DataFrame): Características del conjunto de entrenamiento.
        y_train (pd.Series): Variable objetivo del conjunto de entrenamiento.
        X_test (pd.DataFrame): Características del conjunto de prueba.
        y_test (pd.Series): Variable objetivo del conjunto de prueba.
        params (dict): Diccionario con los hiperparámetros para la búsqueda en GridSearchCV.
        regressor (str): Nombre del modelo de regresión que se va a entrenar 
                         ('RandomForest', 'LinearRegression', 'DecisionTree', 'GradientBoost', 'XGBoost').
        modelo_final (bool, opcional): Si es True, retorna el modelo entrenado (default=False).

    Retorna:
        df_metricas: DataFrame con las métricas del modelo para los conjuntos de entrenamiento y prueba.
        best_model (opcional): El mejor modelo obtenido tras la búsqueda con GridSearchCV, si modelo_final=True.
    """
    # Seleccionar el modelo de regresión
    if regressor == "RandomForest":
        model = RandomForestRegressor()
    elif regressor == "LinearRegression":
        model = LinearRegression()
    elif regressor == "DecisionTree":
        model = DecisionTreeRegressor()
    elif regressor == "GradientBoost":
        model = GradientBoostingRegressor()
    elif regressor == "XGBoost":
        model = xgb.XGBRegressor()
    else:
        raise ValueError("Regressor no reconocido. Use 'RandomForest', 'LinearRegression', 'DecisionTree', 'GradientBoost' o 'XGBoost'.")

    # Configuración del GridSearchCV
    grid_search = GridSearchCV(
        model,
        params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    
    # Entrenamos el modelo
    grid_search.fit(X_train, y_train)
    
    # Obtenemos el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Realizamos predicciones para los conjuntos de entrenamiento y prueba
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculamos métricas
    df_metricas = pd.DataFrame({
        "Conjunto": ["Entrenamiento", "Prueba"],
        "MSE": [
            mean_squared_error(y_train, y_pred_train),
            mean_squared_error(y_test, y_pred_test)
        ],
        "MAE": [
            mean_absolute_error(y_train, y_pred_train),
            mean_absolute_error(y_test, y_pred_test)
        ],
        "R2": [
            r2_score(y_train, y_pred_train),
            r2_score(y_test, y_pred_test)
        ]
    })
    
    # Imprimimos los mejores hiperparámetros
    print(f'''Los mejores parámetros para el modelo {regressor} son:
    {grid_search.best_params_}
    \n
    Y sus mejores métricas son:''')
    display(df_metricas)

    # Retornar los resultados
    if modelo_final:
        return best_model, df_metricas
    else:
        return df_metricas