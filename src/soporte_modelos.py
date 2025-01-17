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

# Para realizar la regresión lineal y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold,LeaveOneOut, cross_val_score

from tqdm import tqdm

from xgboost import XGBRegressor

# Ignorar los warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')


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



def probar_modelo(X_train, y_train, X_test, y_test, params, regressor):
    """
    Entrena y evalúa un modelo de regresión utilizando GridSearchCV.

    Parámetros:
        X_train (pd.DataFrame): Características del conjunto de entrenamiento.
        y_train (pd.Series): Variable objetivo del conjunto de entrenamiento.
        X_test (pd.DataFrame): Características del conjunto de prueba.
        y_test (pd.Series): Variable objetivo del conjunto de prueba.
        params (dict): Diccionario con los hiperparámetros para la búsqueda en GridSearchCV.
        regressor (sklearn estimator): El modelo de regresión que se va a entrenar 
                                       (por ejemplo, RandomForestRegressor).

    Retorna:
        best_model: El mejor modelo obtenido tras la búsqueda con GridSearchCV.
        df_metricas: DataFrame con las métricas del modelo para los conjuntos de entrenamiento y prueba.
    """
    # Configuración del GridSearchCV
    grid_search = GridSearchCV(
        regressor,
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
    
    # Calculamos métricas utilizando la función existente
    df_metricas = metricas(y_train, y_pred_train, y_test, y_pred_test)
    
    # Imprimimos los mejores hiperparámetros
    print(f'''Los mejores parámetros para el modelo con {regressor} son:
    {grid_search.best_params_}
    \n
    Y sus mejores métricas son:''')
    display(df_metricas)

    return best_model, df_metricas

