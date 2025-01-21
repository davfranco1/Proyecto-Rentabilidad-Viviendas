import pandas as pd
import requests
from anthropic import Anthropic
import time
import os
from tqdm.notebook import tqdm

anthropic_key = os.getenv("anthropic_key")


def obtener_puntuaciones_propiedad(cliente, url_cocina, url_banio):
    """
    Obtiene puntuaciones para cocina y baño en un solo llamado a la API
    
    Parámetros:
    cliente (Anthropic): Cliente Anthropic inicializado
    url_cocina (str): URL HTTPS de la imagen de la cocina
    url_banio (str): URL HTTPS de la imagen del baño
    
    Retorna:
    tuple: (puntuacion_cocina, puntuacion_banio) o (None, None) si hay error
    """
    try:
        mensaje = cliente.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url_cocina
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url_banio
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analiza estas dos imágenes. La primera es una cocina y la segunda un baño. Da dos números del 1 al 10 calificando la condición de cada uno, donde 1 es condición extremadamente pobre y 10 es condición perfecta. Responde SOLO con dos números separados por coma, primero la cocina y luego el baño. Por ejemplo: 7,8"
                        }
                    ]
                }
            ]
        )
        
        # Extraer las dos puntuaciones de la respuesta
        cocina, banio = map(float, mensaje.content[0].text.strip().split(','))
        return cocina, banio
        
    except Exception as e:
        print(f"Error procesando imágenes: {str(e)}")
        return None, None

def puntuar_propiedades(df):
    """
    Procesa un dataframe que contiene URLs de cocinas y baños y añade puntuaciones
    
    Parámetros:
    df (pandas.DataFrame): Dataframe con columnas 'url_cocina' y 'url_banio'
    
    Retorna:
    pandas.DataFrame: Dataframe original con columnas de puntuación añadidas
    """
    # Inicializar cliente Anthropic usando la variable global
    cliente = Anthropic(api_key=anthropic_key)
    
    # Crear nuevas columnas para puntuaciones
    df['puntuacion_cocina'] = None
    df['puntuacion_banio'] = None
    
    # Procesar cada fila
    for idx in tqdm(df.index, desc="Puntuando propiedades"):
        # Solo procesar si ambas URLs están presentes
        if pd.notna(df.loc[idx, 'url_cocina']) and pd.notna(df.loc[idx, 'url_banio']):
            puntuacion_cocina, puntuacion_banio = obtener_puntuaciones_propiedad(
                cliente,
                df.loc[idx, 'url_cocina'],
                df.loc[idx, 'url_banio']
            )
            
            df.loc[idx, 'puntuacion_cocina'] = puntuacion_cocina
            df.loc[idx, 'puntuacion_banio'] = puntuacion_banio
            
        # Añadir retraso para respetar límites de tasa
        time.sleep(0.5)
    
    return df