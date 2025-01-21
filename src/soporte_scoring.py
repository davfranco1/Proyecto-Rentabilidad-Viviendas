import pandas as pd
import requests
from anthropic import Anthropic
import time
from tqdm.notebook import tqdm
import base64
import imghdr

import os
from dotenv import load_dotenv


load_dotenv(dotenv_path="/Users/davidfranco/Library/CloudStorage/OneDrive-Personal/Hackio/Jupyter/Proyecto-Rentabilidad-Viviendas/src/.env")

anthropic_key = os.getenv("anthropic_key")
if not anthropic_key:
    raise ValueError("anthropic_key no está definido en las variables de entorno")


def obtener_tipo_mime(contenido_imagen):
    """
    Detecta el tipo MIME de una imagen
    """
    formato_imagen = imghdr.what(None, contenido_imagen)
    if formato_imagen:
        return f'image/{formato_imagen}'
    return 'image/jpeg'  # valor por defecto

def url_a_base64_con_mime(url):
    """
    Descarga una imagen desde URL y la convierte a base64, detectando su tipo MIME
    """
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()
        contenido = respuesta.content
        tipo_mime = obtener_tipo_mime(contenido)
        return base64.b64encode(contenido).decode('utf-8'), tipo_mime
    except Exception as e:
        print(f"Error descargando imagen {url}: {str(e)}")
        return None, None

def obtener_analisis_habitaciones(cliente, url_cocina, url_banio):
    """
    Obtiene puntuaciones y metros cuadrados para cocina y baño usando imágenes en base64
    """
    try:
        # Convertir ambas imágenes a base64 y obtener sus tipos MIME
        cocina_base64, cocina_mime = url_a_base64_con_mime(url_cocina)
        banio_base64, banio_mime = url_a_base64_con_mime(url_banio)
        
        if not cocina_base64 or not banio_base64:
            return None, None, None, None

        mensaje = cliente.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": cocina_mime,
                                "data": cocina_base64
                            }
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": banio_mime,
                                "data": banio_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "Rate kitchen (1st image) and bathroom (2nd): condition (1-10) and size (sqm). Reply ONLY: kitchen_condition,kitchen_sqm,bath_condition,bath_sqm. Example: 7,12,8,6"
                        }
                    ]
                }
            ]
        )
        
        # Parsear los cuatro números de la respuesta
        puntuacion_cocina, metros_cocina, puntuacion_banio, metros_banio = map(float, mensaje.content[0].text.strip().split(','))
        return puntuacion_cocina, metros_cocina, puntuacion_banio, metros_banio
        
    except Exception as e:
        print(f"Error procesando imágenes: {str(e)}")
        return None, None, None, None


def analizar_propiedades(df):
    """
    Procesa el dataframe añadiendo puntuaciones y metros cuadrados
    """
    cliente = Anthropic(api_key=anthropic_key)
    
    # Crear nuevas columnas para puntuaciones y metros cuadrados
    df['puntuacion_cocina'] = None
    df['puntuacion_banio'] = None
    df['mts_cocina'] = None
    df['mts_banio'] = None
    
    for idx in tqdm(df.index, desc="Analizando propiedades"):
        if pd.notna(df.loc[idx, 'url_cocina']) and pd.notna(df.loc[idx, 'url_banio']):
            puntuacion_cocina, metros_cocina, puntuacion_banio, metros_banio = obtener_analisis_habitaciones(
                cliente,
                df.loc[idx, 'url_cocina'],
                df.loc[idx, 'url_banio']
            )
            
            df.loc[idx, 'puntuacion_cocina'] = puntuacion_cocina
            df.loc[idx, 'mts_cocina'] = metros_cocina
            df.loc[idx, 'puntuacion_banio'] = puntuacion_banio
            df.loc[idx, 'mts_banio'] = metros_banio
            
        time.sleep(1)
    
    return df