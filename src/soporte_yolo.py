import pandas as pd
import requests
from PIL import Image
from ultralytics import YOLO
import ast
from tqdm import tqdm

model = YOLO("yolov8n.pt")

kitchen_items = ["microwave", "oven", "refrigerator", "stove", "kitchen", "oven", "dishwasher"]  # Ajusta según tus clases
bathroom_items = ["toilet", "shower", "bathtub", "bathroom"]

def detectar_habitacion(image_url):
    try:
        # Descargar la imagen desde la URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        img = Image.open(response.raw)
        
        # Realizar la detección con YOLO
        results = model(img, verbose=False)
        
        # Extraer las etiquetas detectadas
        detections = [model.names[int(box[-1])] for box in results[0].boxes.data.tolist()]
        
        # Comprobar si contiene elementos de cocina o baño
        if any(item in detections for item in kitchen_items):
            return "kitchen"
        elif any(item in detections for item in bathroom_items):
            return "bathroom"
        else:
            return None
    except Exception as e:
        print(f"Error procesando {image_url}: {e}")
        return None


def procesar_urls(urls_as_string):
    try:
        urls = ast.literal_eval(urls_as_string)
    except Exception as e:
        print(f"Error en la conversión de: {e}")
        return None, None

    kitchen_url = None
    bathroom_url = None

    for url in urls:
        detected_room = detectar_habitacion(url)

        if detected_room == "kitchen" and kitchen_url is None:
            kitchen_url = url
        if detected_room == "bathroom" and bathroom_url is None:
            bathroom_url = url

        if kitchen_url and bathroom_url:
            break

    return kitchen_url, bathroom_url


def identificar_urls_habitaciones(df, columna_urls, drop_nulls=True):
    """
    Procesa las URLs en la columna especificada del DataFrame para identificar las imágenes 
    correspondientes a cocina y baño, y agrega las columnas 'url_cocina' y 'url_banio'.

    Parámetros:
        df (pd.DataFrame): El DataFrame de entrada.
        columna_urls (str): El nombre de la columna que contiene las URLs, almacenadas como 
                            cadenas que representan listas.
        drop_nulls (bool, opcional): Si es True, elimina las filas donde 'url_cocina' o 
                                     'url_banio' son None o NaN. Por defecto, es True.

    Retorna:
        pd.DataFrame: El DataFrame modificado con las columnas 'url_cocina' y 'url_banio' añadidas. 
                      Si drop_nulls es True, el DataFrame también excluirá las filas con valores nulos 
                      en esas columnas.
    """
    tqdm.pandas()
    df[["url_cocina", "url_banio"]] = df[columna_urls].progress_apply(lambda urls: pd.Series(procesar_urls(urls)))

    if drop_nulls:
        print(f"Se han eliminado {df[['url_cocina', 'url_banio']].isnull().sum().sum()} filas, donde 'url_cocina' o 'url_banio' son None o NaN.")
        df.dropna(subset=["url_cocina", "url_banio"], inplace=True)
    
    return df