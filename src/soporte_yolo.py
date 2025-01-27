import pandas as pd
import requests
from PIL import Image
from ultralytics import YOLO
import ast
from tqdm import tqdm

# Carga del modelo YOLO
model = YOLO("../transformers/yolo11l-cls.pt")
# Documentación de soporte: https://docs.ultralytics.com/models/yolo11/#performance-metrics

# Listas de objetos relacionados con cocina y baño
kitchen_items = ["microwave", "oven", "refrigerator", "stove", "kitchen", "oven", "plate_rack"]
bathroom_items = ["toilet", "toilet_seat", "shower", "bathtub", "bathroom", "toothbrush", "medicine_chest"]


def detectar_habitacion(image_url):
    """
    Detecta el tipo de habitación (cocina o baño) en una imagen dada su URL.
    Requiere al menos 2 coincidencias para clasificar la habitación y detiene 
    el procesamiento tan pronto como se encuentran 2 coincidencias para un tipo de habitación.

    Parámetros:
        image_url (str): URL de la imagen a procesar.

    Devuelve:
        str: Tipo de habitación detectada ("kitchen", "bathroom" o None).
        list: Lista de todas las etiquetas detectadas en la imagen.
    """
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(response.raw)

        # Realizar detección con el modelo YOLO
        results = model(img, verbose=False)

        detected_labels = []
        if hasattr(results[0], "probs"):
            detected_labels = [model.names[int(class_id)] for class_id in results[0].probs.top5]

        # Verificar coincidencias con elementos de cocina
        kitchen_matches = 0
        for item in detected_labels:
            if item in kitchen_items:
                kitchen_matches += 1
                if kitchen_matches >= 2:
                    return "kitchen", detected_labels

        # Verificar coincidencias con elementos de baño
        bathroom_matches = 0
        for item in detected_labels:
            if item in bathroom_items:
                bathroom_matches += 1
                if bathroom_matches >= 2:
                    return "bathroom", detected_labels

        # Si no se detecta ningún tipo de habitación con al menos 2 coincidencias
        return None, detected_labels
    except Exception as e:
        print(f"Error processing {image_url}: {e}")
        return None, []


def procesar_urls(urls_as_string):
    """
    Procesa una lista de URLs para identificar las imágenes correspondientes
    a una cocina y un baño, basándose en la detección del tipo de habitación.

    Parámetros:
        urls_as_string (str): Cadena que contiene una lista de URLs en formato string.

    Devuelve:
        str: URL de la imagen identificada como cocina (o None si no se detecta).
        str: URL de la imagen identificada como baño (o None si no se detecta).
        list: Lista de detecciones con información de las URLs procesadas y las etiquetas detectadas.
    """
    try:
        urls = ast.literal_eval(urls_as_string)
    except Exception as e:
        print(f"Error al convertir las URLs: {e}")
        return None, None, []

    kitchen_url, bathroom_url = None, None
    all_detections = []

    for url in urls:
        detected_room, detections = detectar_habitacion(url)
        all_detections.append({"url": url, "detecciones": detections, "habitación": detected_room})

        if detected_room == "kitchen" and not kitchen_url:
            kitchen_url = url
        elif detected_room == "bathroom" and not bathroom_url:
            bathroom_url = url

        if kitchen_url and bathroom_url:
            break

    return kitchen_url, bathroom_url, all_detections


def identificar_urls_habitaciones(df, columna_urls, drop_nulls=True):
    """
    Identifica las URLs correspondientes a cocinas y baños en un DataFrame,
    basándose en la detección del tipo de habitación en las imágenes asociadas.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene una columna con listas de URLs a procesar.
        columna_urls (str): Nombre de la columna que contiene las listas de URLs.
        drop_nulls (bool): Si es True, elimina las filas donde no se detectan URLs de cocina o baño.

    Devuelve:
        pd.DataFrame: DataFrame original actualizado con columnas 'url_cocina' y 'url_banio'.
        pd.DataFrame: DataFrame con todas las detecciones realizadas, incluyendo las etiquetas detectadas.
    """
    tqdm.pandas()
    all_detections = []

    def process_row(urls):
        kitchen_url, bathroom_url, detections = procesar_urls(urls)
        all_detections.extend(detections)
        return pd.Series([kitchen_url, bathroom_url])

    df[["url_cocina", "url_banio"]] = df[columna_urls].progress_apply(process_row)

    if drop_nulls:
        print(f"Se han eliminado {df[['url_cocina', 'url_banio']].isnull().sum().sum()} filas donde 'url_cocina' o 'url_banio' son None o NaN.")
        df = df.dropna(subset=["url_cocina", "url_banio"]).reset_index(drop=True)

    detections_df = pd.DataFrame(all_detections)
    return df, detections_df