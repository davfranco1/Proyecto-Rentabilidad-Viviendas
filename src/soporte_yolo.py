import pandas as pd
import requests
from PIL import Image
from ultralytics import YOLO
import ast
from tqdm import tqdm

# Carga del modelo YOLO
model = YOLO("../transformers/yolo11x-cls.pt")

# Listas de objetos relacionados con cocina y baño
kitchen_items = ["microwave", "oven", "refrigerator", "stove", "kitchen", "oven", "plate_rack"]
bathroom_items = ["toilet", "toilet_seat", "shower", "bathtub", "bathroom", "toothbrush", "medicine_chest"]


def detectar_habitacion(image_url):
    """
    Detects the type of room (kitchen or bathroom) in an image given its URL,
    requiring at least 2 matches to classify the room. Stops processing as soon
    as 2 matches are found for a room type.

    Parameters:
        image_url (str): URL of the image to process.

    Returns:
        str: Detected room type ("kitchen", "bathroom", or None).
        list: List of all detected labels.
    """
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(response.raw)

        # Perform detection with YOLO
        results = model(img, verbose=False)

        detected_labels = []
        if hasattr(results[0], "probs"):
            detected_labels = [model.names[int(class_id)] for class_id in results[0].probs.top5]

        # Check matches for kitchen items
        kitchen_matches = 0
        for item in detected_labels:
            if item in kitchen_items:
                kitchen_matches += 1
                if kitchen_matches >= 2:
                    return "kitchen", detected_labels

        # Check matches for bathroom items
        bathroom_matches = 0
        for item in detected_labels:
            if item in bathroom_items:
                bathroom_matches += 1
                if bathroom_matches >= 2:
                    return "bathroom", detected_labels

        # If no room type is detected with at least 2 matches
        return None, detected_labels
    except Exception as e:
        print(f"Error processing {image_url}: {e}")
        return None, []

def procesar_urls(urls_as_string):
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