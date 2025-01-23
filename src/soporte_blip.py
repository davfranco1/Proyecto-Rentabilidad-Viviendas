import pandas as pd
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO

# Cargar el procesador y modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generar_descripciones(df, columna_url):
    """
    Genera descripciones de imágenes a partir de una columna de URLs en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        columna_url (str): El nombre de la columna que contiene las URLs de las imágenes.

    Returns:
        list: Una lista con las descripciones generadas (o None en caso de errores).
    """
    descripciones = []

    for indice, url_imagen in df[columna_url].items():
        try:
            # Cargar la imagen desde la URL
            response = requests.get(url_imagen)
            response.raise_for_status()  # Verificar que la solicitud fue exitosa
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Procesar la imagen y generar una descripción
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.generate(**inputs)
            descripcion = processor.decode(outputs[0], skip_special_tokens=True)
            
            # Agregar la descripción a los resultados
            descripciones.append(descripcion)
        except Exception as e:
            print(f"Error al generar la descripción para el índice {indice}: {e}")
            descripciones.append(None)  # Agregar None en caso de errores

    return descripciones

def contar_palabras(frases, palabra):
    """
    Cuenta cuántas veces aparece una palabra en una lista de frases.

    Args:
        frases (list of str): La lista de frases.
        palabra (str): La palabra que se desea contar (no distingue entre mayúsculas y minúsculas).

    Returns:
        int: El número total de ocurrencias de la palabra.
    """
    palabra = palabra.lower()  # Normalizar la palabra a minúsculas
    contador_palabras = 0

    for frase in frases:
        # Normalizar la frase a minúsculas, dividir en palabras y contar coincidencias
        palabras = frase.lower().split()
        contador_palabras += palabras.count(palabra)

    return contador_palabras