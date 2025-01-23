import os
import time
import base64
import imghdr
import requests
import pandas as pd
from typing import List, Tuple, Optional
from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm.notebook import tqdm

load_dotenv(dotenv_path="/Users/davidfranco/Library/CloudStorage/OneDrive-Personal/Hackio/Jupyter/Proyecto-Rentabilidad-Viviendas/src/.env")

anthropic_key = os.getenv("anthropic_key")
if not anthropic_key:
    raise ValueError("anthropic_key no está definido en las variables de entorno")


def obtener_tipo_mime(contenido: bytes) -> str:
    """Detecta el tipo MIME de una imagen."""
    formato = imghdr.what(None, contenido)
    return f"image/{formato}" if formato else "image/jpeg"


def url_a_base64_con_mime(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Convierte una imagen de URL a base64 con tipo MIME."""
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()
        contenido = respuesta.content
        return base64.b64encode(contenido).decode('utf-8'), obtener_tipo_mime(contenido)
    except Exception as e:
        print(f"Error descargando imagen {url}: {e}")
        return None, None


def preparar_imagenes_lote(urls_cocinas: List[str], urls_banios: List[str]) -> List[Tuple[str, str, str, str]]:
    """Convierte imágenes de URLs a base64 en lotes."""
    return [
        (cocina_base64, cocina_mime, banio_base64, banio_mime)
        for url_cocina, url_banio in zip(urls_cocinas, urls_banios)
        if (cocina_base64 := url_a_base64_con_mime(url_cocina)[0]) and
           (banio_base64 := url_a_base64_con_mime(url_banio)[0]) and
           (cocina_mime := url_a_base64_con_mime(url_cocina)[1]) and
           (banio_mime := url_a_base64_con_mime(url_banio)[1])
    ]

def analizar_lote_propiedades(
    cliente: Anthropic,
    imagenes_preparadas: List[Tuple[str, str, str, str]],
    batch: int = 3
) -> List[Tuple[int, int, int, int]]:
    """Analiza propiedades en lotes usando Anthropic."""
    try:
        # Construct the API content with all pairs of images in the batch
        content = []
        for idx, (cocina_base64, cocina_mime, banio_base64, banio_mime) in enumerate(imagenes_preparadas):
            content.extend([
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
                    "text": f"Property {idx + 1}: Analyze the images and provide the result."
                }
            ])

        # Add instructions only once
        content.append({
            "type": "text",
            "text": (
                """You are an AI image analysis system specialized in evaluating property conditions. Your task is to analyze multiple property images in batch and provide a precise evaluation.
Instructions for Analysis:

Analyze each pair of images:

Image 1: Kitchen
Image 2: Bathroom

Evaluation Criteria:

Ratings: Whole numbers 1-5
1: Very poor (complete renovation required)
2: Poor (major renovations needed)
3: Fair (some renovations needed)
4: Good (minor improvements required)
5: Excellent (no renovations needed)

Sizes: Whole numbers in square meters (m²)

Analysis Requirements:
A. Detailed Evaluation Format:
Assess condition of each feature and provide an aggregated rating (1-5)
Estimate size using furniture/appliance references

B. Tuple order:
Provide results as a list of tuples, one per property.
EACH tuple must be in this format: (kitchen_rating,kitchen_size,bathroom_rating,bathroom_size)

C. Output format:
[(4,10,5,3),(3,12,4,9),...]

Notes:
Remember, your output always should be a list of tuples, with no additional text or explanation.

"""

            )
        })

        # Make the API call with the updated model
        mensaje = cliente.messages.create(
            model="claude-3-haiku-20240307",  # Updated model name
            max_tokens=300,
            messages=[{"role": "user", "content": content}]
        )

        # Handle the response
        if hasattr(mensaje, "content") and hasattr(mensaje.content[0], "text"):
            respuesta_texto = mensaje.content[0].text.strip()  # Extract the text attribute
        else:
            raise ValueError("Unexpected response format from the API.")

        # Parse the response
        resultados = eval(respuesta_texto)  # Expecting the response as a list of tuples

        # Validate the response format
        if not isinstance(resultados, list) or not all(isinstance(t, tuple) and len(t) == 4 for t in resultados):
            raise ValueError("Formato de respuesta inválido")

        return resultados

    except Exception as e:
        print(f"Error procesando lote: {e}")
        return []


def analizar_propiedades(df: pd.DataFrame, batch: int = 3) -> Tuple[pd.DataFrame, List[Tuple[int, int, int, int]]]:
    """Analiza un DataFrame de propiedades por lotes."""
    cliente = Anthropic(api_key=anthropic_key)

    # Añadir columnas para resultados si no existen
    for col in ['puntuacion_cocina', 'puntuacion_banio', 'mts_cocina', 'mts_banio']:
        if col not in df.columns:
            df[col] = None

    # Filtrar URLs válidas
    indices_validos = df[df['url_cocina'].notna() & df['url_banio'].notna()].index

    resultados_totales = []
    for i in tqdm(range(0, len(indices_validos), batch), desc="Procesando lotes"):
        indices_lote = indices_validos[i:i + batch]
        urls_cocinas = df.loc[indices_lote, 'url_cocina'].tolist()
        urls_banios = df.loc[indices_lote, 'url_banio'].tolist()

        imagenes_preparadas = preparar_imagenes_lote(urls_cocinas, urls_banios)
        if not imagenes_preparadas:
            continue

        resultados = analizar_lote_propiedades(cliente, imagenes_preparadas, batch)
        resultados_totales.extend(resultados)

        for idx, (cocina_p, cocina_m, banio_p, banio_m) in zip(indices_lote, resultados):
            df.loc[idx, ['puntuacion_cocina', 'mts_cocina', 'puntuacion_banio', 'mts_banio']] = [cocina_p, cocina_m, banio_p, banio_m]

        time.sleep(1)  # Evitar saturar la API

    return df, resultados_totales