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


# Obtiene la clave de la API desde las variables de entorno
load_dotenv(dotenv_path=".env")
anthropic_key = os.getenv("anthropic_key")
if not anthropic_key:
    raise ValueError("anthropic_key no está definido en las variables de entorno")


def obtener_tipo_mime(contenido: bytes) -> str:
    """
    Detecta el tipo MIME de una imagen utilizando su contenido en bytes.
    
    Args:
        contenido (bytes): Contenido de la imagen en formato binario.
    
    Returns:
        str: Tipo MIME de la imagen. Por defecto 'image/jpeg' si no se puede determinar.
    """
    formato = imghdr.what(None, contenido)
    return f"image/{formato}" if formato else "image/jpeg"


def url_a_base64_con_mime(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Descarga una imagen desde una URL y la convierte a base64 junto con su tipo MIME.
    
    Args:
        url (str): URL de la imagen.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: Codificación base64 y tipo MIME de la imagen.
    """
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()
        contenido = respuesta.content
        return base64.b64encode(contenido).decode('utf-8'), obtener_tipo_mime(contenido)
    except Exception as e:
        print(f"Error descargando imagen {url}: {e}")
        return None, None


def preparar_imagenes_lote(urls_cocinas: List[str], urls_banios: List[str]) -> List[Tuple[str, str, str, str]]:
    """
    Convierte imágenes de URLs a base64 en lotes, manteniendo las URLs de cocinas y baños emparejadas.
    
    Args:
        urls_cocinas (List[str]): Lista de URLs de imágenes de cocinas.
        urls_banios (List[str]): Lista de URLs de imágenes de baños.
    
    Returns:
        List[Tuple[str, str, str, str]]: Lista de tuplas con codificaciones base64 y tipos MIME de imágenes.
    """
    return [
        (cocina_base64, cocina_mime, banio_base64, banio_mime)
        for url_cocina, url_banio in zip(urls_cocinas, urls_banios)
        if (cocina_base64 := url_a_base64_con_mime(url_cocina)[0]) and
           (banio_base64 := url_a_base64_con_mime(url_banio)[0]) and
           (cocina_mime := url_a_base64_con_mime(url_cocina)[1]) and
           (banio_mime := url_a_base64_con_mime(url_banio)[1])
    ]


def analizar_lote_propiedades(
    cliente,
    imagenes_preparadas: List[Tuple[str, str, str, str]],
    batch: int = 3
) -> List[Tuple[int, int, int, int]]:
    """
    Analiza un lote de propiedades utilizando la API de Anthropic para evaluar cocinas y baños.
    
    Args:
        cliente: Cliente de la API de Anthropic.
        imagenes_preparadas (List[Tuple[str, str, str, str]]): Lista de imágenes preparadas en base64 con tipo MIME.
        batch (int): Tamaño del lote a procesar.
    
    Returns:
        List[Tuple[int, int, int, int]]: Lista de tuplas con evaluaciones (puntuaciones y tamaños).
    """
    try:
        content = []
        for idx, (cocina_base64, cocina_mime, banio_base64, banio_mime) in enumerate(imagenes_preparadas):
            content.extend([
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": cocina_mime, "data": cocina_base64}
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": banio_mime, "data": banio_base64}
                },
                {
                    "type": "text",
                    "text": f"Property {idx + 1}: Analyze the images and provide the result."
                }
            ])

        content.append({
            "type": "text",
            "text": """You are an AI image analysis system specialized in evaluating property conditions. Your task is to analyze multiple property images in batch and provide a precise evaluation.
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
        })

        mensaje = cliente.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[{"role": "user", "content": content}]
        )

        respuesta_texto = mensaje.content[0].text.strip()
        resultados = eval(respuesta_texto)

        if not isinstance(resultados, list) or not all(isinstance(t, tuple) and len(t) == 4 for t in resultados):
            raise ValueError("Formato de respuesta inválido")

        return resultados

    except Exception as e:
        print(f"Error procesando lote: {e}")
        return []


def analizar_propiedades(df: pd.DataFrame, batch: int = 3) -> Tuple[pd.DataFrame, List[Tuple[int, int, int, int]]]:
    """
    Analiza un DataFrame de propiedades dividiendo las imágenes en lotes.
    
    Args:
        df (pd.DataFrame): DataFrame con columnas `url_cocina` y `url_banio`.
        batch (int): Tamaño del lote a procesar.
    
    Returns:
        Tuple[pd.DataFrame, List[Tuple[int, int, int, int]]]: DataFrame actualizado con evaluaciones y resultados totales.
    """
    cliente = Anthropic(api_key=anthropic_key)

    for col in ['puntuacion_cocina', 'puntuacion_banio', 'mts_cocina', 'mts_banio']:
        if col not in df.columns:
            df[col] = None

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