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
load_dotenv(dotenv_path="/Users/davidfranco/Library/CloudStorage/OneDrive-Personal/Hackio/Jupyter/Proyecto-Rentabilidad-Viviendas/src/.env")
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

def url_a_base64_con_mime(url: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Descarga una imagen desde una URL y la convierte a base64 junto con su tipo MIME.
    
    Args:
        url (Optional[str]): URL de la imagen. Puede ser None.
    
    Returns:
        Tuple[Optional[str], Optional[str]]: Codificación base64 y tipo MIME de la imagen.
        Retorna (None, None) si la URL es None o hay un error.
    """
    if url is None:
        return None, None
        
    try:
        respuesta = requests.get(url)
        respuesta.raise_for_status()
        contenido = respuesta.content
        return base64.b64encode(contenido).decode('utf-8'), obtener_tipo_mime(contenido)
    except Exception as e:
        print(f"Error descargando imagen {url}: {e}")
        return None, None

def preparar_imagenes_lote(
    urls_cocinas: List[Optional[str]], 
    urls_banios: List[Optional[str]], 
    allow_partial: bool = True
) -> List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """
    Convierte imágenes de URLs a base64 en lotes, procesando cocinas y baños independientemente.
    
    Args:
        urls_cocinas (List[Optional[str]]): Lista de URLs de imágenes de cocinas. Pueden ser None.
        urls_banios (List[Optional[str]]): Lista de URLs de imágenes de baños. Pueden ser None.
        allow_partial (bool): Si es True, procesa propiedades con solo una imagen disponible.
                            Si es False, requiere ambas imágenes.
    
    Returns:
        List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]]: 
        Lista de tuplas con codificaciones base64 y tipos MIME de imágenes.
        Los valores serán None para las imágenes no disponibles.
    """
    resultados = []
    
    for url_cocina, url_banio in zip(urls_cocinas, urls_banios):
        # Procesar cocina
        cocina_base64, cocina_mime = url_a_base64_con_mime(url_cocina)
        
        # Procesar baño
        banio_base64, banio_mime = url_a_base64_con_mime(url_banio)
        
        # Si no se permite parciales, ambas imágenes deben estar presentes
        if not allow_partial and (not cocina_base64 or not banio_base64):
            continue
            
        # Si se permiten parciales, al menos una imagen debe estar presente
        if allow_partial and (cocina_base64 or banio_base64):
            resultados.append((cocina_base64, cocina_mime, banio_base64, banio_mime))
        
    return resultados


def analizar_lote_propiedades(
    cliente,
    imagenes_preparadas: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]],
    batch: int = 3
) -> List[Tuple[int, int, int, int]]:
    """
    Analiza un lote de propiedades utilizando la API de Anthropic para evaluar cocinas y/o baños.
    Procesa imágenes individuales si solo una está disponible.

    Args:
        cliente: Cliente de la API de Anthropic.
        imagenes_preparadas: Lista de tuplas (cocina_base64, cocina_mime, banio_base64, banio_mime).
            Cualquier elemento puede ser None si la imagen no está disponible.
        batch (int): Tamaño del lote a procesar.

    Returns:
        List[Tuple[int, int, int, int]]: Lista de tuplas con evaluaciones (puntuaciones y tamaños).
        Para imágenes faltantes, los valores correspondientes serán 0.
    """
    try:
        content = []
        for idx, (cocina_base64, cocina_mime, banio_base64, banio_mime) in enumerate(imagenes_preparadas):
            # Agregar imágenes disponibles
            if cocina_base64 and cocina_mime:
                content.extend([
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": cocina_mime, "data": cocina_base64}
                    },
                    {
                        "type": "text",
                        "text": f"Property {idx + 1} Kitchen: Analyze this kitchen image."
                    }
                ])
            
            if banio_base64 and banio_mime:
                content.extend([
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": banio_mime, "data": banio_base64}
                    },
                    {
                        "type": "text",
                        "text": f"Property {idx + 1} Bathroom: Analyze this bathroom image."
                    }
                ])

        if not content:
            return []

        content.append({
            "type": "text",
            "text": """You are an AI image analysis system specialized in evaluating property conditions. Your task is to analyze property images in batch and provide a precise evaluation.

            Instructions for Analysis:
            Analyze each image:
            - Kitchen images: evaluate condition and size
            - Bathroom images: evaluate condition and size

            Evaluation Criteria:
            Ratings: Whole numbers 1-5
            1: Very poor (complete renovation required)
            2: Poor (major renovations needed)
            3: Fair (some renovations needed)
            4: Good (minor improvements required)
            5: Excellent (no renovations needed)
            Sizes: Whole numbers in square meters (m²)

            Analysis Requirements:
            A. For each property, provide a tuple of 4 numbers: (kitchen_rating, kitchen_size, bathroom_rating, bathroom_size)
            B. If an image is missing, use (0, 0) for that room's values
            C. Output format must be a list of tuples: [(4,10,0,0), (0,0,4,9), (3,12,4,9),...]

            Notes:
            Remember, your output must be ONLY a list of tuples, with no additional text or explanation.
            Each property must have all 4 values, using 0 for missing images.
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
    Analiza las propiedades de un DataFrame en lotes, evaluando imágenes de cocina y/o baño.
    Procesa propiedades incluso si solo tienen una de las dos imágenes disponibles.

    Args:
        df (pd.DataFrame): DataFrame que debe contener las columnas "url_cocina" y "url_banio".
        batch (int, opcional): Tamaño del lote a procesar. Por defecto es 3.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[int, int, int, int]]]:
        - DataFrame actualizado con las columnas:
            "puntuacion_cocina": Evaluación de la cocina (0 si url_cocina es nula).
            "mts_cocina": Medida en metros cuadrados de la cocina (0 si url_cocina es nula).
            "puntuacion_banio": Evaluación del baño (0 si url_banio es nula).
            "mts_banio": Medida en metros cuadrados del baño (0 si url_banio es nula).
        - Lista de tuplas con los resultados.
    """
    cliente = Anthropic(api_key=anthropic_key)

    # Inicializar columnas si no existen
    for col in ['puntuacion_cocina', 'puntuacion_banio', 'mts_cocina', 'mts_banio']:
        if col not in df.columns:
            df[col] = None

    # Asignar ceros donde no hay URLs
    df.loc[df['url_cocina'].isna(), ['puntuacion_cocina', 'mts_cocina']] = [0, 0]
    df.loc[df['url_banio'].isna(), ['puntuacion_banio', 'mts_banio']] = [0, 0]

    # Obtener índices donde al menos una URL es válida
    indices_validos = df[df['url_cocina'].notna() | df['url_banio'].notna()].index
    resultados_totales = []

    # Procesar en lotes
    for i in tqdm(range(0, len(indices_validos), batch), desc="Procesando lotes"):
        indices_lote = indices_validos[i:i + batch]
        urls_cocinas = df.loc[indices_lote, 'url_cocina'].tolist()
        urls_banios = df.loc[indices_lote, 'url_banio'].tolist()

        # Preparar imágenes disponibles
        imagenes_preparadas = preparar_imagenes_lote(urls_cocinas, urls_banios, allow_partial=True)
        
        if not imagenes_preparadas:
            # Si la preparación falla, asignar ceros a este lote
            for idx in indices_lote:
                df.loc[idx, ['puntuacion_cocina', 'mts_cocina', 'puntuacion_banio', 'mts_banio']] = [0, 0, 0, 0]
                resultados_totales.append((0, 0, 0, 0))
            continue

        # Analizar el lote
        resultados = analizar_lote_propiedades(cliente, imagenes_preparadas, batch)
        resultados_totales.extend(resultados)

        # Actualizar resultados en el DataFrame
        for idx, (cocina_p, cocina_m, banio_p, banio_m) in zip(indices_lote, resultados):
            df.loc[idx, ['puntuacion_cocina', 'mts_cocina', 'puntuacion_banio', 'mts_banio']] = [
                cocina_p, cocina_m, banio_p, banio_m
            ]

        time.sleep(1)  # Evitar saturar la API

    return df, resultados_totales