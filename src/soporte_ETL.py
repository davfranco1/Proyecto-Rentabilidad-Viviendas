import requests
from tqdm import tqdm
from time import sleep
import pandas as pd
import pickle
import os
from datetime import datetime

# Importamos el usuario y contraseña que hemos guardado en el archivo .env, de modo que podamos utilizarlos como inputs de nuestra función.
geoapify_key = os.getenv("geoapify_key")
rapiapi_key = os.getenv("rapiapi_key")
ruta_descarga = os.getenv("ruta_descarga")


def geoconsulta_distritos(id):

    # El ID del lugar se obtiene del endpoint Geocoding, dentro de la misma API de Geoapify: https://apidocs.geoapify.com/playground/geocoding/?params=%7B%22query%22:%22zaragoza%22,%22filterValue%22:%7B%22radiusMeters%22:1000%7D,%22biasValue%22:%7B%22radiusMeters%22:1000%7D%7D&geocodingSearchType=full
    url = f"https://api.geoapify.com/v1/boundaries/consists-of?id={id}&geometry=geometry_1000&apiKey={geoapify_key}"
    response = requests.get(url)
    return response.json()

def consulta_idealista(operation, locationId, locationName, minPrice, maxPrice, paginas=1):

    #El locationID se puede obtener haciendo una consulta al endpoint https://rapidapi.com/scraperium/api/idealista7/playground/apiendpoint_1c6db49a-0793-4aa7-840b-6b8fc8868c3a.

    url = "https://idealista7.p.rapidapi.com/listhomes"
    headers = {
        "x-rapidapi-key": "d273e2c881mshda69fec8ceb12f0p1af332jsn39723f7f0eb4",
        "x-rapidapi-host": "idealista7.p.rapidapi.com"
    }

    lista_resultados = []

    for pagina in tqdm(range(1, paginas + 1)):
        querystring = {
            "order": "relevance",
            "operation": operation,
            "locationId": locationId,
            "locationName": locationName,
            "numPage": str(pagina),
            "maxItems": "40",
            "location": "es",
            "locale": "es",
            "minPrice": minPrice,
            "maxPrice": maxPrice
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        res = response.json()
        lista_resultados.append(res)
        sleep(5) #Para evitar que se salte páginas.
    
    return lista_resultados


def dataframe_idealista(lista_resultados):
    """
    Convierte los resultados de Idealista en un DataFrame de pandas con varias columnas de interés.

    Parámetros:
    lista_resultados (list): Una lista de diccionarios que contienen los resultados de las búsquedas de Idealista.

    Devuelve:
    DataFrame: Un DataFrame de pandas con columnas que incluyen las características de la casa.
    """
    anuncios = []

    for elemento in lista_resultados:
        for anuncio in elemento.get("elementList", []):

            features = anuncio.get("features")

            multimedia = anuncio.get("multimedia", {}).get("images", [])

            urls = [item.get('url', 'Sin URL') for item in multimedia]
            tags = [item.get('tag', 'Sin Tag') for item in multimedia]

            date_utc = anuncio.get("firstActivationDate")
            date = datetime.fromtimestamp(date_utc / 1000).strftime('%Y-%m-%d') if date_utc else None

            anunciante = anuncio.get("contactInfo", {}).get("commercialName", "ND")
            contact_info = anuncio.get("contactInfo", {})
            contacto = contact_info.get("phone1", {}).get("phoneNumber", "ND") 

            anuncios.append({
                "codigo": anuncio.get("propertyCode"),
                "latitud": anuncio.get("latitude"),
                "longitud": anuncio.get("longitude"),
                "precio": anuncio.get("price"),
                "precio_por_zona": anuncio.get("priceByArea"),
                "tipo": anuncio.get("propertyType"),
                "exterior": anuncio.get("exterior"),
                "planta": anuncio.get("floor"),
                "ascensor": anuncio.get("hasLift"),
                "tamanio": anuncio.get("size"),
                "habitaciones": anuncio.get("rooms"),
                "banios": anuncio.get("bathrooms"),
                "aire_acondicionado": features.get("hasAirConditioning") if features else None,
                "trastero": features.get("hasBoxRoom") if features else None,
                "terraza": features.get("hasTerrace") if features else None,
                "patio": features.get("hasGarden") if features else None,
                "estado": anuncio.get("status"),
                "direccion": anuncio.get("address"),
                "descripcion": anuncio.get("description"),
                #"distrito": anuncio.get("district"),
                "fecha": date,
                "anunciante": anunciante,
                "contacto": contacto,
                "cantidad_imagenes": anuncio.get("numPhotos"),
                #"URL": anuncio.get("url"),
                "urls_imagenes": urls,
                "tags_imagenes": tags
                
            })

    df_idealista = pd.DataFrame(anuncios)

    return df_idealista