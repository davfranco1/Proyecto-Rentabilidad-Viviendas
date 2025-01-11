import requests
from tqdm import tqdm
from time import sleep
import pandas as pd
import pickle
import os

# Importamos el usuario y contraseña que hemos guardado en el archivo .env, de modo que podamos utilizarlos como inputs de nuestra función.
geoapify_key = os.getenv("geoapify_key")
rapiapi_key = os.getenv("rapiapi_key")
ruta_descarga = os.getenv("ruta_descarga")


def geoconsulta_distritos(id):

    # El ID del lugar se obtiene de la API Geocoding: https://apidocs.geoapify.com/playground/geocoding/?params=%7B%22query%22:%22zaragoza%22,%22filterValue%22:%7B%22radiusMeters%22:1000%7D,%22biasValue%22:%7B%22radiusMeters%22:1000%7D%7D&geocodingSearchType=full
    url = "https://api.geoapify.com/v1/boundaries/consists-of?id=516ca3dd0b861fedbf594631421e8bd84440f00101f9018c46050000000000c002069203085a617261676f7a61&geometry=geometry_1000&apiKey=57d002a3495547d4aa8bf60760f28d54"
    response = requests.get(url)
    return response.json()

def consulta_idealista(operacion, locationId, locationName, paginas=1):
    """
    Realiza consultas a la API de Idealista para obtener anuncios de alquiler de viviendas en función del destino y número de páginas especificadas.

    Parámetros:
    locationId (str): El ID de la ubicación.
    locationName (str): El nombre de la ubicación.
    paginas (int): Número de páginas de resultados a consultar (por defecto es 1).

    Devuelve:
    list: Una lista de diccionarios con los resultados de las búsquedas de las páginas especificadas.
    """

    url = "https://idealista7.p.rapidapi.com/listhomes"
    headers = {
        "x-rapidapi-key": "d273e2c881mshda69fec8ceb12f0p1af332jsn39723f7f0eb4",
        "x-rapidapi-host": "idealista7.p.rapidapi.com"
    }

    lista_resultados = []

    for pagina in tqdm(range(1, paginas + 1)):
        querystring = {
            "order": "relevance",
            "operation": operacion,
            "locationId": locationId,
            "locationName": locationName,
            "numPage": str(pagina),
            "maxItems": "40",
            "location": "es",
            "locale": "es",
            "minPrice":"100000",
            "maxPrice":"200000"
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        res = response.json()
        lista_resultados.append(res)
        sleep(5)
    
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

            anuncios.append({
                "Código": anuncio.get("propertyCode"),
                "Latitud": anuncio.get("latitude"),
                "Longitud": anuncio.get("longitude"),
                "Precio": anuncio.get("price"),
                "Precio por zona": anuncio.get("priceByArea"),
                "Tipo": anuncio.get("propertyType"),
                "Exterior": anuncio.get("exterior"),
                "Planta": anuncio.get("floor"),
                "Ascensor": anuncio.get("hasLift"),
                "Tamanio": anuncio.get("size"),
                "Habitaciones": anuncio.get("rooms"),
                "Banios": anuncio.get("bathrooms"),
                "Aire Acondicionado": features.get("hasAirConditioning") if features else None,
                "Trastero": features.get("hasBoxRoom") if features else None,
                "Terraza": features.get("hasTerrace") if features else None,
                "Patio": features.get("hasGarden") if features else None,
                "Direccion": anuncio.get("address"),
                "Descripcion": anuncio.get("description"),
                "Cantidad Imagenes": anuncio.get("numPhotos"),
                #"URL": anuncio.get("url"),
                "URLs Imagenes": urls,
                "Tags Imagenes": tags
                
            })

    df_idealista = pd.DataFrame(anuncios)

    return df_idealista