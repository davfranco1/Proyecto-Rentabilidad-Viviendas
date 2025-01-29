import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import Point
from pandas import json_normalize
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import json


load_dotenv()

mongo_uri = os.getenv("mongo_uri")
if not mongo_uri:
    raise ValueError("mongo_uri no está definido en las variables de entorno")

def mongo():
    print(mongo_uri)


# Conectar a MongoDB Atlas
def conectar_a_mongo(nombre_bd: str):
    """
    Conecta a una base de datos en MongoDB Atlas y devuelve el objeto de la base de datos.

    Args:
        nombre_bd (str): Nombre de la base de datos a la que se desea conectar.

    Returns:
        pymongo.database.Database: Objeto de la base de datos MongoDB.
    """
    cliente = MongoClient(mongo_uri, server_api=ServerApi('1'))
    return cliente[nombre_bd]


# Función para subir un DataFrame a MongoDB
def subir_dataframe_a_mongo(bd, df, nombre_coleccion):
    """
    Sube un DataFrame a una colección especificada de MongoDB.

    Args:
        bd (pymongo.database.Database): Objeto de la base de datos MongoDB.
        df (pd.DataFrame): DataFrame a subir.
        nombre_coleccion (str): Nombre de la colección en MongoDB donde se insertará el DataFrame.

    Returns:
        None
    """
    coleccion = bd[nombre_coleccion]
    registros = df.to_dict(orient="records")
    coleccion.insert_many(registros)
    print(f"DataFrame subido a la colección: {nombre_coleccion}")


# Función para subir un GeoDataFrame a MongoDB
def subir_geodataframe_a_mongo(bd, gdf, nombre_coleccion):
    """
    Sube un GeoDataFrame a una colección especificada de MongoDB en formato GeoJSON.

    Args:
        bd (pymongo.database.Database): Objeto de la base de datos MongoDB.
        gdf (geopandas.GeoDataFrame): GeoDataFrame a subir.
        nombre_coleccion (str): Nombre de la colección en MongoDB donde se insertará el GeoDataFrame.

    Returns:
        None
    """
    coleccion = bd[nombre_coleccion]
    registros_geojson = json.loads(gdf.to_json())['features']
    coleccion.insert_many(registros_geojson)
    print(f"GeoDataFrame subido a la colección: {nombre_coleccion}")


# Función para eliminar una colección de MongoDB
def eliminar_coleccion(db, collection_name):
    """
    Elimina una colección de una base de datos MongoDB.

    Args:
        db (pymongo.database.Database): Objeto de la base de datos MongoDB.
        collection_name (str): Nombre de la colección a eliminar.

    Returns:
        str: Mensaje indicando si la colección fue eliminada o no existe.
    """
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        return f"La colección '{collection_name}' ha sido eliminada."
    else:
        return f"La colección '{collection_name}' no existe en la base de datos."
    

# Función para importar una colección de MongoDB a un DataFrame
def importar_a_dataframe(bd, nombre_coleccion):
    """
    Importa una colección de MongoDB a un DataFrame de pandas, manteniendo los nombres originales de las columnas 
    y eliminando las columnas '_id', 'type', y 'id'.

    Args:
        bd (pymongo.database.Database): Objeto de la base de datos MongoDB.
        nombre_coleccion (str): Nombre de la colección en MongoDB que se desea importar.

    Returns:
        pd.DataFrame: DataFrame con los datos de la colección, con las columnas específicas eliminadas.
    """
    coleccion = bd[nombre_coleccion]
    documentos = list(coleccion.find())

    if documentos:
        # Convertir documentos a DataFrame
        df = json_normalize(documentos, sep="_")
        
        # Eliminar las columnas '_id', 'type' y 'id' si existen
        columnas_a_eliminar = ["_id", "type", "id"]
        df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
        
        return df
    else:
        print(f"La colección '{nombre_coleccion}' está vacía o no existe.")
        return pd.DataFrame()


from shapely.geometry import Point

def importar_a_geodataframe(bd, nombre_coleccion):
    """
    Importa una colección de MongoDB a un GeoDataFrame de geopandas, procesando correctamente la columna 'geometry',
    eliminando columnas innecesarias y ajustando los nombres de las columnas.

    Args:
        bd (pymongo.database.Database): Objeto de la base de datos MongoDB.
        nombre_coleccion (str): Nombre de la colección en MongoDB que se desea importar.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con los datos de la colección.
    """
    coleccion = bd[nombre_coleccion]
    documentos = list(coleccion.find())

    if documentos:
        # Convertir documentos a DataFrame, preservando columnas anidadas
        df = json_normalize(documentos, sep="_")
        
        # Crear la columna 'geometry' a partir de 'geometry.coordinates'
        if "geometry_coordinates" in df.columns:
            df["geometry"] = df["geometry_coordinates"].apply(
                lambda coords: Point(coords) if isinstance(coords, list) else None
            )
        
        # Eliminar columnas '_id', 'type', 'id', 'geometry_type', y 'geometry_coordinates' si existen
        columnas_a_eliminar = ["_id", "type", "id", "geometry_type", "geometry_coordinates"]
        df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns])
        
        # Renombrar columnas para eliminar el prefijo 'properties_'
        df.columns = [col.replace("properties_", "") for col in df.columns]
        
        return gpd.GeoDataFrame(df, geometry="geometry")
    else:
        print(f"La colección '{nombre_coleccion}' está vacía o no existe.")
        return gpd.GeoDataFrame()