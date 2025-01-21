from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

import pandas as pd
import geopandas as gpd
import json

load_dotenv(dotenv_path="/Users/davidfranco/Library/CloudStorage/OneDrive-Personal/Hackio/Jupyter/Proyecto-Rentabilidad-Viviendas/src/.env")

mongo_uri = os.getenv("mongo_uri")
if not mongo_uri:
    raise ValueError("mongo_uri no está definido en las variables de entorno")

def mongo():
    print(mongo_uri)

# Conectar a MongoDB Atlas
def conectar_a_mongo(nombre_bd: str):
    """
    Conecta a MongoDB Atlas y devuelve un objeto de la base de datos.
    """
    cliente = MongoClient(mongo_uri, server_api=ServerApi('1'))
    return cliente[nombre_bd]

# Función para subir un DataFrame a MongoDB
def subir_dataframe_a_mongo(bd, df, nombre_coleccion):
    """
    Sube un DataFrame a una colección especificada de MongoDB.
    """
    coleccion = bd[nombre_coleccion]
    registros = df.to_dict(orient="records")
    coleccion.insert_many(registros)
    print(f"DataFrame subido a la colección: {nombre_coleccion}")

# Función para subir un GeoDataFrame a MongoDB
def subir_geodataframe_a_mongo(bd, gdf, nombre_coleccion):
    """
    Sube un GeoDataFrame a una colección especificada de MongoDB en formato GeoJSON.
    """
    coleccion = bd[nombre_coleccion]
    registros_geojson = json.loads(gdf.to_json())['features']
    coleccion.insert_many(registros_geojson)
    print(f"GeoDataFrame subido a la colección: {nombre_coleccion}")

