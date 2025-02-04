# Proyecto Rentabilidad Viviendas

## Descripción

El **Proyecto Rentabilidad Viviendas** tiene como objetivo analizar la rentabilidad de invertir en viviendas, considerando precios de venta y alquiler, estado de las propiedades y costos de renovación. Utiliza datos obtenidos de APIs, procesados y almacenados en MongoDB, y modelos de machine learning para la estimación del alquiler y evaluación del estado de cocinas y baños.

## Tabla de Contenidos

- [Requerimientos](#requerimientos)
- [Dependencias](#dependencias)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación](#instalación)
- [Uso](#uso)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Contacto](#contacto)
- [Agradecimientos](#agradecimientos)

## Requerimientos

- **Sistema Operativo:** Windows, macOS o Linux.
- **Python:** Versión 3.8 o superior.
- **MongoDB:** Base de datos utilizada para almacenar los datos procesados.

## Dependencias

Este proyecto requiere las siguientes bibliotecas:

- pandas
- numpy
- transformers
- Pillow
- requests
- tqdm
- matplotlib
- seaborn
- scipy
- statsmodels
- category_encoders
- scikit-learn
- geopandas
- shapely
- pymongo
- dotenv
- xgboost
- numpy_financial
- ultralytics
- anthropic

Las dependencias están listadas en `requirements.txt` y `environment.yml` para la instalación con pip o Conda.

## Estructura del Repositorio

```
Proyecto-Rentabilidad-Viviendas/
├── config/                      # Archivos de configuración
│   └── config.yaml               # Parámetros generales del proyecto
├── data/                         # Datos del proyecto
│   ├── raw/                      # Datos originales sin procesar
│   ├── processed/                # Datos transformados
│   └── descripcion_variables.md  # Descripción de variables extraídas
├── docs/                         # Documentación del proyecto
│   ├── report_final.pdf          # Informe final del análisis
│   └── notas/                    # Notas y documentación interna
├── notebooks/                    # Notebooks de Jupyter
│   ├── 1_Extraccion.ipynb        # Obtención de datos de APIs
│   ├── 4_CalculoRentabilidad.ipynb # Cálculo de rentabilidad inmobiliaria
├── reports/                      # Informes generados
│   └── figures/                  # Gráficos de análisis
├── src/                          # Código fuente del proyecto
│   ├── soporte_blip.py           # Análisis de imágenes con BLIP
│   ├── soporte_encoding.py       # Codificación de variables
│   ├── soporte_estadistica.py    # Análisis estadístico
│   ├── soporte_extraccion.py     # Extracción de datos de APIs
│   ├── soporte_mongo.py          # Conexión y manejo de MongoDB
│   ├── soporte_rentabilidad.py   # Modelado de rentabilidad
│   ├── soporte_scoring.py        # Evaluación de viviendas con Anthropic
│   ├── soporte_yolo.py           # Identificación de imágenes con YOLO
├── tests/                        # Pruebas unitarias
│   ├── test_data_processing.py   # Pruebas de procesamiento de datos
│   ├── test_modeling.py          # Pruebas de modelado
├── .gitignore                    # Archivos a ignorar en Git
├── environment.yml                # Configuración del entorno Conda
├── requirements.txt               # Dependencias del proyecto
├── README.md                      # Este archivo
└── LICENSE                        # Información de licencia
```

## Instalación

Para configurar el entorno, sigue estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/davfranco1/Proyecto-Rentabilidad-Viviendas.git
   cd Proyecto-Rentabilidad-Viviendas
   ```

2. **(Opcional) Crear y activar un entorno virtual:**
   ```bash
   python -m venv env
   source env/bin/activate  # En macOS/Linux
   env\Scripts\activate  # En Windows
   ```

3. **Instalar dependencias con pip:**
   ```bash
   pip install -r requirements.txt
   ```

   O con Conda:
   ```bash
   conda env create -f environment.yml
   conda activate ProyectoRentabilidad
   ```

## Uso

### 1. Extracción de Datos

Se obtienen datos de:
- **API Geoapify**: Datos geográficos de Zaragoza (`setl.geoconsulta_distritos()`).
- **API Idealista**: Información de viviendas en venta y alquiler (`setl.consulta_idealista()`).

Ejemplo de ejecución:
```python
from src import soporte_extraccion as setl
resultados_idealista_sale = setl.consulta_idealista("sale", "0-EU-ES-50-17-001-297", "Zaragoza", "60000", "150000", 12)
```

### 2. Procesamiento de Datos

- Conversión de datos geoespaciales con `geopandas`.
- Carga en MongoDB:
  ```python
  from src import soporte_mongo as sm
  bd = sm.conectar_a_mongo("ProyectoRentabilidad")
  sm.subir_geodataframe_a_mongo(bd, gdf_distritos, "distritos")
  ```

### 3. Modelado y Predicción

- **Predicción de alquiler con Machine Learning**:
  ```python
  from src import soporte_rentabilidad as sr
  paths_transformers = ["../transformers/target_encoder.pkl", "../transformers/scaler.pkl", "../transformers/model.pkl"]
  gdf = sr.predecir_alquiler(gdf, paths_transformers)
  ```

- **Identificación de imágenes con YOLO**:
  ```python
  from src import soporte_yolo as sy
  gdf, detecciones = sy.identificar_urls_habitaciones(gdf, 'urls_imagenes', drop_nulls=True)
  ```

- **Análisis de imágenes con BLIP**:
  ```python
  from src import soporte_blip as sb
  captions_cocinas = sb.generar_descripciones(gdf, 'url_cocina')
  ```

- **Evaluación de estado de cocinas y baños con Anthropic**:
  ```python
  from src import soporte_scoring as ss
  gdf, resultados = ss.analizar_propiedades(gdf, batch=3)
  ```

## Contribuciones

¡Las contribuciones son bienvenidas! Para colaborar:

1. **Haz un fork del repositorio.**
2. **Crea una rama para tu funcionalidad:**  
   ```bash
   git checkout -b feature/nueva-funcionalidad
   ```
3. **Realiza cambios y haz commit:**
   ```bash
   git commit -am "Añade nueva funcionalidad"
   ```
4. **Envía los cambios al repositorio remoto:**
   ```bash
   git push origin feature/nueva-funcionalidad
   ```
5. **Abre un pull request** con una descripción detallada de los cambios.

## Licencia

Este proyecto se distribuye bajo la [MIT License](LICENSE).

## Contacto

Para dudas o sugerencias:

- **GitHub:** [davfranco1](https://github.com/davfranco1)
- **Email:** tu_email@example.com *(Reemplaza con tu correo real)*

## Agradecimientos

Agradezco a todos los colaboradores y recursos que han contribuido a este proyecto, así como a la comunidad de analistas de datos.

---

Este README se puede actualizar a medida que el proyecto evolucione. ¡Éxito con el análisis! 🚀
