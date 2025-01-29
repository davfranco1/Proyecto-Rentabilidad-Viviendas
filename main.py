import pandas as pd
import streamlit as st
import plotly.express as px
import ast
import time
from PIL import Image
import base64
from io import BytesIO

import sys
sys.path.append("../src")

import src.soporte_rentabilidad as sr
import src.soporte_mongo as sm

# Set Streamlit page config
st.set_page_config(page_title="Rentabilidad Inmobiliaria", layout="wide")

# Add custom styles
st.markdown(
    """
    <style>
    /* Style for input elements like selectbox, number input, slider, and radio buttons */
    .stSelectbox, .stNumberInput, .stSlider, .stRadio {
        background-color: white;
        border: 2px solid #138cc6;
        border-radius: 10px;
        padding: 10px;
    }


    .stApp {
        background-color: #e6f7ff;
        border-radius: 15px;
        padding: 20px;
    }

    .scrollable-container {
        height: 500px;
        overflow-y: scroll;
        padding: 10px;
        border: 1px solid #cccccc;
        border-radius: 10px;
        background-color: #f9f9f9;
    }

    .card {
        background-color: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }

    .card img {
        width: 150px;
        height: 150px;
        object-fit: cover;
        border-radius: 10px;
        border: 1px solid #cccccc;
    }

    .card-details {
        flex: 1;
        padding-right: 15px;
    }

    .card-details h3 {
        color: #007bff;
        margin-bottom: 5px;
        text-decoration: none;
    }

    .card-details h3 a {
        color: #007bff;
        text-decoration: none;
    }

    .card-details h3 a:hover {
        text-decoration: underline;
    }

    .card-details p {
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create MongoDB connection
bd = sm.conectar_a_mongo('ProyectoRentabilidad')

# Load the data
@st.cache_data
def load_data():
    try:
        # Load dataset
        #data = pd.read_pickle("ejemplo.pkl")
        data = sm.importar_a_geodataframe(bd, 'ventafinal')

        # Process geometry column if it exists
        if "geometry" in data.columns:
            data["lat"] = data["geometry"].apply(lambda x: x.y if hasattr(x, "y") else None)
            data["lon"] = data["geometry"].apply(lambda x: x.x if hasattr(x, "x") else None)

        # Convert `urls_imagenes` from string to list if needed
        if "urls_imagenes" in data.columns:
            data["urls_imagenes"] = data["urls_imagenes"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def render_image_carousel(image_urls):
    # Carousel HTML
    carousel_html = f"""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/slick-carousel/1.8.1/slick.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/slick-carousel/1.8.1/slick-theme.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/slick-carousel/1.8.1/slick.min.js"></script>
    
    <div class="carousel" style="max-width: 300px; margin: auto;">
        {"".join([f'<div><img src="{url}" style="width:100%; border-radius:10px; max-height:200px; object-fit:cover;"></div>' for url in image_urls])}
    </div>

    <script>
    $(document).ready(function() {{
        $('.carousel').slick({{
            infinite: true,
            slidesToShow: 1,
            slidesToScroll: 1,
            arrows: true,
            dots: false
        }});
    }});
    </script>
    """

    st.components.v1.html(carousel_html, height=300)

if 'page' not in st.session_state:
    st.session_state.page = "Datos de compra y financiación"

if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "porcentaje_entrada": 20.0,
        "coste_reformas": 5000,
        "comision_agencia": 3.0,
        "anios": 30,
        "tin": 3.0,
        "seguro_vida": 0,
        "tipo_irpf": 17.0,
        "porcentaje_amortizacion": 40.0,
    }

def handle_nav_change():
    st.session_state.page = st.session_state.navigation

def go_to_results():
    st.session_state.page = "Resultados"
    st.session_state.navigation = "Resultados"

# Navigation
st.sidebar.radio(
    "Navegación",
    ["Datos de compra y financiación", "Resultados", "Mapa", "Datos Completos"],
    key="navigation",
    on_change=handle_nav_change,
    index=["Datos de compra y financiación", "Resultados", "Mapa", "Datos Completos"].index(st.session_state.page)
)


# Function to convert image to Base64
def get_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Convert to PNG
    return base64.b64encode(buffered.getvalue()).decode()

# Load and resize the image with high-quality resampling
image = Image.open("images/logo_transparent.png")
image_resized = image.resize((200, 200), Image.LANCZOS)  # Best for reducing size
image_base64 = get_image_base64(image_resized)

# Center the image in the sidebar using HTML & CSS
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{image_base64}" width="200">
        </div>
        """,
        unsafe_allow_html=True
    )


data = load_data()

# Ensure required columns exist in data
required_columns = ["distrito", "tamanio", "precio", "puntuacion_banio", "puntuacion_cocina", "habitaciones", "urls_imagenes", "codigo"]
for col in required_columns:
    if col not in data.columns:
        st.error(f"Missing column: {col}. Please check your dataset.")
        st.stop()

# Streamlit app title
st.markdown("""
    <style>
    .title {
        color: #0b5394;
        font-size: 36px;
        font-weight: bold;
    }
    </style>
    <div class="title">Calculadora de Rentabilidad Inmobiliaria</div>
    """, unsafe_allow_html=True)



if st.session_state.page == "Datos de compra y financiación":
    st.markdown('<p style="color: #007bff; font-size: 18px;">Introduce los datos correspondientes a la compra y la financiación</p>', unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # General inputs
    col1.write("**Datos generales**")
    st.session_state.inputs["porcentaje_entrada"] = col1.number_input(
        "Porcentaje de entrada (%)", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=st.session_state.inputs["porcentaje_entrada"],
        key="porcentaje_entrada"
    )
    st.session_state.inputs["coste_reformas"] = col1.number_input(
        "Coste de reformas (€)", 
        min_value=0, 
        step=1000, 
        value=st.session_state.inputs["coste_reformas"],
        key="coste_reformas"
    )
    st.session_state.inputs["comision_agencia"] = col1.number_input(
        "Comisión de agencia (%)", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=st.session_state.inputs["comision_agencia"],
        key="comision_agencia"
    )
    st.session_state.inputs["seguro_vida"] = col1.number_input(
        "Seguro de vida (€)", 
        min_value=0, 
        step=50, 
        value=st.session_state.inputs["seguro_vida"],
        key="seguro_vida"
    )

    # Loan inputs
    col2.write("**Datos de financiación**")
    st.session_state.inputs["anios"] = col2.number_input(
        "Años del préstamo", 
        min_value=1, 
        step=1, 
        value=st.session_state.inputs["anios"],
        key="anios"
    )
    st.session_state.inputs["tin"] = col2.number_input(
        "Tasa de interés nominal (TIN %) ", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=st.session_state.inputs["tin"],
        key="tin"
    )
    st.session_state.inputs["tipo_irpf"] = col2.number_input(
        "Tipo de IRPF (%)", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=st.session_state.inputs["tipo_irpf"],
        key="tipo_irpf"
    )
    st.session_state.inputs["porcentaje_amortizacion"] = col2.number_input(
        "Porcentaje de amortización (%)", 
        min_value=0.0, 
        max_value=100.0, 
        step=0.1, 
        value=st.session_state.inputs["porcentaje_amortizacion"],
        key="porcentaje_amortizacion"
    )

    if st.button("Ver resultados", on_click=go_to_results):
        pass

elif st.session_state.page == "Resultados":
    st.markdown('<p style="color: #007bff; font-size: 18px;">Mostrando los resultados de tu consulta, de mayor a menor rentabilidad bruta.</p>', unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_distritos = st.multiselect(
            "Selecciona uno o más distritos",
            options=data["distrito"].unique(),
            default=data["distrito"].unique()
        )

    with col2:
        precio_min, precio_max = st.slider(
            "Precio (€)",
            int(data["precio"].min()),
            int(data["precio"].max()),
            (int(data["precio"].min()), int(data["precio"].max()))
        )
        metros_min, metros_max = st.slider(
            "Metros cuadrados",
            int(data["tamanio"].min()),
            int(data["tamanio"].max()),
            (int(data["tamanio"].min()), int(data["tamanio"].max()))
        )

    with col3:
        estado_bano = st.slider("Estado del baño (1-5)", 1, 5)
        estado_cocina = st.slider("Estado de la cocina (1-5)", 1, 5)

    # Filter data
    filtered_data = data[
        (data["distrito"].isin(selected_distritos)) &
        (data["tamanio"].between(metros_min, metros_max)) &
        (data["precio"].between(precio_min, precio_max)) &
        (data["puntuacion_banio"] >= estado_bano) &
        (data["puntuacion_cocina"] >= estado_cocina)
    ].dropna(subset=["lat", "lon"])

    if not filtered_data.empty:
        # Calculate profitability
        resultados_rentabilidad = sr.calcular_rentabilidad_inmobiliaria_wrapper(
            filtered_data,
            **st.session_state.inputs  # Aquí se pasa correctamente
        )

        for _, row in resultados_rentabilidad.iterrows():
            image_urls = row["urls_imagenes"] if row["urls_imagenes"] else []
            rentabilidad_bruta = (
                f"{float(row['Rentabilidad Bruta']):.2f}%" 
                if pd.notna(row.get("Rentabilidad Bruta")) 
                else "N/A"
            )
            idealista_url = f"https://www.idealista.com/inmueble/{row['codigo']}/"

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-details">
                        <h3><a href="{idealista_url}" target="_blank">{row.get('direccion', 'Sin dirección')}</a></h3>
                        <p><strong>Precio:</strong> €{row['precio']}</p>
                        <p><strong>Metros cuadrados:</strong> {row['tamanio']} m²</p>
                        <p><strong>Habitaciones:</strong> {row['habitaciones']}</p>
                        <p><strong>Rentabilidad Bruta:</strong> {rentabilidad_bruta}</p>
                        <p><strong>Distrito:</strong> {row['distrito']}</p>
                    </div>
                    <div>
                        <img src="{image_urls[0]}" alt="Imagen de la propiedad">
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Expandable details with carousel and bullets
            with st.expander(f"Más detalles: {row.get('direccion', 'Sin dirección')}"):
                # Display row details as bullet points
                st.markdown(
                    f"""
                    - **Precio**: €{row['precio']}
                    - **Metros cuadrados**: {row['tamanio']} m²
                    - **Habitaciones**: {row['habitaciones']}
                    - **Estado del baño**: {row['puntuacion_banio']}
                    - **Estado de la cocina**: {row['puntuacion_cocina']}
                    """
                )

                # Image carousel
                if image_urls:
                    render_image_carousel(image_urls)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("No hay propiedades que coincidan con los filtros.")

elif st.session_state.page == "Mapa":
    st.markdown('<p style="color: #007bff; font-size: 18px;">Configura tus filtros.</p>', unsafe_allow_html=True)
    selected_distritos = st.multiselect("Selecciona distritos", options=data["distrito"].unique(), default=data["distrito"].unique())
    precio_min, precio_max = st.slider("Precio (€)", int(data["precio"].min()), int(data["precio"].max()), (int(data["precio"].min()), int(data["precio"].max())))
    metros_min, metros_max = st.slider("Metros cuadrados", int(data["tamanio"].min()), int(data["tamanio"].max()), (int(data["tamanio"].min()), int(data["tamanio"].max())))
    
    filtered_data = data[(data["distrito"].isin(selected_distritos)) & (data["tamanio"].between(metros_min, metros_max)) & (data["precio"].between(precio_min, precio_max))].dropna(subset=["lat", "lon"])
    
    if not filtered_data.empty:
        # Calculate profitability
        resultados_rentabilidad = sr.calcular_rentabilidad_inmobiliaria_wrapper(
            filtered_data,
            **st.session_state.inputs
        )

        # Map visualization
        st.plotly_chart(
            px.scatter_mapbox(
                resultados_rentabilidad,
                lat="lat",
                lon="lon",
                hover_name="direccion",
                hover_data=["precio", "habitaciones", "tamanio", "Rentabilidad Bruta"],
                zoom=10,
                height=500
            ).update_layout(mapbox_style="open-street-map"),
            use_container_width=True
        )
    else:
        st.write("No hay datos para mostrar en el mapa.")


elif st.session_state.page == "Datos Completos":
    st.header("Datos completos con filtros")

    # Dropdown to select districts
    selected_distritos = st.multiselect(
        "Selecciona distritos",
        options=data["distrito"].unique(),
        default=data["distrito"].unique(),
        key="distrito_filtro"
    )

    # Slider for price range
    precio_min, precio_max = st.slider(
        "Precio (€)",
        int(data["precio"].min()), int(data["precio"].max()),
        (int(data["precio"].min()), int(data["precio"].max())),
        key="precio_filtro"
    )

    # Slider for square meters range
    metros_min, metros_max = st.slider(
        "Metros cuadrados",
        int(data["tamanio"].min()), int(data["tamanio"].max()),
        (int(data["tamanio"].min()), int(data["tamanio"].max())),
        key="metros_filtro"
    )

    # Apply filters
    filtered_data = data[
        (data["distrito"].isin(selected_distritos)) &
        (data["tamanio"].between(metros_min, metros_max)) &
        (data["precio"].between(precio_min, precio_max))
    ]

    if not filtered_data.empty:
        # Run profitability calculations
        resultados_rentabilidad = sr.calcular_rentabilidad_inmobiliaria_wrapper(
            filtered_data,
            **st.session_state.inputs
        )

        # Columns to exclude from selection
        exclude_columns = {"lat", "lon", "urls_imagenes", "url_cocina", "url_banio", "estado", "geometry"}

        # Filter available columns after calculations
        available_columns = [col for col in resultados_rentabilidad.columns if col not in exclude_columns]

        # Column selection for display
        selected_columns = st.multiselect(
            "Selecciona columnas a mostrar",
            options=available_columns,
            default=available_columns,  # Show all allowed columns by default
            key="columnas_filtro"
        )

        # Display DataFrame with selected columns
        st.dataframe(resultados_rentabilidad[selected_columns])

    else:
        st.write("No hay datos que coincidan con los filtros.")