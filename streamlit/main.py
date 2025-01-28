import pandas as pd
import streamlit as st
import plotly.express as px
import ast
import sys

sys.path.append("..")
import src.soporte_rentabilidad as sr

# Set Streamlit page config
st.set_page_config(page_title="Rentabilidad Inmobiliaria", layout="wide")

# Add custom styles
st.markdown(
    """
    <style>
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

# Load the data
@st.cache_data
def load_data():
    try:
        # Load dataset
        data = pd.read_pickle("ejemplo.pkl")

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

data = load_data()

# Ensure required columns exist in data
required_columns = ["distrito", "tamanio", "precio", "puntuacion_banio", "puntuacion_cocina", "habitaciones", "urls_imagenes", "codigo"]
for col in required_columns:
    if col not in data.columns:
        st.error(f"Missing column: {col}. Please check your dataset.")
        st.stop()

# Navigation
page = st.sidebar.radio("Navegación", ["Inputs", "Resultados"])

# Session state initialization for inputs
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "porcentaje_entrada": 20.0,
        "coste_reformas": 0,
        "comision_agencia": 3.0,
        "anios": 20,
        "tin": 3.0,
        "seguro_vida": 0,
        "tipo_irpf": 19.0,
        "porcentaje_amortizacion": 0.0,
    }

if page == "Inputs":
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

elif page == "Resultados":
    st.header("Resultados y Filtros")

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

        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

        for _, row in resultados_rentabilidad.iterrows():
            image_urls = row["urls_imagenes"] if row["urls_imagenes"] else []
            rentabilidad_bruta = (
                f"{float(row['Rentabilidad Bruta']):.2f}%" 
                if pd.notna(row.get("Rentabilidad Bruta")) 
                else "N/A"
            )
            idealista_url = f"https://www.idealista.com/inmueble/{row['codigo']}/"

            # Card
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
                    st.write("Imágenes de la propiedad:")
                    render_image_carousel(image_urls)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.write("No hay propiedades que coincidan con los filtros.")