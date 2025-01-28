import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

import sys
sys.path.append("..")

import src.soporte_rentabilidad as sr


# Set Streamlit page config
st.set_page_config(page_title="Rentabilidad Inmobiliaria", layout="centered")

# Add custom styles
st.markdown(
    """
    <style>
    /* Customize the app's background */
    .stApp {
        background-color: #e6f7ff;
        border-radius: 15px;
        padding: 20px;
    }

    /* Style for buttons */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #0056b3;
    }

    /* Style for input elements like selectbox, number input, slider, and radio buttons */
    .stSelectbox, .stNumberInput, .stSlider, .stRadio {
        background-color: white;
        border: 2px solid #138cc6;
        border-radius: 10px;
        padding: 10px;
    }

    /* Gradient background styling (if needed for specific elements) */
    .custom-gradient {
        background: linear-gradient(
            to right,
            rgba(151, 166, 195, 0.25) 0%,
            rgba(151, 166, 195, 0.25) 50%,
            rgb(25, 75, 75) 100%
        );
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the data
@st.cache_data
def load_data():
    try:
        data = pd.read_pickle("ejemplo.pkl")
        # Ensure geometry column is converted to lat/lon
        if "geometry" in data.columns:
            data['lat'] = data['geometry'].apply(lambda x: x.y if hasattr(x, 'y') else None)
            data['lon'] = data['geometry'].apply(lambda x: x.x if hasattr(x, 'x') else None)
        else:
            st.warning("Column 'geometry' not found in the dataset.")
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

data = load_data()

# Ensure required columns exist in data
required_columns = ["distrito", "tamanio", "precio", "puntuacion_banio", "puntuacion_cocina", "habitaciones"]
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

# Navigation
page = st.sidebar.radio("Navegación", ["Inputs", "Resultados"])

# Initialize session state
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
    # Input section
    st.markdown(
    """
    <style>
    .custom-text {
        color: #007bff;
        font-size: 18px;
        font-weight: normal;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown('<p class="custom-text">Introduce los datos correspondientes a la compra y la financiación</p>', unsafe_allow_html=True)

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
    # Results section
    st.markdown(
    """
    <style>
    h2 {
        color: #073763;
        font-size: 28px;
        font-weight: normal;
    }
    </style>
    """,
    unsafe_allow_html=True
    )   

    # Now use st.header as usual
    st.header("Resultados y Filtros")

    # Filters
    col3, col4, col5 = st.columns(3)

    # District selection
    with col3:
        # Add custom styles
        st.markdown(
            """
            <style>
            .stSelectbox > div, .stMultiSelect > div {
                border-radius: 10px;
                border: 2px solid #138cc6;
                background-color: #ffffff;
                padding: 5px;
            }
            .stSelectbox > div:hover, .stMultiSelect > div:hover {
                background-color: #e6f7ff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Use st.selectbox or st.multiselect
        selected_distritos = st.multiselect(
            "Selecciona uno o más distritos",
            options=data["distrito"].unique(),
            default=data["distrito"].unique()  # Optional: Pre-select all
        )

        precio_min, precio_max = st.slider(
            "Precio (€)", 
            int(data["precio"].min()), 
            int(data["precio"].max()), 
            (int(data["precio"].min()), int(data["precio"].max()))
        )

    # Price and size sliders
    with col4:
        metros_min, metros_max = st.slider(
            "Metros cuadrados", 
            int(data["tamanio"].min()), 
            int(data["tamanio"].max()), 
            (int(data["tamanio"].min()), int(data["tamanio"].max()))
        )
        habitaciones = st.slider(
            "Cantidad de habitaciones", 
            int(data["habitaciones"].min()), 
            int(data["habitaciones"].max())
        )

    # Bathroom and kitchen condition
    with col5:
        estado_bano = st.slider("Estado del baño (1-5)", 1, 5)
        estado_cocina = st.slider("Estado de la cocina (1-5)", 1, 5)

    # Filter the data
    filtered_data = data[
        (data["distrito"].isin(selected_distritos)) &
        (data["tamanio"].between(metros_min, metros_max)) &
        (data["precio"].between(precio_min, precio_max)) &
        (data["puntuacion_banio"] >= estado_bano) &
        (data["puntuacion_cocina"] >= estado_cocina) &
        (data["habitaciones"] >= habitaciones)
    ].dropna(subset=["lat", "lon"])

    if not filtered_data.empty:
        resultados_rentabilidad = sr.calcular_rentabilidad_inmobiliaria_wrapper(
            filtered_data,
            **st.session_state.inputs
        )
        st.dataframe(resultados_rentabilidad)

        fig = px.scatter_mapbox(
            resultados_rentabilidad,
            lat="lat",
            lon="lon",
            hover_name="direccion",
            hover_data={
                "precio": True,
                "habitaciones": True,
                "tamanio": True,
                "Rentabilidad Bruta": True
            },
            color="Rentabilidad Bruta",
            size="tamanio",
            zoom=10,
            height=600
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig)
    else:
        st.write("No hay propiedades que coincidan con los filtros.")