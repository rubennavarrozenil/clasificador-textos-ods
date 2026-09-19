import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Clasificador de textos por ODS",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)


ODS = {
    1: ("Fin de la pobreza", "Promover condiciones de vida dignas y reducir la pobreza en todas sus formas."),
    2: ("Hambre cero", "Mejorar la seguridad alimentaria, la nutrición y la agricultura sostenible."),
    3: ("Salud y bienestar", "Garantizar una vida sana y promover el bienestar para todas las personas."),
    4: ("Educación de calidad", "Impulsar una educación inclusiva, equitativa y de calidad."),
    5: ("Igualdad de género", "Lograr la igualdad de género y fortalecer los derechos de mujeres y niñas."),
    6: ("Agua limpia y saneamiento", "Asegurar la disponibilidad y la gestión sostenible del agua."),
    7: ("Energía asequible y no contaminante", "Ampliar el acceso a energía segura, sostenible y moderna."),
    8: ("Trabajo decente y crecimiento económico", "Promover empleo productivo, trabajo decente y crecimiento sostenible."),
    9: ("Industria, innovación e infraestructura", "Desarrollar infraestructura resiliente e impulsar la innovación."),
    10: ("Reducción de las desigualdades", "Reducir las desigualdades dentro de los países y entre ellos."),
    11: ("Ciudades y comunidades sostenibles", "Construir ciudades inclusivas, seguras, resilientes y sostenibles."),
    12: ("Producción y consumo responsables", "Promover modalidades sostenibles de producción y consumo."),
    13: ("Acción por el clima", "Adoptar medidas frente al cambio climático y sus efectos."),
    14: ("Vida submarina", "Conservar y utilizar sosteniblemente los océanos y recursos marinos."),
    15: ("Vida de ecosistemas terrestres", "Proteger ecosistemas terrestres y detener la pérdida de biodiversidad."),
    16: ("Paz, justicia e instituciones sólidas", "Fortalecer instituciones eficaces, justicia y sociedades pacíficas."),
}


def limpiar_textos(textos):
    """Limpieza utilizada durante el entrenamiento del pipeline."""
    textos_limpios = []
    for texto in textos:
        texto = str(texto).lower()
        texto = re.sub(r"http\S+|www\.\S+", " ", texto)
        texto = re.sub(r"[^a-zA-ZáéíóúñÁÉÍÓÚÑüÜ\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        textos_limpios.append(texto)
    return textos_limpios


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo():
    ruta_modelo = Path(__file__).with_name("modelo_clasificacion_ods_final.joblib")
    return joblib.load(ruta_modelo)


st.markdown(
    """
    <style>
    .stApp {
        background: var(--background-color);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    .eyebrow {
        color: var(--primary-color);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        color: var(--text-color);
        font-size: clamp(2.15rem, 4vw, 3.55rem);
        line-height: 1.06;
        font-weight: 760;
        letter-spacing: -0.035em;
        margin: 0;
    }
    .hero-copy {
        color: var(--text-color);
        opacity: 0.78;
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 760px;
        margin: 1rem 0 1.8rem 0;
    }
    .result-card {
        background: linear-gradient(135deg, #0f766e 0%, #155e75 100%);
        border-radius: 18px;
        padding: 1.65rem 1.8rem;
        color: white;
        box-shadow: 0 18px 45px rgba(15, 118, 110, 0.18);
        margin: 0.35rem 0 1.4rem 0;
    }
    .result-kicker {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.8;
    }
    .result-title {
        font-size: 1.85rem;
        line-height: 1.2;
        font-weight: 750;
        margin: 0.35rem 0 0.5rem 0;
    }
    .result-description {
        font-size: 1rem;
        line-height: 1.55;
        opacity: 0.92;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(127, 127, 127, 0.25);
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    .stButton > button {
        border: 0;
        border-radius: 10px;
        background: #0f766e;
        color: white;
        font-weight: 700;
        min-height: 3rem;
        box-shadow: 0 8px 22px rgba(15, 118, 110, 0.18);
    }
    .stButton > button:hover {
        background: #115e59;
        color: white;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Acerca del modelo")
    st.markdown(
        """
        Este prototipo clasifica textos en los **16 ODS presentes en el conjunto de entrenamiento**.

        **Pipeline**
        - Limpieza de texto
        - Representación TF-IDF
        - TruncatedSVD: 150 componentes
        - Regresión logística

        **Desempeño en prueba**
        - F1 macro: **0.8424**
        - Exactitud: **0.8732**
        """
    )
    st.info(
        "El ODS 17 no aparece en los datos de entrenamiento; por ello, el modelo no puede producir esa categoría.",
        icon="ℹ️",
    )


st.markdown('<div class="eyebrow">Microproyecto 2 · Maestría en Inteligencia Artificial</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Clasificador de textos por Objetivos de Desarrollo Sostenible</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Escribe un fragmento relacionado con políticas públicas, desarrollo o problemáticas sociales y ambientales. El modelo analizará su contenido y propondrá el ODS con mayor afinidad.</p>',
    unsafe_allow_html=True,
)


try:
    modelo = cargar_modelo()
except Exception as error:
    st.error("No fue posible cargar el modelo. Verifica que el archivo del modelo se encuentre junto a app.py.")
    st.exception(error)
    st.stop()


with st.form("formulario_clasificacion", clear_on_submit=False):
    texto = st.text_area(
        "Texto para analizar",
        height=210,
        placeholder=(
            "Ejemplo: El programa busca ampliar el acceso a servicios de salud preventiva "
            "y vacunación en comunidades rurales..."
        ),
        help="Puedes pegar un párrafo completo. El texto se procesará con el mismo pipeline utilizado en el proyecto.",
    )
    enviar = st.form_submit_button("Clasificar texto", width="stretch")


if enviar:
    texto_normalizado = texto.strip()
    if not texto_normalizado:
        st.warning("Escribe un texto antes de solicitar la clasificación.", icon="✍️")
    else:
        prediccion = int(modelo.predict([texto_normalizado])[0])
        nombre_ods, descripcion_ods = ODS[prediccion]

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-kicker">Clasificación principal</div>
                <div class="result-title">ODS {prediccion} · {nombre_ods}</div>
                <p class="result-description">{descripcion_ods}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        palabras = len(re.findall(r"\b\w+\b", texto_normalizado, flags=re.UNICODE))
        caracteres = len(texto_normalizado)
        col1, col2, col3 = st.columns(3)
        col1.metric("Palabras analizadas", f"{palabras:,}".replace(",", "."))
        col2.metric("Caracteres", f"{caracteres:,}".replace(",", "."))
        col3.metric("Categorías posibles", "16 ODS")

        if hasattr(modelo, "predict_proba"):
            afinidades = modelo.predict_proba([texto_normalizado])[0]
            clases = np.asarray(modelo.classes_, dtype=int)
            top_indices = np.argsort(afinidades)[::-1][:3]
            top_ods = clases[top_indices]
            top_df = pd.DataFrame({
                "ODS": [f"ODS {ods} · {ODS[int(ods)][0]}" for ods in top_ods],
                "Afinidad relativa": afinidades[top_indices] * 100,
            })

            st.markdown("### ODS con mayor afinidad relativa")
            st.bar_chart(
                top_df,
                x="ODS",
                y="Afinidad relativa",
                color="#0f766e",
                width="stretch",
            )
            tabla = top_df.copy()
            tabla["Afinidad relativa"] = tabla["Afinidad relativa"].map(lambda valor: f"{valor:.1f} %")
            st.dataframe(tabla, hide_index=True, width="stretch")
            st.caption(
                "Los valores permiten comparar las alternativas del modelo, pero no deben interpretarse como probabilidades calibradas."
            )

        with st.expander("Cómo se obtuvo esta clasificación"):
            st.markdown(
                "El texto se procesó mediante el mismo pipeline del proyecto: limpieza, TF-IDF, reducción de "
                "dimensionalidad con TruncatedSVD y clasificación mediante regresión logística. El modelo no se "
                "reentrena con el texto ingresado."
            )


st.divider()
st.caption("Proyecto académico · Machine Learning No Supervisado · Rubén Navarro y Oriana Sanjuanelo")
