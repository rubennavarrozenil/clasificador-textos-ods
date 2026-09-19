# Clasificador de textos por ODS

Aplicación Streamlit del Microproyecto 2 de Machine Learning No Supervisado.

## Ejecución local

1. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Iniciar la aplicación:

   ```bash
   streamlit run app.py
   ```

3. Abrir en el navegador la dirección indicada por Streamlit.

## Archivos requeridos

- `app.py`: interfaz y lógica de inferencia.
- `modelo_clasificacion_ods_final.joblib`: pipeline entrenado.
- `requirements.txt`: dependencias de ejecución.
- `.streamlit/config.toml`: tema visual.

## Alcance

El modelo clasifica textos entre los ODS 1 y 16. El ODS 17 no estaba presente en el conjunto de entrenamiento y, por tanto, no forma parte de las categorías posibles.

Los valores de afinidad mostrados permiten comparar las alternativas del modelo, pero no corresponden a probabilidades calibradas.
