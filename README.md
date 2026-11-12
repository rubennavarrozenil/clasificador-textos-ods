# Clasificador de textos por Objetivos de Desarrollo Sostenible

Este modelo de clasificación supervisada recibe un texto en español y le asigna el Objetivo de Desarrollo Sostenible (ODS) que mejor corresponde a su contenido. Se entrenó con textos etiquetados en 16 categorías y puede servir para una primera clasificación temática de documentos sobre desarrollo sostenible.

## Cómo funciona el modelo

1. **Limpieza.** Convierte el texto a minúsculas, elimina enlaces y caracteres no alfabéticos, y conserva las tildes y la ñ.
2. **TF-IDF.** Representa el texto mediante palabras y pares de palabras, dando más peso a los términos que ayudan a distinguir unos documentos de otros.
3. **Reducción de dimensionalidad.** `TruncatedSVD` resume la representación en 150 componentes.
4. **Clasificación.** Una regresión logística regularizada predice un ODS por texto.

## Evaluación

El conjunto del proyecto contiene 9,656 textos etiquetados. Se hizo una división estratificada de 80 % para entrenamiento y 20 % para prueba. La configuración se seleccionó por F1 macro mediante validación cruzada estratificada de tres particiones sobre el conjunto de entrenamiento.

| Métrica | Validación cruzada | Prueba (1,932 textos) |
| --- | ---: | ---: |
| F1 macro | 0.8295 | **0.8424** |
| F1 ponderado | 0.8583 | 0.8717 |
| Exactitud | 85.99 % | **87.32 %** |

El F1 macro da el mismo peso a los 16 ODS, aunque tengan distinta cantidad de ejemplos. En prueba, el ODS 4 (Educación de calidad) obtuvo un F1 de 0.954 y el ODS 8 (Trabajo decente y crecimiento económico) obtuvo 0.566. Las cifras provienen del notebook final del proyecto y corresponden a su conjunto de prueba.

## Alcance y límites

- El modelo cubre los ODS 1 a 16. El ODS 17 quedó fuera porque no tenía ejemplos en los datos de entrenamiento.
- El modelo devuelve una sola categoría, incluso cuando un texto aborda varios objetivos.
- La demo muestra las tres alternativas con mayor afinidad. Estos valores sirven para ordenar las alternativas y carecen de calibración.

## Probar el modelo

[![Probar el clasificador ODS en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clasificador-ods-ia.streamlit.app/)

Para ejecutar la demo desde la raíz del repositorio, usa Python 3.11 o posterior.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

[`modelo_clasificacion_ods_final.joblib`](modelo_clasificacion_ods_final.joblib) contiene el pipeline entrenado. [`app.py`](app.py) carga ese archivo y ejecuta la inferencia. Este repositorio contiene los archivos para ejecutar la demo. La evaluación y el entrenamiento están documentados en el notebook del proyecto.

## Créditos

Rubén Navarro y Oriana Sanjuanelo. Microproyecto 2 de Machine Learning No Supervisado, Maestría en Inteligencia Artificial.
