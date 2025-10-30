import streamlit as st
import joblib
import pandas as pd

# Cargar modelos
rf_normal = joblib.load('modeloRandomForest.joblib')
rf_balanceado = joblib.load('randomForestBalanced.joblib')
rf_smote = joblib.load('randomforest_smote.joblib')

# Categorías de salida
categorias_texto = {
    1: 'muy baja',
    2: 'baja',
    3: 'media',
    4: 'alta',
    5: 'muy alta'
}

# Intervalos de indicadores ambientales
intervalos = {
    'Temperatura (°)': [(40, 49.999), (30, 39.999), (20, 29.999), (10, 19.999), (0, 9.999)],
    'Polución (ICA)': [(0, 27.259), (27.26, 54.519), (54.52, 81.779), (81.78, 109.039), (109.04, 136.299)],
    'Humedad (%)': [(0, 19.799), (19.8, 39.599), (39.6, 59.399), (59.4, 79.199), (79.2, 98.999)],
    'CO2 (PPM)': [(0, 194.399), (194.4, 388.799), (388.8, 583.199), (583.2, 777.599), (777.6, 971.999)],
    'NO2 (PPM)': [(0, 0.021), (0.022, 0.043), (0.044, 0.065), (0.066, 0.087), (0.088, 0.109)]
}

indicadoresAmbientales = ['Temperatura (°)', 'Humedad (%)', 'CO2 (PPM)', 'NO2 (PPM)']
zonas = ['Zona_Arbol Solar Juntas',
         'Zona_Arbol Solar Parque Villa Restrepo',
         'Zona_Arbol Solar Parque Ricaurte',
         'Zona_Arbol Solar Skate Park']

modelos_disponibles = {
    "Random Forest estándar": {
        "modelo": rf_normal,
        "descripcion": "Modelo Random Forest normal, no balancea las clases de contaminación. Puede favorecer la categoría más frecuente."
    },
    "Random Forest balanceado": {
        "modelo": rf_balanceado,
        "descripcion": "Random Forest con clases balanceadas. Trata de corregir el sesgo hacia las categorías más frecuentes."
    },
    "Random Forest con SMOTE": {
        "modelo": rf_smote,
        "descripcion": "Random Forest entrenado con SMOTE, genera ejemplos sintéticos para mejorar la predicción en categorías poco frecuentes."
    }
}

metricas_modelos = {
    "Modelo": [
        "Random Forest estándar",
        "Random Forest balanceado",
        "Random Forest con SMOTE"
    ],
    "Accuracy": [
        0.9616,
        0.9325,
        0.9325
    ],
    "F1-macro": [
        0.6151,
        0.6036,
        0.6036
    ],
    "Recall-macro": [
        0.5880,
        0.7553,
        0.7553
    ]
}

# Diccionario de matrices de confusión de ejemplo
# Reemplaza con las matrices reales de tus modelos
matriz_confusion = {
    "Random Forest estándar": [[16000, 181], [900, 800]],
    "Random Forest balanceado": [[15000, 1181], [300, 1400]],
    "Random Forest con SMOTE": [[15000, 1181], [300, 1400]]
}

# Funciones
def valor_a_categoria(indicador, valor):
    for i, (inf, sup) in enumerate(intervalos[indicador], 1):
        if inf <= valor <= sup:
            return i
    if valor < intervalos[indicador][-1][0]:
        return 1
    else:
        return 5

def preparar_input(example, zona_seleccionada):
    # One-hot encoding automático para la zona seleccionada
    for z in zonas:
        example[z] = 1 if z == zona_seleccionada else 0
    
    # Convertir indicadores ambientales a categoría
    for k in indicadoresAmbientales:
        example[k] = valor_a_categoria(k, example[k])
    
    return example

def predecir_modelo(modelo, example):
    X_df = pd.DataFrame([example])
    y_pred = modelo.predict(X_df)[0]
    return categorias_texto[y_pred]

# Convertir a DataFrame
df_metricas = pd.DataFrame(metricas_modelos)

# --- Streamlit UI ---
st.title("Predicción de Contaminación del Aire en Ibagué según los sensores ambientales de los árboles solares")

# Inputs
st.header("Seleccione el modelo para la predicción")
modelo_seleccionado = st.selectbox(
    "Modelo",
    options=list(modelos_disponibles.keys()),
    help="Elija el modelo que desea usar para la predicción"
)
st.info(modelos_disponibles[modelo_seleccionado]["descripcion"])
modelo_usar = modelos_disponibles[modelo_seleccionado]["modelo"]

st.header("Seleccione la zona del sensor")
zona_seleccionada = st.selectbox("Zona", zonas)

st.header("Ingrese los valores de los indicadores ambientales")
example = {}
example['Temperatura (°)'] = st.number_input("Temperatura (°)", min_value=0.0, max_value=50.0, value=25.0)
example['Humedad (%)'] = st.number_input("Humedad (%)", min_value=0.0, max_value=99.0, value=68.0)
example['CO2 (PPM)'] = st.number_input("CO2 (PPM)", min_value=0.0, max_value=972.0, value=420.0)
example['NO2 (PPM)'] = st.number_input("NO2 (PPM)", min_value=0.0, max_value=0.11, value=0.023)

# Botón de predicción
if st.button("Predecir"):
    input_modelo = preparar_input(example.copy(), zona_seleccionada)
    resultado = predecir_modelo(modelo_usar, input_modelo)
    
    # --- Pestañas dinámicas ---
    tabs = st.tabs(["Resultados", "Métricas", "Matriz de Confusión"])
    
    # Tab 1: Resultados
    with tabs[0]:
        st.header("Resultado de Predicción")
        st.write(f"La contaminación del aire es: **{resultado}**")
    
    # Tab 2: Métricas
    with tabs[1]:
        st.header("Métricas de todos los modelos")
        st.table(df_metricas)
    
   