import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from fpdf import FPDF
import xgboost as xgb
import io
import tempfile

# Función para crear el PDF con el resultado de la predicción
def create_pdf_with_prediction(accuracy, class_report, prediction, prob_default):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título del PDF
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte de Predicción de Default", ln=True, align="C")
    pdf.ln(10)

    # Resultado de la predicción
    pdf.set_font("Arial", 'B', 14)
    if prediction == 1:
        pdf.set_text_color(255, 0, 0)  # Rojo para Default
        pdf.cell(200, 10, txt="Resultado de la Predicción:  Riesgo de Default", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(200, 10, txt=f"Probabilidad de Default: {prob_default:.2%}", ln=True)
    else:
        pdf.set_text_color(0, 128, 0)  # Verde para No Default
        pdf.cell(200, 10, txt="Resultado de la Predicción:  Sin Riesgo de Default", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(200, 10, txt=f"Probabilidad de No Default: {1 - prob_default:.2%}", ln=True)
    pdf.ln(10)

    # Explicación de la predicción
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=(
        "La predicción anterior se basa en los datos ingresados y un modelo entrenado para detectar riesgos "
        "de incumplimiento financiero en empresas. Por favor, utilice esta información como referencia."
    ))
    pdf.ln(10)

    # Precisión del modelo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Precisión del Modelo: {accuracy:.2f}", ln=True)
    pdf.ln(10)

    # Reporte de clasificación
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Reporte de Clasificación:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, class_report)

    # Guardar el archivo PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# Función principal
def main():
    st.markdown("<h1 style='text-align: center;'>Descubre el Riesgo Financiero: Predice el Default de tu Empresa en Segundos 🚀</h1>", unsafe_allow_html=True)

   # Mostrar una pequeña explicación antes de la predicción
    st.markdown("<h3 style='text-align: center;'>Anticípate al Futuro Financiero: ¡Conoce el Riesgo de Inversión Ahora!</h3>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
   


    # Cargar y mostrar los datos
    try:
        df = pd.read_csv(r'C:\Users\Sengan\Desktop\Proyecto-ML\Train Data\train.csv')
       
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")

     # Traducir las variables a español
    df.rename(columns={
        'Share pledge ratio of controlling shareholders': 'Proporción de acciones en garantía de los accionistas controladores',
        'Pledge ratio of unlimited shares': 'Proporción de acciones ilimitadas en garantía',
        'audit opinion ': 'Opinión de auditoría',
        'Downgrade or negative': 'baja de calificación o evaluación negativa',
        'Ratio of other receivables to total assets': 'Relación entre otros créditos y los activos totales',
        'ROA': 'ROA (Retorno sobre activos)',
        'Asset liability ratio': 'Relación entre pasivos y activos ajustados',
        'Pledge ratio of limited sale shares': 'Proporción de acciones limitadas en garantía',
        'ROE': 'ROE (Retorno sobre patrimonio)',
        'Enterprise age': 'Edad de la empresa',
        'IsDefault': 'EsDefault'
    }, inplace=True)

     # Definir X y y
    X = df[['Proporción de acciones en garantía de los accionistas controladores', 'Proporción de acciones ilimitadas en garantía', 
            'Opinión de auditoría', 'baja de calificación o evaluación negativa', 'Relación entre otros créditos y los activos totales', 
            'ROA (Retorno sobre activos)', 'Relación entre pasivos y activos ajustados', 'Proporción de acciones limitadas en garantía', 
            'ROE (Retorno sobre patrimonio)', 'Edad de la empresa']]
    y = df['EsDefault']

      # Crear barras deslizantes y entradas manuales con descripciones
    user_input = {}

    with st.expander("Proporción de acciones en garantía de los accionistas controladores"):
        st.write("""
        **Definición:** Mide el porcentaje de acciones controladas por los principales accionistas que están comprometidas como garantía.
        """)
    user_input['Proporción de acciones en garantía de los accionistas controladores'] = st.slider(
            '', 0.0, 1.0, 0.5, 0.01, key=0
        )

    with st.expander("Proporción de acciones ilimitadas en garantía"):
        st.write("""
        **Definición:** Indica el porcentaje de acciones de la empresa que están comprometidas como garantía sin restricciones de venta.
        """)
    user_input['Proporción de acciones ilimitadas en garantía'] = st.slider(
            '', 0.0, 1.0, 0.5, 0.01, key = 1
        )

    with st.expander("Opinión de auditoría"):
        st.write("""
        **Definición:** Refleja si la empresa ha recibido una opinión negativa de la auditoría en cuanto a la calidad de su información financiera.
        """)
    user_input['Opinión de auditoría'] = st.selectbox(
            '', [0, 1], index=0, key = 2
        )

    with st.expander("baja de calificación o evaluación negativa"):
        st.write("""
        **Definición:** Refleja si la calificación crediticia de la empresa ha sido degradada o si se ha emitido una opinión negativa sobre su capacidad de pago.
        """)
    user_input['baja de calificación o evaluación negativa'] = st.selectbox(
            '', [0, 1], index=0, key = 3
        )

    with st.expander("Relación entre otros créditos y los activos totales"):
        st.write("""
        **Definición:** Relaciona el total de los recibos pendientes con los activos totales de la empresa, proporcionando información sobre su liquidez.
        """)
    user_input['Relación entre otros créditos y los activos totales'] = st.slider(
            '', 0.0, 1.0, 0.5, 0.01, key = 4
        )

    with st.expander("ROA (Retorno sobre activos)"):
        st.write("""
        **Definición:** Mide la rentabilidad de la empresa en relación con sus activos totales. Es un indicador clave de eficiencia.
        """)
    user_input['ROA (Retorno sobre activos)'] = st.slider(
            '', 0.0, 2.0, 0.0, 0.01, key = 5
        )

    with st.expander("Relación entre pasivos y activos ajustados"):
        st.write("""
        **Definición:** Mide la proporción de deuda sobre los activos de la empresa. Un valor más alto puede indicar mayor riesgo financiero.
        """)
    user_input['Relación entre pasivos y activos ajustados'] = st.slider(
            '', 0.0, 2.0, 1.0, 0.01, key = 6
        )

    with st.expander("Proporción de acciones limitadas en garantía"):
        st.write("""
        **Definición:** Mide el porcentaje de acciones de la empresa que están comprometidas como garantía y tienen restricciones para ser vendidas.
        """)
    user_input['Proporción de acciones limitadas en garantía'] = st.slider(
            '', 0.0, 1.0, 0.5, 0.01, key = 7
        )

    with st.expander("ROE (Retorno sobre patrimonio)"):
        st.write("""
        **Definición:** Mide la rentabilidad generada sobre el capital propio de la empresa, mostrando qué tan eficiente es para generar ganancias.
        """)
    user_input['ROE (Retorno sobre patrimonio)'] = st.slider(
            '', -10.0, 2.0, -4.0, 0.01, key = 8
        )

    with st.expander("Edad de la empresa"):
        st.write("""
        **Definición:** El número de años desde la fundación de la empresa. A menudo se asocia con la estabilidad y la madurez.
        """)
    user_input['Edad de la empresa'] = st.number_input(
            '', min_value=0, max_value=200, value=10, step=1, key = 9
        )

    

    if st.button("Predecir"):
        try:
            # Dividir en datos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Entrenar el modelo
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)

            # Validar las entradas del usuario
            if len(user_input) != len(X.columns):
                st.error("Por favor, completa todas las entradas para realizar la predicción.")
                return

            # Convertir las entradas del usuario a un array
            input_values = np.array([list(user_input.values())])

            # Realizar la predicción
            prediction = model.predict(input_values)[0]
            probabilities = model.predict_proba(input_values)[0]  # Obtener las probabilidades

            # Mostrar el resultado y la probabilidad
            prob_default = probabilities[1]  # Probabilidad de "Default = 1"
            prob_no_default = probabilities[0]  # Probabilidad de "No Default = 0"

            if prediction == 1:
                st.markdown(
                    f"<h2 style='color: red;'> Predicción de Default: **Sí**</h2>"
                    f"<p>Probabilidad de Default: {prob_default:.2%}</p>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<h2 style='color: green;'> Predicción de Default: **No**</h2>"
                    f"<p>Probabilidad de No Default: {prob_no_default:.2%}</p>",
                    unsafe_allow_html=True,
                )

            # Calcular métricas
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Generar el PDF
            pdf_file_path = create_pdf_with_prediction(acc, report, prediction, prob_default)
            with open(pdf_file_path, "rb") as f:
                st.download_button(
                    label="Descargar Reporte en PDF",
                    data=f,
                    file_name="reporte_prediccion_default.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Hubo un error al realizar la predicción: {e}")
# Ejecutar la aplicación
if __name__ == "__main__":
    main()