import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from fpdf import FPDF
import io
import tempfile

# Función para crear el PDF con fpdf
def create_pdf(accuracy, class_report, conf_matrix):
    # Crear un objeto PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título del PDF
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte de Predicción de Default", ln=True, align="C")
    
    # Agradecimiento por usar la aplicación
    pdf.ln(10)  # Salto de línea
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Gracias por utilizar nuestra aplicación para predecir el riesgo de default de empresas. A continuación, se presenta el resultado de la predicción y una breve explicación.")

    # Explicación del Default
    pdf.ln(10)  # Salto de línea
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="¿Qué significa Default?", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Si el resultado de la predicción es 'Default = 1', esto significa que la empresa está en riesgo de incumplir con sus obligaciones financieras. Es decir, existe una alta probabilidad de que la empresa no pueda hacer frente a sus deudas o compromisos financieros. En este caso, es recomendable que la empresa tome medidas correctivas para evitar caer en default.")

    # Explicación de no Default
    pdf.ln(10)  # Salto de línea
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="¿Qué significa No Default?", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt="Si el resultado de la predicción es 'Default = 0', esto significa que la empresa no está en riesgo de incumplir con sus obligaciones financieras. Es decir, la empresa tiene una situación financiera relativamente estable y debería poder cumplir con sus compromisos financieros a corto y largo plazo.")

    # Mensaje adicional de cierre
    pdf.ln(10)  # Salto de línea
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 10, txt="Recuerda que las predicciones son solo una referencia basada en los datos actuales y patrones históricos. Siempre es recomendable complementar esta información con el análisis de expertos y la supervisión constante de la situación financiera de la empresa.")
       
    # Guardar el archivo PDF en un archivo temporal
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name  # Retornar la ruta del archivo temporal

# Función para formatear el classification_report a un string
def classification_report_str(report):
    """
    Convierte el reporte de clasificación en un formato adecuado para ser insertado en un PDF.
    """
    # Limitar el tamaño de la salida a un formato más legible
    report_lines = report.splitlines()
    formatted_report = "\n".join(report_lines)
    return formatted_report

# Función principal
def main():
    st.title('Predicción de Default de Empresas')

   # Mostrar una pequeña explicación antes de la predicción
    st.write("""
    **Explicación de la predicción:**
    - **Default (Incumplimiento)**: Previsión de incumplimientos de las garantías sobre acciones Sobre el impago de la financiación de los accionistas que controlan las empresas cotizadas .
    - Si la predicción es **"Sí"**: Se predice que la empresa **sí** está en riesgo de default (incumplimiento).
    - Si la predicción es **"No"**: Se predice que la empresa **no** está en riesgo de default.
    """)
   
   # Explicación de las variables
    st.write("""
    **Explicación de las variables:**
    
    - **Ratio de prenda de acciones de los accionistas controladores**: Mide el porcentaje de acciones controladas por los principales accionistas que están comprometidas como garantía.
    
    - **Ratio de prenda de acciones no limitadas**: Indica el porcentaje de acciones de la empresa que están comprometidas como garantía sin restricciones de venta.
    
    - **Opinión de auditoría**: Refleja si la empresa ha recibido una opinión negativa de la auditoría en cuanto a la calidad de su información financiera.
    
    - **Degradación o negativa**: Este indicador refleja si la calificación crediticia de la empresa ha sido degradada o si se ha emitido una opinión negativa sobre su capacidad de pago.
    
    - **Ratio de otros recibos a activos totales**: Relaciona el total de los recibos pendientes con los activos totales de la empresa, proporcionando información sobre su liquidez.
    
    - **ROA (Rentabilidad sobre activos)**: Mide la rentabilidad de la empresa en relación con sus activos totales. Es un indicador clave de eficiencia.
    
    - **Ratio de deuda de activos**: Este ratio mide la proporción de deuda sobre los activos de la empresa. Un valor más alto puede indicar mayor riesgo financiero.
    
    - **Ratio de prenda de acciones de venta limitada**: Mide el porcentaje de acciones de la empresa que están comprometidas como garantía y tienen restricciones para ser vendidas.
    
    - **ROE (Rentabilidad sobre el patrimonio)**: Mide la rentabilidad generada sobre el capital propio de la empresa, mostrando qué tan eficiente es la empresa para generar ganancias a partir de sus inversiones.
    
    - **Edad de la empresa**: El número de años desde la fundación de la empresa. A menudo se asocia con la estabilidad y la madurez.
    """)


    # Cargar y mostrar los datos
    try:
        df = pd.read_csv('train.csv')
       
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")

     # Traducir las variables a español
    df.rename(columns={
        'Share pledge ratio of controlling shareholders': 'Ratio de prenda de acciones de los accionistas controladores',
        'Pledge ratio of unlimited shares': 'Ratio de prenda de acciones no limitadas',
        'audit opinion ': 'Opinión de auditoría',
        'Downgrade or negative': 'Degradación o negativa',
        'Ratio of other receivables to total assets': 'Ratio de otros recibos a activos totales',
        'ROA': 'ROA (Rentabilidad sobre activos)',
        'Asset liability ratio': 'Ratio de deuda de activos',
        'Pledge ratio of limited sale shares': 'Ratio de prenda de acciones de venta limitada',
        'ROE': 'ROE (Rentabilidad sobre el patrimonio)',
        'Enterprise age': 'Edad de la empresa',
        'IsDefault': 'EsDefault'
    }, inplace=True)

     # Definir X y y
    X = df[['Ratio de prenda de acciones de los accionistas controladores', 'Ratio de prenda de acciones no limitadas', 
            'Opinión de auditoría', 'Degradación o negativa', 'Ratio de otros recibos a activos totales', 
            'ROA (Rentabilidad sobre activos)', 'Ratio de deuda de activos', 'Ratio de prenda de acciones de venta limitada', 
            'ROE (Rentabilidad sobre el patrimonio)', 'Edad de la empresa']]
    y = df['EsDefault']

    # Crear barras deslizantes para cada característica
    user_input = {}
    for feature in X.columns:
        user_input[feature] = st.slider(feature, min_value=float(df[feature].min()), 
                                        max_value=float(df[feature].max()), 
                                        value=float(df[feature].mean()), step=0.01)

    

    # Botón para realizar la predicción
    if st.button('Cargar datos y predecir'):
        try:
            # Escalar los datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Entrenar el modelo
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            model = LogisticRegression(max_iter=100000, random_state=42)
            model.fit(X_train, y_train)

            # Predicción
            input_values = np.array([list(user_input.values())])
            input_scaled = scaler.transform(input_values)
            prediction = model.predict(input_scaled)

            # Mostrar predicción con emoticonos y texto grande
            if prediction[0] == 1:
                st.markdown(f"<h2 style='color: red;'>❌ Predicción de Default: **Sí**</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: green;'>✅ Predicción de Default: **No**</h2>", unsafe_allow_html=True)

            # Crear el PDF (sin mostrar precisión ni matriz)
            pdf_file_path = create_pdf(None, classification_report(y_test, model.predict(X_test)), None)
            with open(pdf_file_path, "rb") as f:
                st.download_button(
                    label="Descargar Reporte en PDF",
                    data=f,
                    file_name="reporte_prediccion_default.pdf",
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"Hubo un error al procesar el modelo: {e}")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()