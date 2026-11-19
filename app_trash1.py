import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

# ----------------------------------------------------
# 1. Configuración de la Página
# ----------------------------------------------------
st.set_page_config(
    page_title="Predicción de Abandono Escolar",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------------------------------
# 2. Cargar el Modelo Entrenado (.pkl)
# ----------------------------------------------------
@st.cache_resource
def cargar_modelo():
    try:
        artefacto = joblib.load("modelo_abandono.pkl")
        return artefacto
    except Exception as e:
        st.error(f"❌ Error al cargar 'modelo_abandono.pkl'. Asegúrate de que el archivo existe en esta carpeta: {e}")
        return None

artefacto = cargar_modelo()

st.title("🎓 Sistema de Alerta Temprana - Predicción de Abandono Escolar")
st.markdown("Analítica predictiva conectada a tu base de datos MariaDB / MySQL.")

if artefacto is not None:
    pipeline = artefacto['pipeline']
    features = artefacto['features']
    nombre_modelo = artefacto.get('nombre_modelo', 'Modelo Entrenado')

    # Menú Lateral
    opcion = st.sidebar.selectbox(
        "Navegación",
        ["🗄️ Predicción desde MariaDB / MySQL", "👤 Consulta Individual", "📊 Diagnóstico del Modelo"]
    )

    # ----------------------------------------------------
    # MÓDULO 1: PREDICCIÓN DESDE BASE DE DATOS (alumnos_en_curso_escolar)
    # ----------------------------------------------------
    if opcion == "🗄️ Predicción desde MariaDB / MySQL":
        st.header("Evaluación Masiva desde Base de Datos")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            db_host = st.text_input("Host", value="localhost")
            db_user = st.text_input("Usuario", value="root")
        with col_cfg2:
            db_port = st.number_input("Puerto", value=3306)
            db_pass = st.text_input("Contraseña", type="password", value="")
        with col_cfg3:
            db_name = st.text_input("Base de Datos", value="proy_abandono_topic_progra")
            db_table = st.text_input("Tabla / Consulta SQL", value="alumnos_en_curso_escolar")

        if st.button("🚀 Conectar y Predecir", type="primary"):
            try:
                # Crear conexión SQLAlchemy a MariaDB/MySQL (usa pymysql)
                engine_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
                engine = create_engine(engine_url)
                
                with st.spinner("Cargando registros desde la base de datos..."):
                    df_sql = pd.read_sql(f"SELECT * FROM {db_table}", engine)
                
                st.success(f"✅ Se leyeron exitosamente **{len(df_sql)}** alumnos de la tabla `{db_table}`.")

                # Verificar variables numéricas necesarias
                columnas_faltantes = [f for f in features if f not in df_sql.columns]

                if columnas_faltantes:
                    st.error(f"⚠️ La tabla no contiene las siguientes columnas requeridas por el modelo: {columnas_faltantes}")
                else:
                    # Preparar subset de datos para la predicción
                    X_eval = df_sql[features].copy()
                    
                    # Garantizar conversión a numérico
                    for col in X_eval.columns:
                        X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce')

                    # Predecir probabilidades
                    probs = pipeline.predict_proba(X_eval)[:, 1]
                    
                    # Agregar predicciones al DataFrame original
                    df_sql['Probabilidad_Abandono'] = np.round(probs, 4)
                    
                    def clasificar_riesgo(p):
                        if p < 0.33:
                            return 'Bajo'
                        elif p < 0.66:
                            return 'Medio'
                        else:
                            return 'Alto'

                    df_sql['Nivel_Riesgo'] = df_sql['Probabilidad_Abandono'].apply(clasificar_riesgo)

                    # --- RESUMEN DE RESULTADOS ---
                    st.markdown("---")
                    st.subheader("📌 Resumen General de Riesgo Detectado")
                    
                    col_b, col_m, col_a = st.columns(3)
                    conteo = df_sql['Nivel_Riesgo'].value_counts()
                    
                    col_b.metric("🟢 Riesgo Bajo", conteo.get('Bajo', 0))
                    col_m.metric("🟡 Riesgo Medio", conteo.get('Medio', 0))
                    col_a.metric("🔴 Riesgo Alto", conteo.get('Alto', 0))

                    # --- TABLA DE RESULTADOS ---
                    st.subheader("📋 Detalle por Alumno (Ordenado por mayor probabilidad de abandono)")
                    
                    # Mostrar primero las columnas prioritarias de identificación y resultado
                    cols_identificacion = [c for c in ['curp', 'cct', 'Nivel_Riesgo', 'Probabilidad_Abandono'] if c in df_sql.columns]
                    cols_restantes = [c for c in df_sql.columns if c not in cols_identificacion]
                    
                    df_resultado_final = df_sql[cols_identificacion + cols_restantes].sort_values(by='Probabilidad_Abandono', ascending=False)
                    st.dataframe(df_resultado_final, use_container_width=True)

                    # --- BOTÓN DE DESCARGA ---
                    csv_export = df_resultado_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar Reporte Completo en CSV",
                        data=csv_export,
                        file_name="reporte_prediccion_abandono.csv",
                        mime="text/csv"
                    )

            except Exception as ex:
                st.error(f"❌ Error al realizar la predicción desde la base de datos: {ex}")

    # ----------------------------------------------------
    # MÓDULO 2: CONSULTA INDIVIDUAL
    # ----------------------------------------------------
    elif opcion == "👤 Consulta Individual":
        st.header("Evaluación Individual de un Alumno")
        st.write("Introduce los valores para probar el modelo en tiempo real:")
        
        inputs = {}
        cols = st.columns(3)
        for i, feat in enumerate(features):
            with cols[i % 3]:
                inputs[feat] = st.number_input(f"{feat}", value=7.0, step=0.1)

        if st.button("Evaluar Riesgo"):
            df_ind = pd.DataFrame([inputs])
            prob = pipeline.predict_proba(df_ind)[0][1]
            
            st.markdown("---")
            if prob < 0.33:
                st.success(f"🟢 **Riesgo BAJO**: Probabilidad del **{prob*100:.2f}%**")
            elif prob < 0.66:
                st.warning(f"🟡 **Riesgo MEDIO**: Probabilidad del **{prob*100:.2f}%**")
            else:
                st.error(f"🔴 **Riesgo ALTO**: Probabilidad del **{prob*100:.2f}%**")

    # ----------------------------------------------------
    # MÓDULO 3: DIAGNÓSTICO DEL MODELO
    # ----------------------------------------------------
    elif opcion == "📊 Diagnóstico del Modelo":
        st.header(f"Información del Modelo Activo: {nombre_modelo}")
        
        if 'metricas_comparativas' in artefacto:
            st.subheader("Resultados de la Evaluación de Modelos")
            st.dataframe(artefacto['metricas_comparativas'], use_container_width=True)

        clf = pipeline.named_steps.get('clf')
        if hasattr(clf, 'feature_importances_'):
            st.subheader("Variables Más Influyentes")
            importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            importances.head(10).plot(kind='barh', color='#2b5c8f', ax=ax)
            ax.invert_yaxis()
            st.pyplot(fig)