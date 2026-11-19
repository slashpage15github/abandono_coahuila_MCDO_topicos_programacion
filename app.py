import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy.types import CHAR, VARCHAR, FLOAT

#base de datos MariaDB    
DB_HOST = "localhost"
DB_USER = "root"
DB_PORT = 3306
DB_PASS = ""
DB_NAME = "proy_abandono_topicos_progra"
DB_TABLE = "alumnos_en_curso_escolar"

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
        st.error(f"Error al cargar 'modelo_abandono.pkl'. Asegúrate de que el archivo existe en esta carpeta: {e}")
        return None

artefacto = cargar_modelo()

st.title("Sistema de Alerta Temprana - Predicción de Abandono Escolar")
st.markdown("Analítica predictiva")

if artefacto is not None:
    pipeline = artefacto['pipeline']
    features = artefacto['features']
    nombre_modelo = artefacto.get('nombre_modelo', 'Modelo Entrenado')

    # Menú Lateral
    opcion = st.sidebar.selectbox(
        "Navegación",
        ["Predicción de Abandono", "Consulta Individual", "Diagnóstico del Modelo"]
    )



    # ----------------------------------------------------
    # MÓDULO 1: PREDICCIÓN DESDE BASE DE DATOS Y UPDATE AUTOMÁTICO
    # ----------------------------------------------------
    if opcion == "Predicción de Abandono":
        st.header("Evaluación Masiva y Actualización en la Base de Datos")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            db_host = st.text_input("Host", value=DB_HOST)
            db_user = st.text_input("Usuario", value=DB_USER)
        with col_cfg2:
            db_port = st.number_input("Puerto", value=DB_PORT)
            db_pass = st.text_input("Contraseña", type="password", value=DB_PASS)
        with col_cfg3:
            db_name = st.text_input("Base de Datos", value=DB_NAME)
            db_table = st.text_input("Tabla / Consulta SQL", value=DB_TABLE)

        if st.button("Predecir y Actualizar MySQL", type="primary"):
            try:
                # 1. Crear conexión SQLAlchemy a MariaDB
                engine_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
                engine = create_engine(engine_url)
                
                with st.spinner("Cargando registros desde la base de datos..."):
                    df_sql = pd.read_sql(f"SELECT * FROM {db_table}", engine)
                
                st.success(f"Se leyeron exitosamente **{len(df_sql)}** alumnos de la tabla `{db_table}`.")

                # Verificar variables numéricas necesarias
                columnas_faltantes = [f for f in features if f not in df_sql.columns]

                if columnas_faltantes:
                    st.error(f"La tabla no contiene las siguientes columnas requeridas por el modelo: {columnas_faltantes}")
                else:
                    # Preparar subset de datos para la predicción
                    X_eval = df_sql[features].copy()
                    
                    # Garantizar conversión a numérico
                    for col in X_eval.columns:
                        X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce')

                    # 3. Predecir probabilidades
                    probs = pipeline.predict_proba(X_eval)[:, 1]
                    
                    # Agregar predicciones al DataFrame
                    df_sql['Probabilidad_Abandono'] = np.round(probs, 4)
                    
                    def clasificar_riesgo(p):
                        if p < 0.33:
                            return 'Bajo'
                        elif p < 0.66:
                            return 'Medio'
                        else:
                            return 'Alto'

                    df_sql['Nivel_Riesgo'] = df_sql['Probabilidad_Abandono'].apply(clasificar_riesgo)

                    # 4. EJECUTAR EL UPDATE AUTOMÁTICO EN MYSQL (Tabla Temporal + JOIN)
                    id_col = 'curp' if 'curp' in df_sql.columns else df_sql.columns[0]
                    
                    with st.spinner(" Actualizando registros directamente en MySQL..."):
                        with engine.begin() as conn:
                            # Asegurar que existan las columnas en la tabla real
                            conn.execute(text(f"""
                                ALTER TABLE {DB_TABLE} 
                                ADD COLUMN IF NOT EXISTS Probabilidad_Abandono FLOAT,
                                ADD COLUMN IF NOT EXISTS Nivel_Riesgo VARCHAR(20);
                            """))
                            
                            # Volcar a tabla temporal de paso
                            
                            # Definir los tipos de datos exactos para que no se creen como TEXT
                            dtype_mapping = {
                                id_col: CHAR(18),  # Ajusta la longitud si tu ID es distinto
                                'Probabilidad_Abandono': FLOAT,
                                'Nivel_Riesgo': VARCHAR(20)
                            }
                            df_temp = df_sql[[id_col, 'Probabilidad_Abandono', 'Nivel_Riesgo']].copy()
                            df_temp.to_sql('tmp_predicciones_abandono', 
                                           con=conn, 
                                           if_exists='replace', 
                                           index=False,
                                           dtype=dtype_mapping)
                            
                            # 3. ¡Crear la Primary Key en la tabla temporal para que el JOIN sea instantáneo!
                            conn.execute(text(f"ALTER TABLE tmp_predicciones_abandono ADD PRIMARY KEY ({id_col});"))
                            
                            # Ejecutar UPDATE masivo con JOIN
                            sql_update = f"""
                                UPDATE {DB_TABLE} t
                                JOIN tmp_predicciones_abandono tmp ON t.{id_col} = tmp.{id_col}
                                SET t.Probabilidad_Abandono = tmp.Probabilidad_Abandono,
                                    t.Nivel_Riesgo = tmp.Nivel_Riesgo;
                            """
                            conn.execute(text(sql_update))
                            
                            # Limpiar tabla temporal
                            conn.execute(text("DROP TABLE IF EXISTS tmp_predicciones_abandono;"))
                    
                    st.success(f"¡Predicción realizada y **{len(df_sql)} registros actualizados** en la base de datos MySQL con éxito!")

                    # --- RESUMEN DE RESULTADOS ---
                    st.markdown("---")
                    st.subheader(" Resumen General de Riesgo Detectado")
                    
                    col_b, col_m, col_a = st.columns(3)
                    conteo = df_sql['Nivel_Riesgo'].value_counts()
                    
                    col_b.metric("🟢 Riesgo Bajo", conteo.get('Bajo', 0))
                    col_m.metric("🟡 Riesgo Medio", conteo.get('Medio', 0))
                    col_a.metric("🔴 Riesgo Alto", conteo.get('Alto', 0))

                    # --- TABLA DE RESULTADOS ---
                    st.subheader("Detalle por Alumno (Ordenado por mayor probabilidad de abandono)")
                    
                    cols_identificacion = [c for c in ['curp', 'cct', 'Nivel_Riesgo', 'Probabilidad_Abandono'] if c in df_sql.columns]
                    cols_restantes = [c for c in df_sql.columns if c not in cols_identificacion]
                    
                    df_resultado_final = df_sql[cols_identificacion + cols_restantes].sort_values(by='Probabilidad_Abandono', ascending=False)
                    st.dataframe(df_resultado_final, use_container_width=True)

                    # --- BOTÓN DE DESCARGA ---
                    csv_export = df_resultado_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar Reporte Completo en CSV",
                        data=csv_export,
                        file_name="reporte_prediccion_abandono.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

            except Exception as ex:
                st.error(f"Error durante el proceso con la base de datos: {ex}")

# ----------------------------------------------------
    # MÓDULO 2: CONSULTA INDIVIDUAL
    # ----------------------------------------------------
    elif opcion == "Consulta Individual":
        st.header("Evaluación Individual de un Alumno por CURP")
        st.write("Introduce la CURP del alumno para extraer sus datos de la base de datos y evaluar su riesgo:")
        
        POWER_BI_URL = "https://app.powerbi.com/view?r=eyJrIjoiODljZWI5YTUtYjQxNi00N2I5LTgwNDAtNDk5NTEwNTUyZjZiIiwidCI6IjUxZmZlNDRiLTNhMjktNDQ0Ny04YzNiLTE4MzdiODllMDYxNCIsImMiOjR9"
        st.link_button(
                            "📊 Abrir tablero en Power BI",
                            url=POWER_BI_URL,
                            type="secondary",
                            help="Abre el reporte publicado de Power BI en una pestaña nueva",
                        )
        
        curp_buscada = st.text_input("Introduce la CURP:", value="").strip()

        if st.button("🔍 Buscar y Evaluar Alumno", type="primary"):
            if not curp_buscada:
                st.warning("⚠️ Por favor ingresa una CURP válida.")
            else:
                try:
                    # Usamos los parámetros de conexión que ya están definidos arriba en tu app
                    engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
                    engine = create_engine(engine_url)

                    query = f"SELECT * FROM {DB_TABLE} WHERE curp = :curp_val"

                    with engine.connect() as conn:
                        df_alumno = pd.read_sql(text(query), conn, params={"curp_val": curp_buscada})

                    if df_alumno.empty:
                        st.error(f"❌ No se encontró ningún alumno con la CURP: `{curp_buscada}`")
                    else:
                        
                        #st.success("✅ ¡Alumno encontrado exitosamente!")
                        
                        # Mostrar los datos del alumno
                        st.subheader("📋 Ficha de Datos del Alumno")
                        
                        # Convertir la primera fila del alumno en un diccionario
                        datos_alumno = df_alumno.iloc[0].to_dict()
                        
                        # Extraer los datos de la primera fila como diccionario
                        datos_alumno = df_alumno.iloc[0].to_dict()
                        
                        # Preparar variables para el modelo
                        X_eval = df_alumno[features].copy()
                        for col in X_eval.columns:
                            X_eval[col] = pd.to_numeric(X_eval[col], errors='coerce')

                        prob = pipeline.predict_proba(X_eval)[0][1]
                        
                        st.markdown("---")
                        if prob < 0.33:
                            st.success(f"🟢 **Riesgo BAJO**: Probabilidad del **{prob*100:.2f}%**")
                        elif prob < 0.66:
                            st.warning(f"🟡 **Riesgo MEDIO**: Probabilidad del **{prob*100:.2f}%**")
                        else:
                            st.error(f"🔴 **Riesgo ALTO**: Probabilidad del **{prob*100:.2f}%**")                        
                        
                        
                        # Mostrar en 2 columnas estilo formulario de lectura
                        cols_form = st.columns(2)
                        for idx, (campo, valor) in enumerate(datos_alumno.items()):
                            with cols_form[idx % 2]:
                                st.text_input(
                                    label=str(campo).upper(), 
                                    value=str(valor) if pd.notna(valor) else "", 
                                    disabled=True, 
                                    key=f"field_{campo}"
                                )



                except Exception as ex_ind:
                    st.error(f"Error al consultar el alumno: {ex_ind}")

    # ----------------------------------------------------
    # MÓDULO 3: DIAGNÓSTICO DEL MODELO
    # ----------------------------------------------------
    elif opcion == "Diagnóstico del Modelo":
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
            

        # 1. Información General y Justificación de Selección
        st.subheader("¿Por qué seleccionamos este modelo?")
        st.markdown("""
        * Desbalance de clases: Dado que el abandono escolar representa un porcentaje menor del total de alumnos (~5%), la métrica AUC-PR (Area Under Precision-Recall Curve) y el Recall son los mejores criterios cuantitativos para elegir al ganador, ya que el Accuracy suele ser engañoso.
        * Escalado y Preprocesamiento: Algoritmos basados en distancia como KNeighborsClassifier requieren normalización de características (StandardScaler), mientras que modelos basados en árboles no. Un Pipeline de scikit-learn garantiza que no exista fuga de datos (Data Leakage).
        * Manejo de KNeighborsClassifier y SMOTE: Se incluye el manejo defensivo de dependencias imblearn, utilizando el parámetro nativo class_weight='balanced' en los modelos donde aplique, garantizando ejecutabilidad sin errores.
        """)

# 1. Información General y Justificación de Selección
        st.subheader("¿Cómo se interpreta el resultado para elegir el modelo?")
        st.markdown("""
        * Criterio de Selección Primario (AUC-PR): En problemas de alerta temprana donde el evento positivo es minoritario (~5%), el área bajo la curva Precision-Recall (AUC-PR) penaliza duramente los falsos positivos sin distorsionar la evaluación por la gran cantidad de verdaderos negativos.
        * Criterio de Selección Secundario (Recall / Sensibilidad): Mide la capacidad de detectar a los alumnos en riesgo real ($TP / (TP + FN)$). En el contexto escolar, un FN (Falso Negativo) es costoso porque implica perder de vista a un alumno que terminará desertando.
        * Modelos Ensamblados (HistGradientBoosting (LGBM Equivalent) / RandomForest): Suelen superar consistentemente a DecisionTreeClassifier (que sobreajusta fácilmente) y a KNeighborsClassifier (que sufre con alta dimensionalidad y grandes volúmenes de datos).
        """)

        st.subheader("Inferencias y Recomendaciones de Selección")
        st.markdown("""
        * AUC-PR (Area Under the Precision-Recall Curve o Área Bajo la Curva de Precisión-Recall) es una métrica utilizada para evaluar el rendimiento de modelos de clasificación binaria, especialmente potente y recomendada cuando se trabaja con desbalance de clases (por ejemplo, cuando la clase que buscas predecir, como la deserción escolar o una enfermedad rara, es una minoría frente al total de registros).
        * Para entenderla a fondo, se compone de dos métricas principales que se evalúan a diferentes umbrales:
        * Precisión (Precision): De todos los alumnos que el modelo predijo que van a abandonar, ¿qué porcentaje realmente abandonó? (Mide cuántas falsas alarmas evitamos).
        * Recall / Exhaustividad (Sensibilidad): De todos los alumnos que realmente abandonaron en la realidad, ¿qué porcentaje logró detectar el modelo? (Mide cuántos casos críticos logramos capturar sin que se nos escapen).
        """)

        st.subheader("¿Por qué es tan importante frente a otras métricas como el ROC-AUC?")
        st.markdown("""
        * Cuando tus datos tienen un fuerte desbalance (por ejemplo, 150,000 alumnos que continúan sus estudios y solo 2,000 que abandonan), la métrica tradicional ROC-AUC puede dar una sensación de éxito engañosa porque el número de verdaderos negativos es abrumadoramente alto, lo que distorsiona la tasa de falsos positivos.
        * En cambio, el AUC-PR se enfoca exclusivamente en la clase positiva (la minoría). Si el modelo es malo detectando los abandonos reales, el AUC-PR cae drásticamente, lo que lo convierte en el estándar de oro para auditorías estrictas en problemas de predicción de riesgo. Un AUC-PR cercano a 1.0 indica un modelo excepcional para detectar los casos críticos sin generar un volumen inmanejable de falsos positivos.
        * La diferencia principal entre ROC-AUC y AUC-PR (a veces llamada AUC-P) radica en qué métricas evalúan y cómo se comportan cuando las clases de tu dataset están desbalanceadas (como ocurre típicamente en la predicción de deserción escolar).
        """)

        st.subheader("ROC-AUC (Receiver Operating Characteristic)")
        st.markdown("""
        * Qué evalúa: Compara la Tasa de Verdaderos Positivos (Sensibilidad / Recall) frente a la Tasa de Falsos Positivos ($1 - \text{Especificidad}$) a diferentes umbrales de decisión.
        * Enfoque: Toma en cuenta tanto el rendimiento sobre la clase positiva como sobre la clase negativa.
        * El problema con el desbalance: Si tienes muchos más alumnos que continúan sus estudios en comparación con los que abandonan, el número de verdaderos negativos es masivo. Esto hace que la Tasa de Falsos Positivos se mantenga artificialmente baja, inflando la métrica ROC-AUC y haciendo que un modelo mediocre parezca excelente.
        """)

        st.subheader("AUC-PR / AUC-P (Precision-Recall)")
        st.markdown("""
        * Qué evalúa: Relaciona la Precisión (de todos los que el modelo predijo que abandonarían, cuántos lo hicieron realmente) frente al Recall (de todos los que realmente abandonaron, cuántos logró capturar el modelo).
        * Enfoque: Se concentra exclusivamente en la clase positiva (la minoría de interés) y no toma en cuenta los verdaderos negativos.
        * Por qué es ideal para el abandono escolar: Al ignorar la clase mayoritaria (los alumnos que no abandonan), es una métrica mucho más estricta y transparente. Si el modelo genera muchas falsas alarmas, la precisión se desploma y el AUC-PR lo refleja de inmediato.
        """)
        
        st.subheader("Resumen de uso:")
        st.markdown("""
        * ROC-AUC: Es útil cuando las clases están más o menos equilibradas o cuando te interesa evaluar el comportamiento general del modelo en todo el espectro binario.
        * AUC-PR: Es la métrica estándar de oro cuando trabajas con clases fuertemente desbalanceadas donde la clase minoritaria (el riesgo o abandono) es el objetivo crítico que necesitas cazar con precisión.
        """)
        
        st.subheader("SMOTE (Synthetic Minority Over-sampling Technique)")
        st.markdown("""
        * En lugar de crear muestras sintéticas con SMOTE, lo que tu script está haciendo para lidiar con el desbalance de clases es utilizar el parámetro class_weight='balanced' directamente en los algoritmos basados en árboles:
        * DecisionTreeClassifier, RandomForestClassifier y HistGradientBoostingClassifier tienen activado class_weight='balanced', lo cual penaliza más los errores en la clase minoritaria (abandono) durante el entrenamiento penalizando la función de costo.
        * En la partición de datos (train_test_split), estás usando stratify=y, lo que garantiza que tanto el conjunto de entrenamiento como el de prueba conserven exactamente la misma proporción de alumnos que abandonan y que continúan.
        * Al KNeighborsClassifier no se le está aplicando ninguna corrección específica para desbalance de clases (class_weight no existe nativamente en los KNN estándar de scikit-learn), lo que explica por qué en la tabla de resultados anterior fue el que tuvo peor desempeño en Recall y AUC-PR.
        * Como los algoritmos basados en vecinos (KNeighborsClassifier) no tienen un parámetro nativo como class_weight='balanced' (que es exclusivo de los árboles de decisión y bosques aleatorios), la forma ideal de equilibrar las clases para este modelo en específico es utilizando SMOTE.
        * Random Forest sigue siendo una opción fantástica porque te da gráficas de importancia de variables (feature_importances_) muy limpias y fáciles de interpretar para la institución.
        * HistGradientBoosting es extremadamente rápido y eficiente, con un rendimiento idéntico.
        """)

        # 4. Distribuciones de Relevancia (Consultadas directo de la BD para auditoría visual)
        st.subheader("Distribuciones Poblacionales y de Riesgo")
        st.write("Análisis visual del comportamiento de la población estudiantil evaluada:")

        try:
            # Conexión rápida para extraer un resumen analítico de la BD
            engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(engine_url)
            
            df_auditoria = pd.read_sql(f"SELECT Probabilidad_Abandono, Nivel_Riesgo, promedio_general FROM {DB_TABLE} WHERE Probabilidad_Abandono IS NOT NULL", engine)
            
            if not df_auditoria.empty:
                col_g1, col_g2 = st.columns(2)
                
                with col_g1:
                    st.markdown("**Distribución de Probabilidades Predichas**")
                    fig_hist, ax_h = plt.subplots(figsize=(6, 4))
                    ax_h.hist(df_auditoria['Probabilidad_Abandono'], bins=20, color='#336699', edgecolor='black')
                    ax_h.set_xlabel("Probabilidad de Abandono")
                    ax_h.set_ylabel("Cantidad de Alumnos")
                    st.pyplot(fig_hist)
                
                with col_g2:
                    st.markdown("**Relación: Promedio General vs Riesgo**")
                    fig_scat, ax_s = plt.subplots(figsize=(6, 4))
                    # Muestra una muestra aleatoria si son muchos datos para mantener fluidez en el gráfico
                    df_sample = df_auditoria.sample(min(5000, len(df_auditoria)))
                    scatter = ax_s.scatter(df_sample['promedio_general'], df_sample['Probabilidad_Abandono'], 
                                           c=df_sample['Probabilidad_Abandono'], cmap='coolwarm', alpha=0.6, s=15)
                    ax_s.set_xlabel("Promedio General")
                    ax_s.set_ylabel("Probabilidad de Abandono")
                    st.pyplot(fig_scat)
            else:
                st.info("Ejecuta primero la predicción masiva en el Módulo 1 para visualizar las distribuciones gráficas basadas en la base de datos.")

        except Exception as e_diag:
            st.info("Realiza la predicción masiva en el Módulo 1 para habilitar las gráficas de distribución poblacional.")
            
# 1. Verificar y mostrar la tabla de métricas directamente del artefacto
        if 'metricas_comparativas' in artefacto:
            st.subheader("Resultados de la Evaluación de Modelos (Desde el Artefacto)")
            df_metricas = artefacto['metricas_comparativas']
            st.dataframe(df_metricas, use_container_width=True)

            # 2. Generar gráfica dinámica con todas las métricas de la tabla
            st.subheader("Gráfica Comparativa de Rendimiento")
            
            # Preparar los datos para el gráfico
            df_plot = df_metricas.copy()
            if 'Modelo' in df_plot.columns:
                df_plot = df_plot.set_index('Modelo')
            else:
                df_plot = df_plot.set_index(df_plot.columns[0])
            
            # Seleccionar únicamente las columnas numéricas de puntajes
            df_plot_num = df_plot.select_dtypes(include=['number'])

            if not df_plot_num.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot_num.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', alpha=0.85)
                
                ax.set_title("Comparativa Dinámica de Algoritmos Evaluados", fontsize=12, fontweight='bold')
                ax.set_ylabel("Puntuación (0 a 1)")
                ax.set_xlabel("Modelo")
                plt.xticks(rotation=15, ha='right')
                ax.legend(title="Métricas", bbox_to_anchor=(1.02, 1), loc='upper left')
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                
                st.pyplot(fig)