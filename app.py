import streamlit as st
import pandas as pd
import torch
import plotly.graph_objects as go
from prophet import Prophet
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error
import time
import os
import pickle
from io import BytesIO
from datetime import datetime

# =====================================================
# 1. CONFIGURACIÓN INICIAL
# =====================================================

st.set_page_config(page_title="Pronóstico IA - Amandau", layout="wide")

# --- ESTILOS CSS PARA BORDES ---
st.markdown("""
<style>
    /* Bordes para KPIs */
    div[data-testid="stMetric"] {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: #fafafa;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] label {
        font-weight: 500;
    }
    
     /* Borde para el gráfico */
    .stPlotlyChart {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 10px;
        margin-bottom: 100px;  /* Aumentado para más espacio */
    }
    
    /* Borde para recuadros de filtros */
    .filtros-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #fafafa;
        margin-bottom: 15px;
    }
    .filtros-container h4 {
        margin-top: 0;
        margin-bottom: 10px;
        color: #333;
        font-size: 16px;
    }
    
    /* Estilo para selectbox con selección múltiple */
    .stMultiSelect [data-baseweb="select"] span {
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

# Directorio de proyectos
PROYECTOS_DIR = "proyectos"
if not os.path.exists(PROYECTOS_DIR):
    os.makedirs(PROYECTOS_DIR)

# Usuarios autorizados
USUARIOS = {
    "amandau": "amandau_2026",
    "analista": "pronostico2024"
}

# =====================================================
# 2. FUNCIONES DE AUTENTICACIÓN Y PROYECTOS
# =====================================================

def verificar_login(usuario, password):
    return usuario in USUARIOS and USUARIOS[usuario] == password

def guardar_proyecto(nombre_proyecto, df_final, df_agg, fechas_dt, usar_colaborado, 
                     horizonte, nombres_columnas_pron, rango_ventas, hist_totales):
    proyecto_path = os.path.join(PROYECTOS_DIR, f"{nombre_proyecto}.pkl")
    datos = {
        'nombre': nombre_proyecto,
        'df_final': df_final,
        'df_agg': df_agg,
        'fechas_dt': fechas_dt,
        'usar_colaborado': usar_colaborado,
        'horizonte': horizonte,
        'nombres_columnas_pron': nombres_columnas_pron,
        'rango_ventas': rango_ventas,
        'hist_totales': hist_totales,
        'fecha_creacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(proyecto_path, 'wb') as f:
        pickle.dump(datos, f)
    return True

def cargar_proyecto(nombre_proyecto):
    proyecto_path = os.path.join(PROYECTOS_DIR, f"{nombre_proyecto}.pkl")
    if os.path.exists(proyecto_path):
        with open(proyecto_path, 'rb') as f:
            return pickle.load(f)
    return None

def listar_proyectos():
    proyectos = []
    for archivo in os.listdir(PROYECTOS_DIR):
        if archivo.endswith('.pkl'):
            nombre = archivo[:-4]
            with open(os.path.join(PROYECTOS_DIR, archivo), 'rb') as f:
                datos = pickle.load(f)
                proyectos.append({
                    'nombre': nombre,
                    'fecha_creacion': datos.get('fecha_creacion', 'Desconocida'),
                    'horizonte': datos.get('horizonte', 12)
                })
    return sorted(proyectos, key=lambda x: x['fecha_creacion'], reverse=True)

def eliminar_proyecto(nombre_proyecto):
    proyecto_path = os.path.join(PROYECTOS_DIR, f"{nombre_proyecto}.pkl")
    if os.path.exists(proyecto_path):
        os.remove(proyecto_path)
        return True
    return False

# =====================================================
# 3. FUNCIONES DE MODELOS
# =====================================================

@st.cache_resource
def cargar_modelo():
    return ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
        torch_dtype=torch.float32
    )

pipeline = cargar_modelo()

def fit_prophet(serie, periods, regressors_df=None):
    try:
        df = pd.DataFrame({'ds': serie.index, 'y': serie.values})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, uncertainty_samples=0)
        if regressors_df is not None:
            df = df.merge(regressors_df, left_on='ds', right_index=True, how='left')
            for col in regressors_df.columns:
                model.add_regressor(col)
        model.fit(df, iter=300)

        future = model.make_future_dataframe(periods=periods, freq='MS')
        if regressors_df is not None:
            future = future.merge(regressors_df, left_on='ds', right_index=True, how='left')
            future = future.fillna(0)

        forecast = model.predict(future)
        preds = forecast['yhat'].iloc[-periods:].values
        preds = [max(0, float(p)) for p in preds]
        if len(preds) > periods:
            preds = preds[:periods]
        elif len(preds) < periods:
            preds += [0.0] * (periods - len(preds))
        return preds
    except Exception:
        last_val = float(serie.iloc[-1])
        return [last_val] * periods

def fit_chronos(serie, periods):
    try:
        context = torch.tensor(serie.values, dtype=torch.float32)
        forecast = pipeline.predict(context, prediction_length=periods)
        preds = forecast.mean(dim=0).numpy().flatten()
        preds = [max(0, float(p)) for p in preds]
        if len(preds) > periods:
            preds = preds[:periods]
        elif len(preds) < periods:
            preds += [0.0] * (periods - len(preds))
        return preds
    except Exception:
        last_val = float(serie.iloc[-1])
        return [last_val] * periods

# =====================================================
# 4. FUNCIONES DE PROCESAMIENTO
# =====================================================

def procesar_archivo(archivo, rango_ventas, horizonte, usar_colaborado, col_colaborado):
    start_time = time.time()
    progress_bar = st.progress(0)
    status_text = st.empty()

    fechas_df = pd.read_excel(archivo, sheet_name="Base", header=None,
                              usecols=rango_ventas, skiprows=2, nrows=1)
    fechas_dt = pd.to_datetime(fechas_df.iloc[0].values)

    df_fijos = pd.read_excel(archivo, sheet_name="Base", header=None,
                             usecols='B:F', skiprows=3)
    df_fijos.columns = ['GERENCIA', 'GRUPO', 'ARTICULO_FAMILIA',
                        'COD_ARTICULO', 'DESCRIPCION']

    df_ventas = pd.read_excel(archivo, sheet_name="Base", header=None,
                              usecols=rango_ventas, skiprows=3)
    df_ventas = df_ventas.apply(pd.to_numeric, errors='coerce').fillna(0)

    if usar_colaborado:
        try:
            df_colab = pd.read_excel(archivo, sheet_name="Base", header=None,
                                     usecols=col_colaborado, skiprows=3)
            colaborado_series = df_colab.iloc[:, 0]
            colaborado_series = pd.to_numeric(colaborado_series, errors='coerce').fillna(0)
        except Exception as e:
            st.error(f"Error al leer columna {col_colaborado}: {e}")
            usar_colaborado = False
            colaborado_series = None
    else:
        colaborado_series = None

    hist_totales = df_ventas.sum(axis=0)
    hist_totales.index = fechas_dt

    total = len(df_fijos)
    resultados = []

    for i in range(total):
        elapsed = time.time() - start_time
        minutos = int(elapsed // 60)
        segundos = int(elapsed % 60)
        status_text.text(f"Procesando {i+1}/{total}: {df_fijos['DESCRIPCION'].iloc[i]} "
                         f"| Tiempo: {minutos}m {segundos}s")
        progress_bar.progress((i + 1) / total)

        serie_full = pd.Series(df_ventas.iloc[i].values, index=fechas_dt)
        serie_hasta_anteultimo = serie_full.iloc[:-1]
        real_ultimo = max(0, float(serie_full.iloc[-1]))

        if usar_colaborado and colaborado_series is not None:
            colab_val = max(0, float(colaborado_series.iloc[i]))
        else:
            colab_val = None

        # Prophet
        p_ultimo_prophet = fit_prophet(serie_hasta_anteultimo, 1)[0]
        # Chronos
        p_ultimo_chronos = fit_chronos(serie_hasta_anteultimo, 1)[0]

        mape_p = mean_absolute_percentage_error([real_ultimo + 1], [p_ultimo_prophet + 1])
        mape_c = mean_absolute_percentage_error([real_ultimo + 1], [p_ultimo_chronos + 1])

        if mape_p <= mape_c:
            mejor_modelo = "Prophet"
            mape_elegido = mape_p
            pron_ultimo = p_ultimo_prophet
            pronosticos = fit_prophet(serie_full, horizonte)
        else:
            mejor_modelo = "Chronos"
            mape_elegido = mape_c
            pron_ultimo = p_ultimo_chronos
            pronosticos = fit_chronos(serie_full, horizonte)

        pronosticos = pronosticos[:horizonte] + [0.0] * (horizonte - len(pronosticos))

        res = list(df_fijos.iloc[i]) + [
            real_ultimo, round(pron_ultimo, 2), round(mape_elegido * 100, 2), mejor_modelo
        ] + [round(p, 2) for p in pronosticos]
        
        if usar_colaborado and colab_val is not None:
            mape_colab = mean_absolute_percentage_error([real_ultimo + 1], [colab_val + 1]) * 100
            res.extend([colab_val, round(mape_colab, 2)])

        resultados.append(res)

    fechas_futuras = [fechas_dt[-1] + pd.DateOffset(months=i+1) for i in range(horizonte)]
    nombres_columnas_pron = [fecha.strftime('%m-%Y') for fecha in fechas_futuras]

    columnas_base = list(df_fijos.columns) + [
        'REAL_ULTIMO', 'PRON_ULTIMO', 'MAPE_%', 'MODELO_GANADOR'
    ] + nombres_columnas_pron
    
    if usar_colaborado:
        columnas_base.extend(['COLABORADO_ULTIMO', 'MAPE_COLABORADO_LINEA_%'])

    df_final = pd.DataFrame(resultados, columns=columnas_base)

    group_cols = ['COD_ARTICULO', 'GERENCIA']
    agg_dict = {'GRUPO': 'first', 'ARTICULO_FAMILIA': 'first', 'DESCRIPCION': 'first',
                'REAL_ULTIMO': 'sum', 'PRON_ULTIMO': 'sum'}
    for col in nombres_columnas_pron:
        agg_dict[col] = 'sum'
    if usar_colaborado:
        agg_dict['COLABORADO_ULTIMO'] = 'sum'

    df_agg = df_final.groupby(group_cols).agg(agg_dict).reset_index()

    def mape_agg(row):
        real = max(0, row['REAL_ULTIMO'])
        pron = max(0, row['PRON_ULTIMO'])
        return round(mean_absolute_percentage_error([real + 1], [pron + 1]) * 100, 2)

    df_agg['MAPE_%'] = df_agg.apply(mape_agg, axis=1)

    if usar_colaborado:
        def mape_colab_agg(row):
            real = max(0, row['REAL_ULTIMO'])
            colab = max(0, row['COLABORADO_ULTIMO'])
            return round(mean_absolute_percentage_error([real + 1], [colab + 1]) * 100, 2)
        df_agg['MAPE_COLABORADO_%'] = df_agg.apply(mape_colab_agg, axis=1)

    status_text.text("✅ Análisis finalizado.")
    progress_bar.progress(1.0)
    st.info(f"Tiempo: {(time.time() - start_time) / 60:.2f} minutos")

    return df_final, df_agg, fechas_dt, hist_totales, nombres_columnas_pron, usar_colaborado

# =====================================================
# 5. FUNCIONES DE VISUALIZACIÓN
# =====================================================

def mostrar_resultados(df_final, df_agg, usar_colaborado, horizonte, fechas_dt, 
                       hist_totales, nombres_columnas_pron):
    
    ultimo_mes = fechas_dt[-1]
    siguiente_mes = ultimo_mes + pd.DateOffset(months=1)
    nombre_ultimo = ultimo_mes.strftime('%b %Y')
    nombre_siguiente = siguiente_mes.strftime('%b %Y')

    # --- FILTROS CON BORDE (Gerencia, Familia, Producto) ---
    st.markdown('<div class="filtros-container">', unsafe_allow_html=True)
    st.markdown('<h4>🔍 Filtros de productos</h4>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        gerencias = sorted(df_agg['GERENCIA'].unique())
        gerencias_sel = st.multiselect("Gerencia", gerencias, default=gerencias)
    
    # Filtrar por gerencia seleccionada
    if gerencias_sel:
        df_filt = df_agg[df_agg['GERENCIA'].isin(gerencias_sel)]
    else:
        df_filt = df_agg
    
    with col_f2:
        familias = sorted(df_filt['ARTICULO_FAMILIA'].unique())
        familias_sel = st.multiselect("Familia", familias, default=familias)
    
    # Filtrar por familia seleccionada
    if familias_sel:
        df_filt = df_filt[df_filt['ARTICULO_FAMILIA'].isin(familias_sel)]
    
    with col_f3:
        productos = sorted(df_filt['DESCRIPCION'].unique())
        prod_sel = st.selectbox("Producto (búsqueda)", options=[""] + productos, index=0,
                                format_func=lambda x: "🔍 Buscar..." if x == "" else x)
    
    if prod_sel:
        df_filt = df_filt[df_filt['DESCRIPCION'] == prod_sel]
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- KPIs (con más espacio arriba) ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    total_real = int(df_agg['REAL_ULTIMO'].sum())
    total_pron = int(df_agg['PRON_ULTIMO'].sum())
    primer_mes_futuro = nombres_columnas_pron[0] if nombres_columnas_pron else "M1"
    total_pron_marzo = int(df_agg[primer_mes_futuro].sum()) if primer_mes_futuro in df_agg.columns else 0
    mape_promedio = round(df_filt['MAPE_%'].mean(), 1) if len(df_filt) > 0 else 0.0

    st.subheader("📊 Resultado Global")
    
    if usar_colaborado:
        total_colab = int(df_agg['COLABORADO_ULTIMO'].sum())
        mape_colab_promedio = round(df_filt['MAPE_COLABORADO_%'].mean(), 1) if len(df_filt) > 0 else 0.0
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric(f"Real {nombre_ultimo}", f"{total_real:,.0f}".replace(',', '.'))
        c2.metric(f"Pronóstico {nombre_ultimo}", f"{total_pron:,.0f}".replace(',', '.'))
        c3.metric(f"Colaborado {nombre_ultimo}", f"{total_colab:,.0f}".replace(',', '.'))
        c4.metric(f"Pronóstico {nombre_siguiente}", f"{total_pron_marzo:,.0f}".replace(',', '.'))
        c5.metric("MAPE pronóstico", f"{mape_promedio:.1f}%")
        c6.metric("MAPE colaborado", f"{mape_colab_promedio:.1f}%")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Real {nombre_ultimo}", f"{total_real:,.0f}".replace(',', '.'))
        c2.metric(f"Pronóstico {nombre_ultimo}", f"{total_pron:,.0f}".replace(',', '.'))
        c3.metric(f"Pronóstico {nombre_siguiente}", f"{total_pron_marzo:,.0f}".replace(',', '.'))
        c4.metric("MAPE pronóstico", f"{mape_promedio:.1f}%")

    # --- GRÁFICO con leyenda arriba ---
    fecha_ultimo_real = fechas_dt[-1]
    fechas_futuras = pd.date_range(start=fecha_ultimo_real + pd.DateOffset(months=1), periods=horizonte, freq='MS')

    fig = go.Figure()
    if hist_totales is not None and len(hist_totales) == len(fechas_dt):
        fig.add_trace(go.Scatter(x=fechas_dt, y=hist_totales,
                                 mode='lines', name='Venta Real',
                                 line=dict(color='#1f77b4', width=2)))

    proyeccion = []
    for col in nombres_columnas_pron[:horizonte]:
        proyeccion.append(df_filt[col].sum() if col in df_filt.columns else 0)

    fig.add_trace(go.Scatter(x=fechas_futuras, y=proyeccion,
                             mode='lines+markers', name='Proyección',
                             line=dict(color='#00CC96', width=2, dash='dash'),
                             marker=dict(size=6)))

    if hist_totales is not None and len(hist_totales) > 0 and proyeccion:
        ultimo_real = hist_totales.iloc[-1] if isinstance(hist_totales, pd.Series) else hist_totales[-1]
        fig.add_trace(go.Scatter(x=[fechas_dt[-1], fechas_futuras[0]],
                                 y=[ultimo_real, proyeccion[0]],
                                 mode='lines', line=dict(color='#00CC96', width=1, dash='dot'),
                                 showlegend=False))

    # Leyenda arriba en el medio
    fig.update_layout(
        title="Histórico de ventas y proyección",
        title_x=0.5,
        xaxis_title="Fecha",
        yaxis_title="Ventas (unidades)",
        hovermode="x unified",
        xaxis=dict(tickformat="%d/%m/%Y", tickangle=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLA DE PRODUCTOS con filtro REAL en el título ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_titulo1, col_titulo2 = st.columns([3, 2])
    with col_titulo1:
        st.subheader("📋 Detalle por producto (agregado)")
    with col_titulo2:
        col_min, col_max = st.columns(2)
        with col_min:
            filtro_min = st.number_input("REAL mín.", value=0, step=1, key="filtro_min_tabla")
        with col_max:
            filtro_max = st.number_input("REAL máx.", value=100000, step=1000, key="filtro_max_tabla")
    
    # Aplicar filtro de REAL a la tabla
    df_filt = df_filt[(df_filt['REAL_ULTIMO'] >= filtro_min) & (df_filt['REAL_ULTIMO'] <= filtro_max)]

    columnas_fijas = ['COD_ARTICULO', 'DESCRIPCION', 'ARTICULO_FAMILIA', 'GERENCIA',
                      'REAL_ULTIMO', 'PRON_ULTIMO', 'MAPE_%']
    columnas_pron = nombres_columnas_pron[:min(horizonte, 6)]
    if usar_colaborado:
        columnas_fijas.extend(['COLABORADO_ULTIMO', 'MAPE_COLABORADO_%'])
    columnas_tabla = columnas_fijas + columnas_pron
    st.dataframe(df_filt[columnas_tabla], use_container_width=True)

    # --- DESCARGA EXCEL ---
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_final.to_excel(writer, sheet_name='Pronósticos (líneas)', index=False)
        cols_export = list(df_agg.columns)
        for col in ['COD_ARTICULO', 'DESCRIPCION']:
            if col in cols_export:
                cols_export.remove(col)
                cols_export.insert(0, col)
        cols_export = [c for c in cols_export if c != 'GRUPO']
        df_agg[cols_export].to_excel(writer, sheet_name='Pronósticos (productos)', index=False)
    output.seek(0)
    st.download_button(label="📥 Descargar pronósticos (Excel)", data=output,
                       file_name="pronosticos_final_limpio.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =====================================================
# 6. INTERFAZ PRINCIPAL
# =====================================================

if 'autenticado' not in st.session_state:
    st.session_state.autenticado = False
if 'proyecto_actual' not in st.session_state:
    st.session_state.proyecto_actual = None

if not st.session_state.autenticado:
    st.title("🔐 Acceso al Sistema de Pronósticos")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            usuario = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Iniciar sesión"):
                if verificar_login(usuario, password):
                    st.session_state.autenticado = True
                    st.session_state.usuario = usuario
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")
    st.stop()

st.sidebar.title(f"👤 {st.session_state.usuario}")
if st.sidebar.button("🚪 Cerrar sesión"):
    st.session_state.autenticado = False
    st.session_state.proyecto_actual = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Gestión de Proyectos")

proyectos = listar_proyectos()
if proyectos:
    st.sidebar.markdown("### Proyectos guardados")
    for p in proyectos:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button(f"📂 {p['nombre']}", key=f"load_{p['nombre']}"):
                proyecto = cargar_proyecto(p['nombre'])
                if proyecto:
                    st.session_state.proyecto_actual = proyecto
                    st.session_state.proyecto_nombre = p['nombre']
                    st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{p['nombre']}"):
                eliminar_proyecto(p['nombre'])
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Crear nuevo proyecto")

with st.sidebar.form("nuevo_proyecto_form"):
    nombre_nuevo = st.text_input("Nombre del proyecto (ej: Febrero 2026)")
    archivo_subido = st.file_uploader("Subir archivo Excel", type=["xlsx", "xls"], key="nuevo_proyecto")
    rango_ventas_nuevo = st.text_input("Rango de columnas de ventas", value="I:BF")
    horizonte_nuevo = st.slider("Horizonte de pronóstico (meses)", 1, 12, 12)
    usar_colaborado_nuevo = st.checkbox("Incluir plan colaborado", value=False)
    col_colaborado_nuevo = None
    if usar_colaborado_nuevo:
        col_colaborado_nuevo = st.text_input("Columna del colaborado", value="CC").strip().upper()
    
    if st.form_submit_button("🚀 Crear y procesar proyecto"):
        if not nombre_nuevo:
            st.error("Ingrese un nombre")
        elif not archivo_subido:
            st.error("Suba un archivo Excel")
        else:
            with st.spinner("Procesando... esto puede tomar varios minutos"):
                try:
                    df_final, df_agg, fechas_dt, hist_totales, nombres_cols, uc = procesar_archivo(
                        archivo_subido, rango_ventas_nuevo, horizonte_nuevo,
                        usar_colaborado_nuevo, col_colaborado_nuevo)
                    guardar_proyecto(nombre_nuevo, df_final, df_agg, fechas_dt, uc,
                                    horizonte_nuevo, nombres_cols, rango_ventas_nuevo, hist_totales)
                    st.success(f"Proyecto '{nombre_nuevo}' creado exitosamente")
                    proyecto = cargar_proyecto(nombre_nuevo)
                    if proyecto:
                        st.session_state.proyecto_actual = proyecto
                        st.session_state.proyecto_nombre = nombre_nuevo
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

if st.session_state.proyecto_actual:
    proyecto = st.session_state.proyecto_actual
    st.title(f"📊 Pronóstico - {st.session_state.proyecto_nombre}")
    st.caption(f"Proyecto guardado localmente")
    if st.button("◀️ Volver a proyectos"):
        st.session_state.proyecto_actual = None
        st.rerun()
    mostrar_resultados(proyecto['df_final'], proyecto['df_agg'], proyecto['usar_colaborado'],
                      proyecto['horizonte'], proyecto['fechas_dt'], proyecto['hist_totales'],
                      proyecto['nombres_columnas_pron'])
else:
    st.title("🏛️ Sistema de Pronóstico de Demanda")
    st.markdown("""
    ### Bienvenido al sistema
    
    Para comenzar:
    1. En el panel izquierdo, crea un nuevo proyecto o carga uno existente
    2. Sube un archivo Excel con el formato estándar
    3. Configura los parámetros de pronóstico
    4. Espera a que se complete el análisis
    5. Explora los resultados, aplica filtros y descarga los datos
    
    Los proyectos se guardan automáticamente y puedes retomarlos en cualquier momento.
    """)
