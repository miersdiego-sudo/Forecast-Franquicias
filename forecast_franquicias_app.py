import streamlit as st
import pandas as pd
import torch
import plotly.graph_objects as go
from prophet import Prophet
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error
import time
from io import BytesIO

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="IA Executive Amandau", layout="wide")

@st.cache_resource
def cargar_modelos():
    return ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", dtype=torch.float32)

pipeline = cargar_modelos()

# --- 2. LÓGICA DE COMPETENCIA DE MODELOS ---
def obtener_mejor_pronostico(serie_full):
    serie_ent = serie_full.iloc[:-1]
    real_feb = serie_full.iloc[-1]
    
    # PROPHET con estacionalidad anual forzada y 300 iteraciones
    try:
        df_p = pd.DataFrame({'ds': serie_ent.index, 'y': serie_ent.values})
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0
        )
        m.fit(df_p, iter=300)
        p_feb_p = max(0, float(m.predict(m.make_future_dataframe(periods=1, freq='MS'))['yhat'].iloc[-1]))
    except:
        p_feb_p = float(serie_ent.iloc[-1])

    # CHRONOS
    try:
        ctx = torch.tensor(serie_ent.values, dtype=torch.float32)
        f_c = pipeline.predict(ctx, 1).mean(dim=0).numpy()
        p_feb_c = max(0, float(f_c[0, 0]))
    except:
        p_feb_c = float(serie_ent.iloc[-1])

    # SELECCIÓN
    mape_p = mean_absolute_percentage_error([real_feb + 1], [p_feb_p + 1])
    mape_c = mean_absolute_percentage_error([real_feb + 1], [p_feb_c + 1])
    
    if mape_p <= mape_c:
        df_f = pd.DataFrame({'ds': serie_full.index, 'y': serie_full.values})
        m_f = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0
        )
        m_f.fit(df_f, iter=300)
        p_mar = max(0, float(m_f.predict(m_f.make_future_dataframe(periods=1, freq='MS'))['yhat'].iloc[-1]))
        mejor_mape_item = mape_p
    else:
        ctx_m = torch.tensor(serie_full.values, dtype=torch.float32)
        f_m = pipeline.predict(ctx_m, 1).mean(dim=0).numpy()
        p_mar = max(0, float(f_m[0, 0]))
        mejor_mape_item = mape_c
        
    return float(p_feb_p), float(p_mar), float(mejor_mape_item * 100)

# --- 3. INTERFAZ ---
st.title("🏛️ Sistema de Demanda Amandau - Precisión Optimizada")
archivo = st.file_uploader("Subir Plan de Ventas (Excel)", type=["xlsx"])

if archivo:
    if st.button("🚀 Ejecutar Análisis IA"):
        start_time = time.time()

        fechas_df = pd.read_excel(archivo, sheet_name="Base", header=None, usecols='I:BF', skiprows=2, nrows=1)
        fechas_dt = pd.to_datetime(fechas_df.iloc[0].values)
        df_fijos = pd.read_excel(archivo, sheet_name="Base", header=None, usecols='B:F', skiprows=3)
        df_fijos.columns = ['GERENCIA', 'GRUPO', 'ARTICULO_FAMILIA', 'COD_ARTICULO', 'DESCRIPCION']
        df_v = pd.read_excel(archivo, sheet_name="Base", header=None, usecols='I:BF', skiprows=3).apply(pd.to_numeric, errors='coerce').fillna(0)
        df_v.columns = fechas_dt

        total = len(df_fijos)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        resultados = []
        for i in range(total):
            elapsed = time.time() - start_time
            minutos = int(elapsed // 60)
            segundos = int(elapsed % 60)
            status_text.text(f"Procesando: {df_fijos['DESCRIPCION'].iloc[i]} | Tiempo: {minutos}m {segundos}s")
            
            resultados.append(obtener_mejor_pronostico(df_v.iloc[i]))
            progress_bar.progress((i + 1) / total)
        
        status_text.text("✅ Análisis finalizado.")
        
        df_m = df_fijos.copy()
        df_m['PRON_FEB'], df_m['PRON_MAR'], df_m['MAPE_ITEM'] = zip(*resultados)
        df_m['PRON_FEB'] = df_m['PRON_FEB'].astype(float)
        df_m['PRON_MAR'] = df_m['PRON_MAR'].astype(float)
        df_m['MAPE_ITEM'] = df_m['MAPE_ITEM'].astype(float)
        df_m['REAL_FEB'] = df_v.iloc[:, -1].values.astype(float)
        df_m['REAL_MAR_25'] = (df_v.iloc[:, -13].values if len(df_v.columns) > 12 else 0).astype(float)
        
        st.session_state['master'] = df_m
        st.session_state['hist'] = df_v.sum()
        st.session_state['fechas'] = fechas_dt

if 'master' in st.session_state:
    df_m = st.session_state['master']
    fechas = st.session_state['fechas']

    # --- NOMBRES DINÁMICOS DE MESES ---
    ultimo_mes_real = fechas[-1]
    siguiente_mes = ultimo_mes_real + pd.DateOffset(months=1)
    nombre_ultimo = ultimo_mes_real.strftime('%b %Y')
    nombre_siguiente = siguiente_mes.strftime('%b %Y')

    # --- FILTROS EN SIDEBAR ---
    st.sidebar.header("⚙️ Ajustes")

    # 1. Filtro por venta total mínima por artículo (sumando canales)
    min_venta = st.sidebar.number_input("Filtrar artículos con venta total < (unidades):", value=0, step=1)
    
    # Calcular total real por artículo (usando COD_ARTICULO)
    df_articulo_total = df_m.groupby('COD_ARTICULO')['REAL_FEB'].sum().reset_index()
    df_articulo_total.columns = ['COD_ARTICULO', 'TOTAL_REAL']
    df_m = df_m.merge(df_articulo_total, on='COD_ARTICULO', how='left')
    df_f = df_m[df_m['TOTAL_REAL'] >= min_venta].copy()
    df_f.drop(columns=['TOTAL_REAL'], inplace=True)

    # 2. Filtro por familia (multiselect)
    familias = sorted(df_f['ARTICULO_FAMILIA'].unique())
    familias_seleccionadas = st.sidebar.multiselect(
        "Seleccionar familia(s):",
        options=familias,
        default=familias
    )
    if familias_seleccionadas:
        df_f = df_f[df_f['ARTICULO_FAMILIA'].isin(familias_seleccionadas)]

    # 3. Filtro por descripción (búsqueda por texto)
    desc_search = st.sidebar.text_input("Buscar por descripción (contiene):", value="")
    if desc_search:
        df_f = df_f[df_f['DESCRIPCION'].str.contains(desc_search, case=False, na=False)]

    # --- CÁLCULO DE MÉTRICAS (sobre el DataFrame filtrado) ---
    suma_real_feb = float(df_f['REAL_FEB'].sum())
    suma_pron_feb = float(df_f['PRON_FEB'].sum())
    mape_promedio = df_f['MAPE_ITEM'].mean() if len(df_f) > 0 else 0.0

    # --- KPIs (5 columnas) ---
    st.subheader("📊 Resumen de Desempeño")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    v_mar26 = float(df_f['PRON_MAR'].sum())
    v_feb26 = suma_real_feb
    v_mar25 = float(df_f['REAL_MAR_25'].sum())

    c1.metric(f"Venta Real {nombre_ultimo}", f"{v_feb26:,.0f}",
              f"{((v_mar26/v_feb26)-1)*100:.1f}%" if v_feb26 > 0 else "0%")
    c2.metric(f"Pronóstico {nombre_ultimo}", f"{suma_pron_feb:,.0f}")
    c3.metric(f"Venta Real Mar 25", f"{v_mar25:,.0f}",
              f"{((v_mar26/v_mar25)-1)*100:.1f}%" if v_mar25 > 0 else "0%")
    c4.metric(f"Pronóstico {nombre_siguiente}", f"{v_mar26:,.0f}")
    c5.metric("Precisión IA (MAPE promedio)", f"{mape_promedio:.1f}%",
              delta="Excelente" if mape_promedio < 15 else "Revisar", delta_color="inverse")

    # --- GRÁFICO ---
    st.divider()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fechas, y=st.session_state['hist'], name="Venta Real (total)", line=dict(color='#636EFA', width=2)))
    fig.add_trace(go.Scatter(x=[ultimo_mes_real, siguiente_mes], y=[v_feb26, v_mar26], 
                             name="Proyección (filtrada)", line=dict(color='#00CC96', width=4, dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

    # --- TABLA DETALLADA ---
    st.subheader("📋 Detalle por Artículo (filtrado)")
    columnas_mostrar = ['DESCRIPCION', 'ARTICULO_FAMILIA', 'REAL_FEB', 'PRON_FEB', 'PRON_MAR', 'MAPE_ITEM']
    st.dataframe(df_f[columnas_mostrar], use_container_width=True)

    # --- BOTÓN DESCARGA DETALLE EXCEL ---
    df_export = df_f[columnas_mostrar].copy()
    for col in ['REAL_FEB', 'PRON_FEB', 'PRON_MAR', 'MAPE_ITEM']:
        df_export[col] = df_export[col].astype(float)
    
    output_detalle = BytesIO()
    with pd.ExcelWriter(output_detalle, engine='openpyxl') as writer:
        df_export.to_excel(writer, sheet_name='Detalle', index=False)
    output_detalle.seek(0)
    
    st.download_button(
        label="📥 Descargar detalle en Excel",
        data=output_detalle,
        file_name="detalle_pronosticos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --- TABLA RESUMEN POR FAMILIA ---
    st.subheader("📊 Resumen por Familia")
    df_familia_sum = df_f.groupby('ARTICULO_FAMILIA').agg({
        'REAL_FEB': 'sum',
        'PRON_FEB': 'sum',
        'PRON_MAR': 'sum',
        'DESCRIPCION': 'count',
        'MAPE_ITEM': 'mean'
    }).rename(columns={'DESCRIPCION': 'CANT_ARTICULOS', 'MAPE_ITEM': 'MAPE_FAMILIA'}).reset_index()
    
    columnas_familia = ['ARTICULO_FAMILIA', 'REAL_FEB', 'PRON_FEB', 'PRON_MAR', 'CANT_ARTICULOS', 'MAPE_FAMILIA']
    st.dataframe(df_familia_sum[columnas_familia], use_container_width=True)
    
    output_fam = BytesIO()
    with pd.ExcelWriter(output_fam, engine='openpyxl') as writer:
        df_familia_sum[columnas_familia].to_excel(writer, sheet_name='Resumen_Familia', index=False)
    output_fam.seek(0)
    
    st.download_button(
        label="📥 Descargar resumen por familia en Excel",
        data=output_fam,
        file_name="resumen_familias.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )