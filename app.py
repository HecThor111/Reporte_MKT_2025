import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
CSV_FILE = "hubspot_marketing_y_posteriores_2025_clean.csv"

st.set_page_config(
    page_title="Dashboard HubSpot – Marketing → Comerciales",
    layout="wide",
)

# -------------------------
# THEME / ESTILO FUTURISTA
# -------------------------
SPACE_BG = "#050816"
SPACE_PANEL = "#0b1020"
SPACE_PANEL_SOFT = "#111827"
NEON_CYAN = "#22d3ee"
NEON_BLUE = "#38bdf8"
NEON_PURPLE = "#a855f7"
NEON_GREEN = "#22c55e"
NEON_PINK = "#ec4899"
TEXT_MAIN = "#e5e7eb"
TEXT_MUTED = "#9ca3af"

SPACE_PALETTE = [NEON_CYAN, NEON_PURPLE, NEON_GREEN, NEON_BLUE, NEON_PINK, "#eab308"]

st.markdown(
    f"""
    <style>
    /* Fondo general */
    [data-testid="stAppViewContainer"] {{
        background: radial-gradient(circle at top left, rgba(56,189,248,0.12), transparent 55%),
                    radial-gradient(circle at bottom right, rgba(168,85,247,0.18), transparent 55%),
                    {SPACE_BG};
        color: {TEXT_MAIN};
    }}

    [data-testid="stHeader"] {{
        background: linear-gradient(90deg, rgba(15,23,42,0.95), rgba(15,23,42,0.75));
        border-bottom: 1px solid rgba(148,163,184,0.15);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(180deg, #020617, #020617);
        border-right: 1px solid rgba(148,163,184,0.25);
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_MAIN} !important;
        letter-spacing: 0.03em;
    }}

    .metric-label {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.15rem;
    }}

    .metric-value {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {NEON_CYAN};
    }}

    .metric-card {{
        background: radial-gradient(circle at top left, rgba(34,211,238,0.07), transparent 55%),
                    {SPACE_PANEL};
        border-radius: 1rem;
        padding: 0.9rem 1rem;
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 18px 45px rgba(15,23,42,0.75);
    }}

    .section-divider {{
        border-top: 1px solid rgba(148,163,184,0.25);
        margin: 2rem 0 1.2rem 0;
    }}

    /* Tablas */
    .stDataFrame {{ color: {TEXT_MAIN}; }}

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.2rem;">
        <span style="font-size:1.9rem;">📡</span>
        <div>
            <div style="font-size:2rem;font-weight:800;
                        background:linear-gradient(90deg,{NEON_CYAN},{NEON_PURPLE});
                        -webkit-background-clip:text;color:transparent;">
                Reporte de Leads de Marketing 2025
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Origen de datos: hubspot_marketing_y_posteriores_2025_clean.csv")

# -------------------------
# CARGA DE DATOS
# -------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # Parseamos fechas desde el CSV limpio
    df = pd.read_csv(
        path,
        parse_dates=["origen_created_date", "deal_created_date"],
    )

    # company_id como string (para evitar notación científica rara)
    if "company_id" in df.columns:
        df["company_id"] = df["company_id"].astype(str)

    # Asegurar montos numéricos
    for col in ["origen_amount", "deal_amount", "monto_origen", "monto_posterior"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Normalizar texto principal
    text_cols = [
        "tipo_negocio",
        "origen_pipeline_name",
        "deal_pipeline_name",
        "pipeline_marketing",
        "pipeline_comercial",
        "origen_dealstage",
        "deal_dealstage",
        "etapa_marketing",
        "etapa_comercial",
        "estado_marketing",
        "estado_comercial",
        "origen_owner_name",
        "deal_owner_name",
        "company_name",
        "company_domain",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Por seguridad, rellenamos pipelines bonitos si vienen nulos
    if "pipeline_marketing" in df.columns and "origen_pipeline_name" in df.columns:
        df["pipeline_marketing"] = df["pipeline_marketing"].replace(
            {"": np.nan}
        ).fillna(df["origen_pipeline_name"])

    if "pipeline_comercial" in df.columns and "deal_pipeline_name" in df.columns:
        df["pipeline_comercial"] = df["pipeline_comercial"].replace(
            {"": np.nan}
        ).fillna(df["deal_pipeline_name"])

    return df


df = load_data(CSV_FILE)

if df.empty:
    st.error("El CSV está vacío o no se pudo cargar.")
    st.stop()

# -------------------------
# SEPARAR ORIGEN Y POSTERIORES
# -------------------------
df_origen = df[df["tipo_negocio"] == "origen_marketing"].copy()
df_post = df[df["tipo_negocio"] == "posterior_empresa"].copy()

# Versión deduplicada por negocio (para métricas más limpias)
df_origen_unique = (
    df_origen.sort_values("origen_created_date")
    .drop_duplicates(subset=["origen_deal_id"])
    .copy()
)
df_post_unique = (
    df_post.sort_values("deal_created_date")
    .drop_duplicates(subset=["deal_id"])
    .copy()
)

# -------------------------
# SIDEBAR – FILTROS
# -------------------------
st.sidebar.header("🔍 Filtros")

# Rango de fechas del negocio origen (usando negocios únicos)
min_date = df_origen_unique["origen_created_date"].min()
max_date = df_origen_unique["origen_created_date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.error("No hay fechas válidas en 'origen_created_date'.")
    st.stop()

date_range = st.sidebar.date_input(
    "Rango de fecha (creación del negocio origen)",
    value=(min_date.date(), max_date.date()),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date.date(), max_date.date()

# Pipeline origen (usando pipeline_marketing “bonito”)
pipelines_origen = sorted(
    df_origen_unique["pipeline_marketing"].replace({"": np.nan}).dropna().unique()
)
pipeline_filter = st.sidebar.multiselect(
    "Pipeline de marketing (origen)",
    options=pipelines_origen,
    default=pipelines_origen,
)

# Owner origen
owners_origen = sorted(
    df_origen_unique["origen_owner_name"].replace({"": np.nan}).dropna().unique()
)
owner_filter = st.sidebar.multiselect(
    "Owner del negocio origen",
    options=owners_origen,
    default=owners_origen,
)

# Aplicar filtros sobre los negocios de origen (unique)
mask_origen = (
    (df_origen_unique["origen_created_date"].dt.date >= start_date)
    & (df_origen_unique["origen_created_date"].dt.date <= end_date)
    & (df_origen_unique["pipeline_marketing"].isin(pipeline_filter))
    & (df_origen_unique["origen_owner_name"].isin(owner_filter))
)

df_origen_f = df_origen_unique[mask_origen].copy()

# Filtrar también los posteriores en función de los origen filtrados
origen_ids_filtrados = df_origen_f["origen_deal_id"].astype(str).unique()
df_post_f = df_post[df_post["origen_deal_id"].astype(str).isin(origen_ids_filtrados)].copy()
df_post_f_unique = (
    df_post_f.sort_values("deal_created_date")
    .drop_duplicates(subset=["deal_id"])
    .copy()
)

# -------------------------
# MÉTRICAS GENERALES (fila 1)
# -------------------------
st.markdown("### 🔢 Métricas generales")

col1, col2, col3, col4 = st.columns(4)

num_origen = df_origen_f["origen_deal_id"].nunique()
num_post_unicos = df_post_f_unique["deal_id"].nunique()

# Monto origen: sumando una vez por negocio origen
total_origen_amount = df_origen_f["origen_amount"].sum()

# Monto posterior: sumando una vez por negocio posterior
total_post_amount = df_post_f_unique["deal_amount"].sum()

deals_post_por_origen = num_post_unicos / num_origen if num_origen > 0 else 0

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Negocios de marketing (origen)</div>
            <div class="metric-value">{num_origen:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Negocios posteriores únicos</div>
            <div class="metric-value">{num_post_unicos:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Monto total posterior (único)</div>
            <div class="metric-value">${total_post_amount:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Deals posteriores por negocio origen</div>
            <div class="metric-value">{deals_post_por_origen:.2f}x</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# MÉTRICAS AVANZADAS (fila 2)
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 🧠 Métricas avanzadas")

col5, col6, col7, col8 = st.columns(4)

# Conversión: origen con al menos 1 posterior
agg_post = (
    df_post_f.groupby("origen_deal_id")["deal_id"]
    .nunique()
    .reset_index(name="posterior_deals")
)
num_origen_con_post = agg_post[agg_post["posterior_deals"] > 0]["origen_deal_id"].nunique()
conversion_rate = (num_origen_con_post / num_origen * 100) if num_origen > 0 else 0

# ROI / factor de multiplicación
roi_factor = (total_post_amount / total_origen_amount) if total_origen_amount > 0 else 0

# Estados comerciales (sobre deals únicos posteriores)
estado_counts = (
    df_post_f_unique["estado_comercial"]
    .value_counts(dropna=False)
    .rename_axis("estado_comercial")
    .reset_index(name="num_deals")
)
ganados = estado_counts.loc[
    estado_counts["estado_comercial"] == "Ganado", "num_deals"
].sum()
win_rate = (ganados / num_post_unicos * 100) if num_post_unicos > 0 else 0

# Tiempo medio marketing → primer negocio posterior
primer_posterior = (
    df_post_f.groupby("origen_deal_id")["deal_created_date"]
    .min()
    .reset_index(name="fecha_primer_posterior")
)
tmp_time = df_origen_f.merge(primer_posterior, on="origen_deal_id", how="inner")
tmp_time["dias_a_primer_posterior"] = (
    tmp_time["fecha_primer_posterior"] - tmp_time["origen_created_date"]
).dt.days
dias_prom = tmp_time["dias_a_primer_posterior"].mean() if not tmp_time.empty else np.nan

for col, label, value in [
    (col5, "Origen con ≥1 negocio posterior", f"{num_origen_con_post:,}"),
    (col6, "Tasa de conversión marketing → posterior", f"{conversion_rate:.1f}%"),
    (col7, "Factor de multiplicación (posterior / origen)", f"{roi_factor:.2f}x"),
    (
        col8,
        "Días promedio a primer posterior",
        f"{dias_prom:.1f} días" if not np.isnan(dias_prom) else "N/A",
    ),
]:
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# DISTRIBUCIÓN DE ESTADOS
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 🧩 Distribución de estados comerciales y de marketing")
col_est1, col_est2 = st.columns(2)

with col_est1:
    st.markdown("**Distribución de negocios posteriores por estado comercial**")
    if not df_post_f_unique.empty:
        estado_counts_amt = (
            df_post_f_unique.groupby("estado_comercial")["deal_amount"]
            .sum()
            .reset_index()
        )
        fig_estado = px.pie(
            estado_counts_amt,
            names="estado_comercial",
            values="deal_amount",
            hole=0.45,
            color_discrete_sequence=SPACE_PALETTE,
        )
        fig_estado.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
        )
        st.plotly_chart(fig_estado, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

with col_est2:
    st.markdown("**Estados de marketing por pipeline**")
    if not df_origen_f.empty:
        mkt_estado = (
            df_origen_f.groupby(["pipeline_marketing", "estado_marketing"])["origen_deal_id"]
            .nunique()
            .reset_index(name="num_deals")
        )
        fig_mkt = px.bar(
            mkt_estado,
            x="pipeline_marketing",
            y="num_deals",
            color="estado_marketing",
            barmode="stack",
            color_discrete_sequence=SPACE_PALETTE,
        )
        fig_mkt.update_layout(
            template="plotly_dark",
            xaxis_title="Pipeline de marketing",
            yaxis_title="Núm. negocios",
            margin=dict(l=10, r=10, t=30, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
        )
        st.plotly_chart(fig_mkt, use_container_width=True)
    else:
        st.info("No hay negocios de origen con los filtros actuales.")

# -------------------------
# EVOLUCIÓN TEMPORAL
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("### 📆 Evolución temporal")

col_time1, col_time2 = st.columns(2)

# 1) Negocios de marketing por mes
with col_time1:
    st.markdown("**Negocios de marketing por mes (cantidad y monto)**")
    if not df_origen_f.empty:
        tmp_o = df_origen_f.copy()
        tmp_o["mes"] = tmp_o["origen_created_date"].dt.to_period("M").dt.to_timestamp()
        evo_o = (
            tmp_o.groupby("mes")
            .agg(
                num_negocios=("origen_deal_id", "nunique"),
                monto_origen=("origen_amount", "sum"),
            )
            .reset_index()
        )
        fig_evo = px.bar(
            evo_o,
            x="mes",
            y="num_negocios",
            hover_data=["monto_origen"],
            color_discrete_sequence=[NEON_CYAN],
        )
        fig_evo.update_layout(
            template="plotly_dark",
            xaxis_title="Mes",
            yaxis_title="Negocios de marketing",
            margin=dict(l=10, r=10, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
        )
        st.plotly_chart(fig_evo, use_container_width=True)
    else:
        st.info("No hay negocios de marketing con los filtros actuales.")

# 2) Negocios posteriores por mes
with col_time2:
    st.markdown("**Negocios posteriores por mes (cantidad y monto)**")
    if not df_post_f_unique.empty:
        tmp_p = df_post_f_unique.copy()
        tmp_p["mes"] = tmp_p["deal_created_date"].dt.to_period("M").dt.to_timestamp()
        evo_p = (
            tmp_p.groupby("mes")
            .agg(
                num_negocios=("deal_id", "nunique"),
                monto_posterior=("deal_amount", "sum"),
            )
            .reset_index()
        )
        fig_evo2 = px.bar(
            evo_p,
            x="mes",
            y="num_negocios",
            hover_data=["monto_posterior"],
            color_discrete_sequence=[NEON_PURPLE],
        )
        fig_evo2.update_layout(
            template="plotly_dark",
            xaxis_title="Mes",
            yaxis_title="Negocios posteriores",
            margin=dict(l=10, r=10, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
        )
        st.plotly_chart(fig_evo2, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")


# -------------------------
# TABLA RESUMEN POR NEGOCIO ORIGEN
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📌 Resumen por negocio de marketing")

# Info base del negocio origen (1 fila por origen_deal_id)
base_origen = df_origen_f.copy()

base_origen = (
    base_origen.sort_values("origen_created_date")
    .drop_duplicates(subset=["origen_deal_id"])
    .loc[
        :,
        [
            "origen_deal_id",
            "origen_deal_name",
            "origen_created_date",
            "pipeline_marketing",
            "etapa_marketing",
            "estado_marketing",
            "origen_amount",
            "origen_owner_name",
            "company_id",
            "company_name",
        ],
    ]
)

# Agregados de posteriores por negocio origen (con df_post_f, pero agregando por origen_deal_id)
agg_post_count = (
    df_post_f.groupby("origen_deal_id")["deal_id"]
    .nunique()
    .reset_index(name="posterior_deals")
)

agg_post_monto = (
    df_post_f.groupby("origen_deal_id")["deal_amount"]
    .agg(posterior_monto_total="sum", posterior_monto_promedio="mean")
    .reset_index()
)

resumen = base_origen.merge(agg_post_count, on="origen_deal_id", how="left")
resumen = resumen.merge(agg_post_monto, on="origen_deal_id", how="left")

resumen["posterior_deals"] = resumen["posterior_deals"].fillna(0).astype(int)
resumen["posterior_monto_total"] = resumen["posterior_monto_total"].fillna(0.0)
resumen["posterior_monto_promedio"] = resumen["posterior_monto_promedio"].fillna(0.0)

st.dataframe(
    resumen.sort_values("posterior_monto_total", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# -------------------------
# SECCIÓN INSIGHTS VISUALES (barras)
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📈 Insights visuales")

col_g1, col_g2 = st.columns(2)

# 1) Monto posterior por owner (negocios posteriores, deals únicos)
with col_g1:
    st.markdown("**Monto total posterior por owner comercial (deals únicos)**")
    if not df_post_f_unique.empty:
        monto_por_owner = (
            df_post_f_unique.groupby("deal_owner_name")["deal_amount"]
            .sum()
            .reset_index()
            .sort_values("deal_amount", ascending=False)
        )
        fig_owner = px.bar(
            monto_por_owner,
            x="deal_owner_name",
            y="deal_amount",
            color_discrete_sequence=[NEON_GREEN],
        )
        fig_owner.update_layout(
            template="plotly_dark",
            xaxis_title="Owner comercial",
            yaxis_title="Monto posterior",
            margin=dict(l=10, r=10, t=30, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
        )
        st.plotly_chart(fig_owner, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

# 2) Cantidad de deals posteriores por pipeline comercial
with col_g2:
    st.markdown("**Cantidad de negocios posteriores por pipeline comercial (deals únicos)**")
    if not df_post_f_unique.empty:
        deals_por_pipeline = (
            df_post_f_unique.groupby("pipeline_comercial")["deal_id"]
            .nunique()
            .reset_index()
            .sort_values("deal_id", ascending=False)
        )
        deals_por_pipeline.rename(columns={"deal_id": "num_deals"}, inplace=True)

        fig_pipe = px.bar(
            deals_por_pipeline,
            x="pipeline_comercial",
            y="num_deals",
            color_discrete_sequence=[NEON_BLUE],
        )
        fig_pipe.update_layout(
            template="plotly_dark",
            xaxis_title="Pipeline comercial",
            yaxis_title="Núm. negocios",
            margin=dict(l=10, r=10, t=30, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color=TEXT_MAIN,
        )
        st.plotly_chart(fig_pipe, use_container_width=True)
    else:
        st.info("No hay negocios posteriores con los filtros actuales.")

# -------------------------
# SANKEY: FLUJO MARKETING → PIPELINES COMERCIALES
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("🔀 Flujo de negocio: Marketing → Pipelines comerciales (Sankey)")

if df_post_f.empty:
    st.info("No hay negocios posteriores para construir el diagrama de flujo con los filtros actuales.")
else:
    st.markdown("Ajustes del diagrama:")
    col_s1, _ = st.columns(2)
    with col_s1:
        metrica_flujo = st.radio(
            "Métrica para el ancho del flujo",
            ("Monto posterior atribuido", "Número de negocios posteriores (atribuidos)"),
            horizontal=True,
        )

    # 🔎 BASE DEL SANKEY: UNA FILA POR (origen_deal_id, deal_id)
    sankey_base = (
        df_post_f
        .drop_duplicates(subset=["origen_deal_id", "deal_id"])
        .copy()
    )

    # SOLO negocios cuyo pipeline de origen es iNBest.marketing
    sankey_base = sankey_base[sankey_base["pipeline_marketing"] == "iNBest.marketing"]

    if sankey_base.empty:
        st.info("No hay datos suficientes para el Sankey con los filtros seleccionados (y origen iNBest.marketing).")
    else:
        sankey_base["pipeline_origen_label"] = sankey_base["pipeline_marketing"]
        sankey_base["pipeline_comercial_label"] = sankey_base["pipeline_comercial"]

        sankey_group = (
            sankey_base.groupby(["pipeline_origen_label", "pipeline_comercial_label"])
            .agg(
                total_amount=("deal_amount", "sum"),
                num_deals=("deal_id", "nunique"),
            )
            .reset_index()
        )

        if sankey_group.empty:
            st.info("No hay datos suficientes para el Sankey después de agrupar.")
        else:
            # Métrica a usar
            if metrica_flujo == "Monto posterior atribuido":
                values = sankey_group["total_amount"].values
            else:
                values = sankey_group["num_deals"].values

            # Nodos: pipelines de origen (izq) y comerciales (der)
            origen_labels = sankey_group["pipeline_origen_label"].unique().tolist()
            destino_labels = sankey_group["pipeline_comercial_label"].unique().tolist()

            # Índices separados para evitar source == target
            origen_index = {label: i for i, label in enumerate(origen_labels)}
            destino_index = {
                label: i + len(origen_labels) for i, label in enumerate(destino_labels)
            }

            labels = origen_labels + destino_labels

            sources = sankey_group["pipeline_origen_label"].map(origen_index).values
            targets = sankey_group["pipeline_comercial_label"].map(destino_index).values

            n_origen = len(origen_labels)
            n_destino = len(destino_labels)

            node_colors = (
                [NEON_BLUE] * n_origen
                + [NEON_GREEN] * n_destino
            )

            link_color = "rgba(56,189,248,0.4)" if metrica_flujo == "Monto posterior atribuido" else "rgba(168,85,247,0.45)"

            fig = go.Figure(
                data=[
                    go.Sankey(
                        arrangement="snap",
                        node=dict(
                            pad=24,
                            thickness=22,
                            line=dict(width=1, color="rgba(148,163,184,0.6)"),
                            label=labels,
                            color=node_colors,
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            color=[link_color] * len(values),
                        ),
                    )
                ]
            )

            fig.update_layout(
                template="plotly_dark",
                height=520,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_MAIN,
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
                **Cómo leer el gráfico:**

                - El bloque de la izquierda es el *pipeline de origen* (iNBest.marketing).
                - Los bloques de la derecha son los *pipelines comerciales* donde aterrizan los negocios posteriores.
                - El grosor de cada flujo representa la métrica seleccionada:
                  - **Monto posterior atribuido**: suma del `deal_amount` de los negocios posteriores.
                  - **Número de negocios posteriores (atribuidos)**: cantidad de deals posteriores distintos.
                """
            )

# -------------------------
# TABLAS ADICIONALES DE PIPELINES / ETAPAS
# -------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📊 Desglose por pipeline y etapa comercial")

# 👉 Usamos deals únicos posteriores para evitar duplicar montos
if df_post_f_unique.empty:
    st.info("No hay datos posteriores con los filtros actuales.")
else:
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("**Top pipelines comerciales por monto posterior (deals únicos)**")
        top_pipelines = (
            df_post_f_unique.groupby("pipeline_comercial")
            .agg(
                num_deals=("deal_id", "nunique"),
                monto_total=("deal_amount", "sum"),
                monto_promedio=("deal_amount", "mean"),
            )
            .reset_index()
            .sort_values("monto_total", ascending=False)
        )
        st.dataframe(
            top_pipelines,
            use_container_width=True,
            hide_index=True,
        )

    with col_t2:
        st.markdown("**Detalle de etapas dentro de un pipeline comercial (deals únicos)**")
        pipelines_disp = sorted(df_post_f_unique["pipeline_comercial"].unique())
        pipeline_sel = st.selectbox(
            "Selecciona pipeline comercial",
            options=pipelines_disp,
        )

        df_etapas = df_post_f_unique[df_post_f_unique["pipeline_comercial"] == pipeline_sel]

        etapas = (
            df_etapas.groupby("etapa_comercial")
            .agg(
                num_deals=("deal_id", "nunique"),
                monto_total=("deal_amount", "sum"),
            )
            .reset_index()
            .sort_values("monto_total", ascending=False)
        )

        st.dataframe(
            etapas,
            use_container_width=True,
            hide_index=True,
        )




