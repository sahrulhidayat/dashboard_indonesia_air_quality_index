import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Dashboard Kualitas Udara Indonesia - Anomalix",
    page_icon="🍃",
    layout="wide",
)

st.title("🍃 Dashboard Monitoring & Analisis Kualitas Udara Indonesia")
st.caption("Dikembangkan oleh **Tim Anomalix** | Lomba Data Science TRUST TECH 2026")


# ==========================================
# LOAD DATASET (DENGAN CACHING UNTUK PERFORMA)
# ==========================================
@st.cache_data
def load_data():
    # Ganti dengan path file parquet Anda
    df = pd.read_parquet("indonesia_air_quality_hourly_clean.parquet")
    df = df.dropna(subset=["us_aqi"]).copy()
    df["time_local"] = pd.to_datetime(df["time_local"])
    df["hour"] = df["time_local"].dt.hour
    return df


with st.spinner("Memuat dataset 1,1+ juta baris..."):
    df = load_data()

# ==========================================
# SIDEBAR FILTER INTERAKTIF
# ==========================================
st.sidebar.header("🔍 Filter Data")

# Filter Pulau
selected_island = st.sidebar.multiselect(
    "Pilih Pulau:", options=df["island"].unique(), default=df["island"].unique()
)

# Filter Kota berdasarkan Pulau yang dipilih
filtered_cities = df[df["island"].isin(selected_island)]["city"].unique()
selected_city = st.sidebar.multiselect(
    "Pilih Kota:",
    options=filtered_cities,
    default=filtered_cities[:5] if len(filtered_cities) >= 5 else filtered_cities,
)

# Filter Rentang Tanggal
min_date = df["time_local"].dt.date.min()
max_date = df["time_local"].dt.date.max()

date_range = st.sidebar.date_input(
    "Rentang Waktu:", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

# Memfilter Dataframe berdasarkan input sidebar
if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (
        (df["island"].isin(selected_island))
        & (df["city"].isin(selected_city))
        & (df["time_local"].dt.date >= start_date)
        & (df["time_local"].dt.date <= end_date)
    )
    df_filtered = df[mask]
else:
    df_filtered = df.copy()

# ==========================================
# METRIK UTAMA (KPI CARDS)
# ==========================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Data Terfilter", f"{len(df_filtered):,} baris")
col2.metric("Rata-rata PM2.5", f"{df_filtered['pm2_5'].mean():.2f} µg/m³")
col3.metric("Rata-rata PM10", f"{df_filtered['pm10'].mean():.2f} µg/m³")
col4.metric("Rata-rata US AQI", f"{df_filtered['us_aqi'].mean():.1f}")

st.markdown("---")

# ==========================================
# TAB VISUALISASI UTAMA
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏙️ Tren Geografis",
        "⏰ Pola Waktu Harian",
        "🔥 Korelasi Polutan",
        "🚨 Anomali & Lonjakan",
    ]
)

# --- TAB 1: TREN GEOGRAFIS ---
with tab1:
    st.subheader("Top Kota dengan Polusi Rata-rata Tertinggi")
    city_summary = (
        df_filtered.groupby("city")["pm2_5"]
        .mean()
        .reset_index()
        .sort_values("pm2_5", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=city_summary, x="pm2_5", y="city", palette="Reds_r", ax=ax)
    ax.set_xlabel("Rata-rata PM2.5 (µg/m³)")
    ax.set_ylabel("Kota")
    st.pyplot(fig)

# --- TAB 2: POLA WAKTU HARIAN ---
with tab2:
    st.subheader("Pola Fluktuasi Polusi Berdasarkan Jam Lokal (00:00 - 23:00)")
    hourly_summary = df_filtered.groupby("hour")["pm2_5"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=hourly_summary,
        x="hour",
        y="pm2_5",
        marker="o",
        color="crimson",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Jam Lokal")
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

# --- TAB 3: MATRIKS KORELASI POLUTAN ---
with tab3:
    st.subheader("Matriks Korelasi Antar Parameter Polutan")
    cols = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
        "dust",
        "uv_index",
        "us_aqi",
    ]
    corr = df_filtered[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# --- TAB 4: DETEKSI ANOMALI EKSTREM ---
with tab4:
    st.subheader("Deteksi Lonjakan Ekstrem PM2.5 (99th Percentile)")
    threshold = df_filtered["pm2_5"].quantile(0.99)
    df_anom = df_filtered[df_filtered["pm2_5"] >= threshold]

    st.warning(
        f"Ambang batas anomali terdeteksi pada nilai PM2.5 ≥ **{threshold:.2f} µg/m³** ({len(df_anom):,} titik data)."
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(
        data=df_anom, x="time_local", y="pm2_5", hue="city", alpha=0.7, s=40, ax=ax
    )
    ax.set_xlabel("Waktu Lokal")
    ax.set_ylabel("PM2.5 Ekstrem (µg/m³)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
