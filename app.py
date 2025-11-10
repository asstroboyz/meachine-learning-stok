import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Helper: Clustering
# =========================
def run_clustering(df: pd.DataFrame, n_clusters: int):
    # Ambil semua kolom numerik
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("Tidak ada kolom numerik yang dapat digunakan untuk clustering.")

    features = df[numeric_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["Cluster"] = labels + 1
    return df_clustered, kmeans, X_scaled, numeric_cols


def run_elbow(df: pd.DataFrame, max_k: int = 10):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    features = df[numeric_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    return sse


# =========================
# Helper: Statistik Pelanggan
# =========================
def compute_cluster_stats(df_clustered: pd.DataFrame, numeric_cols):
    df = df_clustered.copy()

    # 💡 Rata-rata tiap cluster
    cluster_mean = (
        df.groupby("Cluster")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )

    # 💡 Jumlah pelanggan per cluster (cara aman & konsisten)
    cluster_size = (
        df.groupby("Cluster")
        .size()
        .reset_index(name="Jumlah Pelanggan")
        .sort_values(by="Cluster")
        .reset_index(drop=True)
    )

    return df, cluster_mean, cluster_size



# =========================
# Helper: Export Excel
# =========================
def make_excel_report(df_clustered: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_clustered.to_excel(writer, index=False, sheet_name="Hasil Clustering")
    output.seek(0)
    return output.getvalue()


# =========================
# Main App
# =========================
def main():
    st.set_page_config(page_title="Customer Clustering - Toko Salma Company", layout="wide")
    st.title("🧮 Customer Segmentation — Toko Salma Company")

    tabs = st.tabs(["📊 Dashboard", "🧪 Proses Clustering", "📈 Hasil & Analisis", "📄 Laporan"])

    # =======================
    # TAB 1: DASHBOARD
    # =======================
    with tabs[0]:
        st.header("📊 Dashboard Business Intelligence — Analisis Pelanggan")

        st.write("""
Aplikasi ini digunakan untuk menganalisis **data pelanggan** menggunakan algoritma 
**K-Means Clustering** untuk membantu **Toko Salma Company** memahami perilaku pelanggan 
dan membuat keputusan bisnis yang lebih akurat.

Dengan analisis ini, perusahaan dapat mengidentifikasi segmen pelanggan potensial, 
meningkatkan layanan, serta mengoptimalkan strategi pemasaran dan stok produk.
""")

        # tampilkan grafik kalau data sudah ada
        if "df_clustered" in st.session_state:
            df_dashboard = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]

            # 1️⃣ Jumlah pelanggan per cluster
            cluster_count = df_dashboard["Cluster"].value_counts().sort_index()
            st.subheader("1️⃣ Jumlah Pelanggan per Cluster")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=cluster_count.index, y=cluster_count.values, palette="coolwarm", ax=ax1)
            ax1.set_xlabel("Cluster")
            ax1.set_ylabel("Jumlah Pelanggan")
            ax1.set_title("Distribusi Pelanggan Berdasarkan Cluster")
            st.pyplot(fig1)

            # 2️⃣ Rata-rata per cluster
            st.subheader("2️⃣ Rata-rata Nilai per Cluster")
            cluster_mean = df_dashboard.groupby("Cluster")[numeric_cols].mean().round(2)
            st.dataframe(cluster_mean)

            # 3️⃣ KPI Ringkas
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pelanggan", len(df_dashboard))
            col2.metric("Jumlah Cluster", df_dashboard["Cluster"].nunique())
            col3.metric("Cluster Terbesar", int(cluster_count.idxmax()))

            # 4️⃣ Kesimpulan
            st.subheader("🧠 Kesimpulan BI Otomatis")
            st.write(f"""
- **Cluster {cluster_count.idxmax()}** memiliki jumlah pelanggan terbanyak (**{cluster_count.max()}** pelanggan).
- Berdasarkan analisis rata-rata, tiap cluster menunjukkan pola berbeda (misal: pengeluaran tinggi, pembelian sering, dll).
- Hasil ini dapat digunakan untuk menentukan **strategi pemasaran** dan **pengelolaan stok barang** sesuai karakter pelanggan.
- Dengan penerapan *Business Intelligence*, proses pengambilan keputusan menjadi **lebih cepat, akurat, dan berbasis data.**
""")

        else:
            st.info("Belum ada data clustering. Silakan jalankan proses di tab **Proses Clustering** terlebih dahulu.")

    # =======================
    # TAB 2: PROSES CLUSTERING
    # =======================
    with tabs[1]:
        st.header("🧪 Proses Clustering Data Pelanggan")

        uploaded_file = st.file_uploader("Unggah file Excel / CSV data pelanggan", type=["xlsx", "csv"])
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state["df_raw"] = df
            st.subheader("📋 Data Awal")
            st.write(f"Jumlah baris: **{len(df)}**")
            st.dataframe(df, use_container_width=True)

            # Elbow Method
            st.subheader("📈 Grafik Elbow Method")
            sse = run_elbow(df, max_k=10)
            fig, ax = plt.subplots()
            ax.plot(range(1, 11), sse, marker="o")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("SSE")
            ax.set_title("Metode Elbow untuk Menentukan Cluster Optimal")
            st.pyplot(fig)

            # Slider pilih cluster
            n_clusters = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=3)

            # Jalankan clustering
            try:
                df_clustered, kmeans, X_scaled, numeric_cols = run_clustering(df, n_clusters)
            except Exception as e:
                st.error(f"Gagal melakukan clustering: {e}")
                return

            st.session_state["df_clustered"] = df_clustered
            st.session_state["numeric_cols"] = numeric_cols

            # Visualisasi
            if len(numeric_cols) >= 2:
                st.subheader("📊 Visualisasi Dua Dimensi (2 Fitur Pertama)")
                fig, ax = plt.subplots()
                sns.scatterplot(
                    x=X_scaled[:, 0], y=X_scaled[:, 1],
                    hue=df_clustered["Cluster"], palette="Set2", ax=ax
                )
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                st.pyplot(fig)

            # DBI
            dbi = davies_bouldin_score(X_scaled, df_clustered["Cluster"])
            st.write(f"📉 **Davies-Bouldin Index (DBI):** `{dbi:.3f}` — Semakin kecil semakin baik.")
        else:
            st.info("Unggah file pelanggan terlebih dahulu.")

    # =======================
    # TAB 3: HASIL & ANALISIS
    # =======================
    with tabs[2]:
        st.header("📈 Hasil dan Analisis Clustering")
        if "df_clustered" not in st.session_state:
            st.warning("Belum ada hasil clustering.")
        else:
            df_clustered = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]

            df_with_stats, cluster_mean, cluster_size = compute_cluster_stats(df_clustered, numeric_cols)
            st.session_state["df_with_stats"] = df_with_stats

            st.subheader("📊 Data Pelanggan dengan Cluster")
            st.dataframe(df_with_stats, use_container_width=True)

            st.subheader("📋 Statistik Tiap Cluster")
            st.table(cluster_mean)

            st.subheader("👥 Jumlah Pelanggan per Cluster")
            st.table(cluster_size)

    # =======================
    # TAB 4: LAPORAN
    # =======================
    with tabs[3]:
        st.header("📄 Unduh Laporan Hasil Clustering")
        if "df_with_stats" not in st.session_state:
            st.warning("Belum ada data hasil clustering.")
        else:
            df_report = st.session_state["df_with_stats"]
            excel_bytes = make_excel_report(df_report)
            st.download_button(
                label="📥 Unduh Laporan Excel",
                data=excel_bytes,
                file_name="Laporan_Clustering_Pelanggan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.success("✅ Laporan siap diunduh!")


if __name__ == "__main__":
    main()
