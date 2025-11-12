import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import re
import itertools

# ==== CONFIGURABLE (tinggal ganti kalau Excel berubah header) ====
MONEY_COLS = ["Total Spent", "Avg Order Value"]

# ============== Helper: Clustering ==============
def run_clustering(df: pd.DataFrame, n_clusters: int):
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

def compute_cluster_stats(df_clustered: pd.DataFrame, numeric_cols):
    df = df_clustered.copy()
    cluster_mean = (
        df.groupby("Cluster")[numeric_cols]
        .mean()
        .round(2)
        .reset_index()
    )
    cluster_size = (
        df.groupby("Cluster")
        .size()
        .reset_index(name="Jumlah Pelanggan")
        .sort_values(by="Cluster")
        .reset_index(drop=True)
    )
    return df, cluster_mean, cluster_size

def make_excel_report(df_clustered: pd.DataFrame):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_clustered.to_excel(writer, index=False, sheet_name="Hasil Clustering")
    output.seek(0)
    return output.getvalue()

def format_rupiah(x):
    try:
        x = float(x)
        return "Rp {:,.0f}".format(x).replace(",", ".")
    except:
        return x 

def show_df_with_rupiah(df, money_cols=MONEY_COLS, **kwargs):
    df_show = df.copy()
    for col in money_cols:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(format_rupiah)
    st.dataframe(df_show, **kwargs)

def show_table_with_rupiah(df, money_cols=MONEY_COLS):
    df_show = df.copy()
    for col in money_cols:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(format_rupiah)
    st.table(df_show)

def parse_rupiah(x):
    if isinstance(x, str):
        return float(re.sub(r"[^\d]", "", x))
    return x

# ============== Main App ==============
def main():
    st.set_page_config(page_title="Customer Clustering - Toko Salma Company", layout="wide")
    st.title("🧮Salma Company")

    if "n_clusters" not in st.session_state:
        st.session_state["n_clusters"] = 3

    tabs = st.tabs([
        "📊 Dashboard", 
        "🧪 Proses Clustering", 
        "📈 Hasil & Analisis", 
        "📄 Laporan"
    ])

    # =========== TAB 1: DASHBOARD ===========
    with tabs[0]:
        st.header("📊 Dashboard Business Intelligence — Analisis Pelanggan")

        st.write("""
Aplikasi ini digunakan untuk menganalisis **data pelanggan** menggunakan algoritma 
**K-Means Clustering** untuk membantu **Toko Salma Company** memahami perilaku pelanggan 
dan membuat keputusan bisnis yang lebih akurat.

Dengan analisis ini, perusahaan dapat mengidentifikasi segmen pelanggan potensial, 
meningkatkan layanan, serta mengoptimalkan strategi pemasaran dan stok produk.
""")
        if "df_clustered" in st.session_state:
            df_dashboard = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]

            cluster_count = df_dashboard["Cluster"].value_counts().sort_index()
            st.subheader("1️⃣ Jumlah Pelanggan per Cluster")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(x=cluster_count.index, y=cluster_count.values, palette="coolwarm", ax=ax1)
            ax1.set_xlabel("Cluster")
            ax1.set_ylabel("Jumlah Pelanggan")
            ax1.set_title("Distribusi Pelanggan Berdasarkan Cluster")
            for i, v in enumerate(cluster_count.values):
                ax1.text(i, v + 0.2, str(v), ha='center', va='bottom')
            st.pyplot(fig1)

            st.subheader("2️⃣ Statistik Fitur per Cluster")
            cluster_mean = df_dashboard.groupby("Cluster")[numeric_cols].mean().round(2)
            for col in ["Total Spent", "Avg Order Value"]:
                if col in cluster_mean.columns:
                   cluster_mean[col] = cluster_mean[col].apply(format_rupiah)
            st.dataframe(cluster_mean)
            
            if "Produk Favorit" in df_dashboard.columns:
                st.subheader("🏆 Top 5 Produk Favorit Seluruh Pelanggan")
                top_product = df_dashboard["Produk Favorit"].value_counts().head(5)
                st.table(
                    top_product.reset_index().rename(
                        columns={"index": "Produk", "Produk Favorit": "Jumlah Pembelian"}
                    )
                )
                st.subheader("🏆 Produk Favorit per Cluster")
                for cluster in sorted(df_dashboard["Cluster"].unique()):
                    st.write(f"#### Cluster {cluster}")
                    top_product = df_dashboard[df_dashboard["Cluster"] == cluster]["Produk Favorit"].value_counts().head(3)
                    st.table(
                        top_product.reset_index().rename(
                            columns={"index": "Produk", "Produk Favorit": "Jumlah Pembelian"}
                        )
                    )

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pelanggan", len(df_dashboard))
            col2.metric("Jumlah Cluster", df_dashboard["Cluster"].nunique())
            col3.metric("Cluster Terbesar", int(cluster_count.idxmax()))

            st.subheader("🧠 Ringkasan")
            st.write(f"""
- **Cluster {cluster_count.idxmax()}** memiliki jumlah pelanggan terbanyak (**{cluster_count.max()}** pelanggan).
- Statistik rata-rata fitur tiap cluster berbeda, gunakan hasil ini untuk segmentasi promosi dan pengelolaan stok!
- **Jumlah cluster aktif: {st.session_state['n_clusters']}**
""")
            if "Produk Favorit" in df_dashboard.columns:
                most_fav = df_dashboard["Produk Favorit"].mode()[0]
                st.info(f"💡 **Insight :** Produk favorit pelanggan saat ini adalah **{most_fav}**. Pertimbangkan untuk menambah stok atau memprioritaskan promosi produk ini.")

        else:
            st.info("Belum ada data clustering. Silakan jalankan proses di tab **Proses Clustering** terlebih dahulu.")


    # =========== TAB 2: PROSES CLUSTERING ===========
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

            for col in MONEY_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(parse_rupiah)

            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            if len(numeric_cols) == 0:
                st.error("Tidak ada kolom numerik untuk clustering.")
                st.stop()
            df_numerical = df[numeric_cols]

            # Elbow Method
            st.subheader("📈 Grafik Elbow Method")
            max_k = 10
            sse = run_elbow(df, max_k)
            fig, ax = plt.subplots()
            ax.plot(range(1, max_k + 1), sse, marker="o")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("SSE")
            ax.set_title("Metode Elbow untuk Menentukan Cluster Optimal")
            st.pyplot(fig)
            st.markdown("<p style='font-size:25px;'>3. Tabel Nilai SSE (Sum of Squared Errors)</p>", unsafe_allow_html=True)
            sse_table = pd.DataFrame({
                "Jumlah Cluster": range(1, max_k + 1),
                "SSE": sse
            })
            st.table(sse_table)
            st.write("SSE (Sum of Squared Errors) adalah ukuran untuk menilai seberapa baik model klasterisasi. Semakin kecil nilai SSE, semakin baik model memisahkan data.")

            n_clusters = st.slider(
                "Pilih jumlah cluster (k)", min_value=2, max_value=10,
                value=st.session_state["n_clusters"], key="n_clusters"
            )

            if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != filename \
                or st.session_state.get("last_n_clusters", None) != n_clusters \
                or "df_clustered" not in st.session_state:
                try:
                    df_clustered, kmeans, X_scaled, used_numeric_cols = run_clustering(df, n_clusters)
                    st.session_state["df_clustered"] = df_clustered
                    st.session_state["numeric_cols"] = used_numeric_cols
                    st.session_state["X_scaled"] = X_scaled
                    st.session_state["last_uploaded"] = filename
                    st.session_state["last_n_clusters"] = n_clusters
                except Exception as e:
                    st.error(f"Gagal melakukan clustering: {e}")
                    st.stop()
            else:
                df_clustered = st.session_state["df_clustered"]
                used_numeric_cols = st.session_state["numeric_cols"]
                X_scaled = st.session_state["X_scaled"]

            # Visualisasi: scatterplot semua kombinasi fitur
            st.subheader("📊 Visualisasi Cluster pada Berbagai Fitur")
            combis = list(itertools.combinations(used_numeric_cols, 2))
            if len(combis) == 0:
                st.info("Tidak ada kombinasi dua fitur numerik untuk divisualisasikan.")
            else:
                for i, (col_x, col_y) in enumerate(combis, start=1):
                    st.markdown(f"**{i}. Scatterplot: `{col_x}` vs `{col_y}`**")
                    fig, ax = plt.subplots()
                    sns.scatterplot(
                        x=col_x, y=col_y,
                        hue='Cluster',
                        data=df_clustered,
                        palette='Set1',
                        ax=ax
                    )
                    ax.set_title(f"Cluster pada '{col_x}' vs '{col_y}'")
                    ax.legend(title='Cluster')
                    st.pyplot(fig)

            # DBI
            st.subheader("📉 Hasil dari Davies-Bouldin Index (DBI)")
            dbi = davies_bouldin_score(X_scaled, df_clustered["Cluster"])
            st.write(f"📉 **Davies-Bouldin Index (DBI):** `{dbi:.3f}` — Semakin kecil semakin baik.")
            st.write("Nilai DBI yang lebih rendah menunjukkan hasil clustering yang lebih baik.")
           
            # Tabel hasil clustering per cluster (format rupiah)
            st.write("### Tabel Hasil Clustering per Cluster:")
            for cluster in sorted(df_clustered['Cluster'].unique()):
                st.write(f"#### Cluster {cluster}")
                df_sub = df_clustered[df_clustered['Cluster'] == cluster]
                show_df_with_rupiah(df_sub, use_container_width=True)

            st.markdown("### Summary Jumlah Data per Cluster")
            st.dataframe(
                df_clustered['Cluster'].value_counts().sort_index().reset_index().rename(
                    columns={'index': 'Cluster', 'Cluster': 'Jumlah Data'})
            )
        else:
            st.info("Unggah file pelanggan terlebih dahulu.")

    # =========== TAB 3: HASIL & ANALISIS ===========
    with tabs[2]:
        st.header("📈 Hasil dan Analisis Clustering")
        if "df_clustered" not in st.session_state:
            st.warning("Belum ada hasil clustering.")
        else:
            df_clustered = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]
            n_clusters = st.session_state["n_clusters"]

            st.info(f"Jumlah cluster aktif: {n_clusters}")

            df_with_stats, cluster_mean, cluster_size = compute_cluster_stats(df_clustered, numeric_cols)
            st.session_state["df_with_stats"] = df_with_stats

            st.subheader("📊 Data Pelanggan dengan Cluster")
            show_df_with_rupiah(df_with_stats, use_container_width=True)

            cluster_mean_fmt = cluster_mean.copy()
            for col in MONEY_COLS:
                if col in cluster_mean_fmt.columns:
                    cluster_mean_fmt[col] = cluster_mean_fmt[col].apply(format_rupiah)
            st.subheader("📋 Statistik Tiap Cluster (Rata-rata Fitur)")
            st.table(cluster_mean_fmt)
            
            st.subheader("🗂️ Tabel Hasil Clustering per Cluster")
            for cluster in sorted(df_clustered['Cluster'].unique()):
                st.write(f"#### Cluster {cluster}")
                df_sub = df_clustered[df_clustered['Cluster'] == cluster]
                show_df_with_rupiah(df_sub, use_container_width=True)

            st.subheader("👥 Jumlah Pelanggan per Cluster")
            st.table(cluster_size)

    # =========== TAB 4: LAPORAN ===========
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
