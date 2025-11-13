import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==== CONFIGURABLE ====
MONEY_COLS = ["Total Spent", "Avg Order Value"]

# ============== Helper: Clustering ==============
def safe_numeric_convert(df: pd.DataFrame):
    """Pastikan hanya kolom numerik yang dikonversi"""
    for col in df.columns:
        # ❌ Abaikan kolom non-numerik penting
        if any(keyword in col.lower() for keyword in ["produk", "favorit", "nama", "kategori", "jenis"]):
            continue

        if df[col].dtype == "object":
            # Coba parse tanggal
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() > 0:
                    df[col] = (pd.Timestamp.today() - parsed).dt.days
                    continue
            except Exception:
                pass

        # Konversi ke numerik kalau bisa
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def run_clustering(df: pd.DataFrame, n_clusters: int):
    df = safe_numeric_convert(df)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("Tidak ada kolom numerik yang dapat digunakan untuk clustering.")
    
    features = df[numeric_cols].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels + 1
    return df_clustered, kmeans, X_scaled, numeric_cols


def run_elbow(df: pd.DataFrame, max_k: int = 10):
    df = safe_numeric_convert(df)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    features = df[numeric_cols].fillna(0).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    return sse


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


def parse_rupiah(x):
    if isinstance(x, str):
        return float(re.sub(r"[^\d]", "", x)) if re.sub(r"[^\d]", "", x) else 0
    return x


def show_df_with_rupiah(df, money_cols=MONEY_COLS, **kwargs):
    df_show = df.copy()
    for col in money_cols:
        if col in df_show.columns:
            df_show[col] = df_show[col].apply(format_rupiah)
    st.dataframe(df_show, width="stretch", **kwargs)


def compute_cluster_stats(df_clustered: pd.DataFrame, numeric_cols: list):
    cluster_mean = df_clustered.groupby("Cluster")[numeric_cols].mean().round(2)
    cluster_size = df_clustered["Cluster"].value_counts().sort_index().reset_index()
    cluster_size.columns = ["Cluster", "Jumlah Pelanggan"]

    df_with_stats = df_clustered.copy()
    for col in numeric_cols:
        df_with_stats[f"{col} (Cluster Mean)"] = df_with_stats["Cluster"].map(cluster_mean[col])

    return df_with_stats, cluster_mean, cluster_size


# ================== MAIN ==================
def main():
    st.set_page_config(page_title="Salma Company", layout="wide")
    st.title("🧮 Salma Company")

    if "n_clusters" not in st.session_state:
        st.session_state["n_clusters"] = 3

    tabs = st.tabs([
        "📊 Dashboard",
        "🧪 Proses Clustering",
        "📈 Analisis Tiap Cluster",
        "📄 Laporan"
    ])
    # =========== TAB 2: PROSES CLUSTERING ===========
    with tabs[1]:
        st.header("🧪 Proses Clustering Data Pelanggan")
        uploaded_file = st.file_uploader("Unggah file Excel / CSV data pelanggan", type=["xlsx", "csv"])

        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            df = pd.read_excel(uploaded_file) if filename.endswith(".xlsx") else pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.title()

            for col in MONEY_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(parse_rupiah)

            st.subheader("📋 Data Awal")
            st.dataframe(df.head(), width="stretch")

            # Slider cluster (aman, tidak bentrok session state)
            n_clusters = st.slider(
                "Pilih jumlah cluster (k)",
                2, 10,
                value=st.session_state["n_clusters"],
                key="slider_n_clusters"
            )
            st.session_state["n_clusters"] = n_clusters

            # Jalankan clustering
            try:
                df_clustered, kmeans, X_scaled, numeric_cols = run_clustering(df, n_clusters)

                # Simpan ke session state (AMAN)
                st.session_state["df_clustered"] = df_clustered
                st.session_state["numeric_cols"] = numeric_cols
                st.session_state["X_scaled"] = X_scaled

                st.success("✅ Clustering selesai — Dashboard & Analisis otomatis diperbarui.")

            except Exception as e:
                st.error(f"Gagal melakukan clustering: {e}")
                st.stop()

            # ==================
            #   ELBOW METHOD
            # ==================
            st.subheader("📈 Elbow Method")
            sse = run_elbow(df)
            fig, ax = plt.subplots()
            ax.plot(range(1, len(sse) + 1), sse, marker="o")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("SSE")
            ax.set_title("Grafik Elbow Method")
            st.pyplot(fig)

            st.markdown("### 🧮 Tabel Nilai SSE")
            sse_table = pd.DataFrame({"Jumlah Cluster": range(1, len(sse) + 1), "SSE": sse})
            st.table(sse_table)

            # ==================
            #   FIX UTAMA
            # ==================
            if "df_clustered" not in st.session_state:
                st.warning("⚠️ Hasil clustering belum tersedia.")
                st.stop()

            df_clustered = st.session_state["df_clustered"]
            X_scaled = st.session_state["X_scaled"]

            # DBI SCORE
            st.subheader("📉 Nilai Davies-Bouldin Index (DBI)")
            dbi = davies_bouldin_score(X_scaled, df_clustered["Cluster"])
            st.info(f"DBI = `{dbi:.3f}` (semakin kecil semakin baik)")

            # ============================
            #   TABEL PER CLUSTER (AMAN)
            # ============================
            st.write("### 📋 Tabel Hasil Clustering per Cluster:")

            for cluster in sorted(df_clustered["Cluster"].unique()):
                st.write(f"#### Cluster {cluster}")
                df_sub = df_clustered[df_clustered["Cluster"] == cluster]
                show_df_with_rupiah(df_sub)

            # ============================
            #   SUMMARY CLUSTER
            # ============================
            st.markdown("### 📊 Summary Jumlah Data per Cluster")
            st.dataframe(
                df_clustered["Cluster"]
                .value_counts()
                .sort_index()
                .reset_index()
                .rename(columns={'index': 'Cluster', 'Cluster': 'Jumlah Data'}),
                width="stretch"
            )
    # =========== TAB 1: DASHBOARD ===========
    with tabs[0]:

        total_pelanggan = 0
        total_cluster = 0
        dbi_score = "-"
        best_cluster = "-"
        worst_cluster = "-"
        
        df_dashboard = None

        if "df_clustered" in st.session_state:
            df_dashboard = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]
            X_scaled = st.session_state["X_scaled"]

            # ==============================
            #   HEADER KPI CLUSTERING
            # ==============================
            total_pelanggan = len(df_dashboard)
            total_cluster = df_dashboard["Cluster"].nunique()
            dbi_score = round(davies_bouldin_score(X_scaled, df_dashboard["Cluster"]), 3)
            best_cluster = df_dashboard.groupby("Cluster")["Total Spent"].mean().idxmax()
            worst_cluster = df_dashboard.groupby("Cluster")["Total Spent"].mean().idxmin()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style='background:#1D4ED8; padding:18px; border-radius:12px; color:white; text-align:center;'>
                <h4>👥</h4>
                <p>Total Pelanggan</p>
                <h2>{total_pelanggan:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background:#0EA5E9; padding:18px; border-radius:12px; color:white; text-align:center;'>
                <h4>📦</h4>
                <p>Jumlah Cluster</p>
                <h2>{total_cluster}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='background:#22C55E; padding:18px; border-radius:12px; color:white; text-align:center;'>
                <h4>📉</h4>
                <p>DBI Score</p>
                <h2>{dbi_score}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style='background:#9333EA; padding:18px; border-radius:12px; color:white; text-align:center;'>
                <h4>💎</h4>
                <p>Cluster Terbaik</p>
                <h2>Cluster {best_cluster}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if df_dashboard is None:
            st.info("Silakan upload file di tab **Proses Clustering** untuk melihat grafik dan analisis.")
            st.stop()
            # ==============================
            #   DESKRIPSI MACHINE LEARNING
            # ==============================
        st.markdown("""
        ### 🤖 Machine Learning Insight 
        Segmentasi pelanggan dilakukan menggunakan **K-Means Clustering** berdasarkan pola belanja, total pengeluaran, dan frekuensi transaksi.
        Tujuan analisis:
        - Mengelompokkan pelanggan berdasarkan daya beli
        - Menentukan prioritas stok
        - Membantu strategi promo
        """)

            # ==============================
            # 1️⃣ DISTRIBUSI PELANGGAN PER CLUSTER
            # ==============================
        st.subheader("1️⃣ Distribusi Pelanggan per Cluster")
        cluster_count = df_dashboard["Cluster"].value_counts().sort_index()

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.barplot(x=cluster_count.index, y=cluster_count.values, palette="viridis", ax=ax1)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Jumlah Pelanggan")
        ax1.set_title("Distribusi Jumlah Pelanggan per Cluster")
        st.pyplot(fig1)

            # ==============================
            # 2️⃣ NILAI TRANSAKSI RATA-RATA
            # ==============================
        st.subheader("2️⃣ Nilai Transaksi Rata-Rata")
        cluster_mean = df_dashboard.groupby("Cluster")[["Total Spent", "Avg Order Value"]].mean().round(2)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        cluster_mean.plot(kind="bar", ax=ax2, color=["#3b82f6", "#10b981"])
        ax2.set_title("Rata-Rata Nilai Transaksi per Cluster")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Rata-rata (Rp)")
        st.pyplot(fig2)

            # ==============================
            # 3️⃣ PRODUK FAVORIT PER CLUSTER
            # ==============================
        if "Produk Favorit" in df_dashboard.columns:
            st.subheader("3️⃣ Produk Favorit per Cluster")

            for cluster in sorted(df_dashboard["Cluster"].unique()):
                st.markdown(f"#### Cluster {cluster}")
                top_product = df_dashboard[df_dashboard["Cluster"] == cluster]["Produk Favorit"].value_counts().head(5)

                if not top_product.empty:
                    st.table(
                        top_product.reset_index().rename(
                            columns={"index": "Produk", "Produk Favorit": "Jumlah Pembelian"}
                        )
                    )
                else:
                    st.info("Tidak ada data produk favorit.")

        else:
            st.info("Belum ada data clustering. Silakan lakukan proses di tab **Proses Clustering**.")

    # ============================================================
    # =========== TAB 3: ANALISIS TIAP CLUSTER ====================
    # ============================================================
    with tabs[2]:
        st.header("📈 Analisis Tiap Cluster")

        if "df_clustered" not in st.session_state:
            st.warning("Belum ada hasil clustering.")
            st.stop()

        df_clustered = st.session_state["df_clustered"]
        numeric_cols = st.session_state["numeric_cols"]
        n_clusters = st.session_state["n_clusters"]

        st.info(f"Jumlah cluster: {n_clusters}")

        df_with_stats, cluster_mean, cluster_size = compute_cluster_stats(df_clustered, numeric_cols)

        st.subheader("📊 Data Pelanggan dengan Statistik Cluster")
        show_df_with_rupiah(df_with_stats)

        # Statistik cluster
        st.subheader("📋 Statistik Tiap Cluster")
        cluster_mean_fmt = cluster_mean.copy()
        for col in MONEY_COLS:
            if col in cluster_mean_fmt.columns:
                cluster_mean_fmt[col] = cluster_mean_fmt[col].apply(format_rupiah)
        st.table(cluster_mean_fmt)

        st.subheader("👥 Jumlah Pelanggan per Cluster")
        st.dataframe(cluster_size)

        # ============================
        # ANALISIS PER CLUSTER
        # ============================
        st.markdown("### 🧠 Analisis & Interpretasi")

        for cluster in sorted(df_clustered["Cluster"].unique()):
            subset = df_clustered[df_clustered["Cluster"] == cluster]
            total = len(subset)

            spent_mean = subset["Total Spent"].mean()
            order_mean = subset["Avg Order Value"].mean()
            recency_mean = subset["Recency Days"].mean() if "Recency Days" in subset.columns else None

            top_product = (
                subset["Produk Favorit"].mode()[0]
                if "Produk Favorit" in subset.columns and not subset["Produk Favorit"].dropna().empty
                else "Tidak ada"
            )

            st.markdown(f"#### 📦 Cluster {cluster}")
            show_df_with_rupiah(subset)

            st.markdown(f"""
            **Analisis Ringkas**
            - Jumlah pelanggan: **{total}**
            - Pengeluaran rata-rata: **{format_rupiah(spent_mean)}**
            - Nilai transaksi rata-rata: **{format_rupiah(order_mean)}**
            - Produk dominan: **{top_product}**
            """)

    # ============================================================
    # =========== TAB 4: LAPORAN =================================
    # ============================================================
    with tabs[3]:
        st.markdown("""
        ### 📄 Unduh Laporan Clustering
        Download laporan lengkap hasil clustering dalam format Excel.
        """)

        if "df_clustered" in st.session_state:
            df_report = st.session_state["df_clustered"]
            excel_bytes = make_excel_report(df_report)

            st.download_button(
                label="⬇️ Unduh Laporan Excel",
                data=excel_bytes,
                file_name="Laporan_Clustering_Pelanggan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("Belum ada hasil clustering.")

# Run App
if __name__ == "__main__":
    main()
