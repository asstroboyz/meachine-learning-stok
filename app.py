import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import re
import streamlit.components.v1 as components
# ==== CONFIGURABLE ====
MONEY_COLS = ["Harga (Rp)", "Total Pendapatan (Rp)"]

# ============== Helper: Clustering ==============
def safe_numeric_convert(df: pd.DataFrame):
    """Konversi otomatis ke numerik, abaikan kolom teks."""
    for col in df.columns:
        # abaikan kolom non-numerik
        if any(keyword in col.lower() for keyword in ["produk", "nama", "kategori", "id"]):
            continue

        if df[col].dtype == "object":
            # parse rupiah atau angka lain
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d]", "", regex=True), errors="coerce")

    return df


def run_clustering(df: pd.DataFrame, n_clusters: int):
    df = safe_numeric_convert(df)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("Tidak ada kolom numerik untuk clustering.")

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
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
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
        return "Rp {:,.0f}".format(float(x)).replace(",", ".")
    except:
        return x


def parse_rupiah(x):
    if isinstance(x, str):
        x = re.sub(r"[^\d]", "", x)
        return float(x) if x else 0
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
    cluster_size.columns = ["Cluster", "Jumlah Produk"]

    df_with_stats = df_clustered.copy()
    for col in numeric_cols:
        df_with_stats[f"{col} (Cluster Mean)"] = df_with_stats["Cluster"].map(cluster_mean[col])

    return df_with_stats, cluster_mean, cluster_size


# ================== MAIN ==================
def main():
    st.set_page_config(page_title="Salma Company", layout="wide")
    st.title("🧶 ANALISIS CLUSTER PRODUK — Salma Company")

    if "n_clusters" not in st.session_state:
        st.session_state["n_clusters"] = 3

    tabs = st.tabs([
        "📊 Dashboard",
        "🧪 Proses Clustering",
        "📈 Analisis Tiap Cluster",
        "📄 Laporan"
    ])

    # ================================
    # TAB 2 — PROSES CLUSTERING
    # ================================
    with tabs[1]:
        st.header("🧪 Proses Clustering Data Produk")

        uploaded_file = st.file_uploader("Unggah file Excel / CSV data produk", type=["xlsx", "csv"])

        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            df = pd.read_excel(uploaded_file) if filename.endswith(".xlsx") else pd.read_csv(uploaded_file)

            # Normalisasi nama kolom
            df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip().str.title()

            # Parse rupiah
            for col in MONEY_COLS:
                if col in df.columns:
                    df[col] = df[col].apply(parse_rupiah)

            st.subheader("📋 Data Awal")
            st.dataframe(df.head(), width="stretch")

            n_clusters = st.slider("Jumlah cluster (k)", 2, 10, value=st.session_state["n_clusters"])
            st.session_state["n_clusters"] = n_clusters

            try:
                df_clustered, kmeans, X_scaled, numeric_cols = run_clustering(df, n_clusters)
                st.session_state["df_clustered"] = df_clustered
                st.session_state["numeric_cols"] = numeric_cols
                st.session_state["X_scaled"] = X_scaled

                st.success("✅ Clustering selesai!")

            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

            # Elbow
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

    # ================================
    # TAB 1 — DASHBOARD
    # ================================
    with tabs[0]:

        total_items = 0
        total_cluster = 0
        dbi_score = "-"
        best_cluster = "-"
        worst_cluster = "-"
        df_dashboard = None

        if "df_clustered" in st.session_state:
            df_dashboard = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]
            X_scaled = st.session_state["X_scaled"]

            total_items = len(df_dashboard)
            total_cluster = df_dashboard["Cluster"].nunique()
            dbi_score = round(davies_bouldin_score(X_scaled, df_dashboard["Cluster"]), 3)

            best_cluster = df_dashboard.groupby("Cluster")["Total Pendapatan (Rp)"].mean().idxmax()
            worst_cluster = df_dashboard.groupby("Cluster")["Total Pendapatan (Rp)"].mean().idxmin()

        # Header Cards
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style='background:#1D4ED8; padding:18px; border-radius:12px; color:white; text-align:center;'>
                <h4>📦</h4>
                <p>Total Produk</p>
                <h2>{total_items:,}</h2>
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
            st.info("Upload data terlebih dahulu.")
            st.stop()

        # Distribusi cluster
        components.html("""
        <style>
        .insight-box {
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            padding: 26px;
            border-radius: 20px;
            color: white;
            box-shadow: 0 6px 18px rgba(0,0,0,0.20);
            margin-bottom: 25px;
        }
        .insight-title {
            font-size: 26px;
            margin-top: 0;
            margin-bottom: 10px;
            font-weight: bold;
        }
        .insight-text {
            font-size: 16px;
            line-height: 1.5;
        }
        .insight-list {
            font-size: 16px;
            margin-left: 25px;
            line-height: 1.6;
        }
        </style>

        <div class="insight-box">
            <div class="insight-title">🤖 Machine Learning Insight</div>

            <p class="insight-text">
                Analisis cluster dilakukan terhadap <strong>produk</strong>, berdasarkan:
            </p>

            <ul class="insight-list">
                <li>Jumlah terjual</li>
                <li>Total pendapatan</li>
                <li>Stok tersisa</li>
            </ul>

            <p class="insight-text" style="margin-top:14px;">
                <strong>Tujuan Analisis:</strong>
            </p>

            <ul class="insight-list">
                <li>Mengidentifikasi produk paling laris</li>
                <li>Menentukan produk slow-moving</li>
                <li>Menentukan prioritas produksi / restock</li>
                <li>Mendukung strategi promosi & bundling</li>
            </ul>
        </div>
        """, height=420)


        #####################################
        # st.subheader("1️⃣ Distribusi Produk per Cluster")
        # cluster_count = df_dashboard["Cluster"].value_counts().sort_index()

        # fig1, ax1 = plt.subplots(figsize=(7, 4))
        # sns.barplot(x=cluster_count.index, y=cluster_count.values, palette="viridis", ax=ax1)
        # ax1.set_xlabel("Cluster")
        # ax1.set_ylabel("Jumlah Produk")
        # ax1.set_title("Distribusi Jumlah Produk per Cluster")
        # st.pyplot(fig1)
        #####################################
        
        
        # st.subheader("2️⃣ Rata-Rata Penjualan & Pendapatan per Cluster")
        # cluster_mean = df_dashboard.groupby("Cluster")[["Total Pendapatan (Rp)", "Jumlah Terjual"]].mean().round(2)

        # fig2, ax2 = plt.subplots(figsize=(7, 4))
        # cluster_mean.plot(kind="bar", ax=ax2, color=["#3b82f6", "#10b981"])
        # ax2.set_title("Rata-Rata Pendapatan & Jumlah Terjual per Cluster")
        # ax2.set_xlabel("Cluster")
        # ax2.set_ylabel("Nilai")
        # st.pyplot(fig2)
        
        #####################################
        # st.subheader("2️⃣ Rata-Rata Pendapatan & Jumlah Terjual per Cluster")
        # cluster_mean = df_dashboard.groupby("Cluster")[["Total Pendapatan (Rp)", "Jumlah Terjual"]].mean().round(2)
        # fig, ax1 = plt.subplots(figsize=(10, 5))
        # ax2 = ax1.twinx()
        # # Pendapatan
        # b1 = ax1.bar(
        #     cluster_mean.index - 0.2,
        #     cluster_mean["Total Pendapatan (Rp)"],
        #     width=0.4,
        #     color="#3b82f6",
        #     label="Total Pendapatan (Rp)"
        # )
        # # Jumlah Terjual
        # b2 = ax2.bar(
        #     cluster_mean.index + 0.2,
        #     cluster_mean["Jumlah Terjual"],
        #     width=0.4,
        #     color="#10b981",
        #     label="Jumlah Terjual"
        # )
        # # Format Rupiah
        # import matplotlib.ticker as ticker
        # ax1.yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda x, pos: f"Rp {x:,.0f}".replace(",", "."))
        # )
        # ax1.set_ylim(0, cluster_mean["Total Pendapatan (Rp)"].max() * 1.25)
        # ax2.set_ylim(0, cluster_mean["Jumlah Terjual"].max() * 1.25)
        # ax1.set_ylabel("Pendapatan (Rp)", color="#3b82f6")
        # ax2.set_ylabel("Jumlah Terjual", color="#10b981")
        # ax1.set_xlabel("Cluster")
        # ax1.set_title("Rata-Rata Pendapatan & Jumlah Terjual per Cluster")
        # ax1.grid(axis="y", linestyle="--", alpha=0.3)
        # ax1.set_xticks(cluster_mean.index)
        # fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92))
        # st.pyplot(fig)
#####################################

        
        
        #####################################
        # st.subheader("3️⃣ Produk Favorit per Cluster (Penjualan Terbanyak)")
        # if "Jumlah Terjual" in df_dashboard.columns and "Nama_Produk" in df_dashboard.columns:
        #     for cluster in sorted(df_dashboard["Cluster"].unique()):
        #         st.markdown(f"#### ⭐ Cluster {cluster}")

        #         subset = df_dashboard[df_dashboard["Cluster"] == cluster]

        #         # Urutkan berdasarkan Jumlah Terjual
        #         top_products = subset.sort_values(
        #             by="Jumlah Terjual",
        #             ascending=False
        #         ).head(5)

        #         if not top_products.empty:
        #             st.table(
        #                 top_products[[
        #                     "Id Produk",
        #                     "Nama_Produk",
        #                     "Kategori",
        #                     "Jumlah Terjual",
        #                     "Total Pendapatan (Rp)",
        #                     "Stok Tersisa"
        #                 ]]
        #             )
        #         else:
        #             st.info("Tidak ada data untuk cluster ini.")
        # else:
        #     st.warning("Kolom 'Jumlah Terjual' atau 'Nama_Produk' tidak ditemukan.")
        #####################################    

        # ============================
#  4️⃣ ANALISIS STOK — BI SALMA COMPANY
# ============================

        st.markdown("---")
        st.subheader("4️⃣ Analisis Kesehatan Stok (Stock Health Analysis)")

        # --- Fungsi kategori stok ---
        def categorize_stock(row):
            if row["Jumlah Terjual"] == 0:
                return "Deadstock"
            elif row["Stok Tersisa"] > row["Jumlah Terjual"] * 3:
                return "Overstock"
            elif row["Stok Tersisa"] <= 5:
                return "Critical Reorder"
            elif row["Stok Tersisa"] <= row["Jumlah Terjual"]:
                return "Understock"
            return "Normal"

        df_dashboard["Stock_Health"] = df_dashboard.apply(categorize_stock, axis=1)

        # Pie Chart Stock Health
        stock_health_count = df_dashboard["Stock_Health"].value_counts()

        fig_stock, axsh = plt.subplots(figsize=(6, 6))
        axsh.pie(
            stock_health_count,
            labels=stock_health_count.index,
            autopct='%1.1f%%',
            startangle=90,
        )
        axsh.set_title("Distribusi Kesehatan Stok Produk")
        st.pyplot(fig_stock)

        st.write("### 🧾 Detail Kesehatan Stok")
        st.dataframe(df_dashboard[["Nama_Produk", "Jumlah Terjual", "Stok Tersisa", "Stock_Health"]], width="stretch")

        # ============================
        #  5️⃣ REORDER RECOMMENDATION
        # ============================

        st.markdown("---")
        st.subheader("5️⃣ Rekomendasi Pembelian Ulang (Reorder Recommendation)")

        df_dashboard["Prediksi_Habis_Hari"] = (
            df_dashboard["Stok Tersisa"] / df_dashboard["Jumlah Terjual"].replace(0, 0.1)
        ).round(1)

        def rekomendasi(row):
            if row["Prediksi_Habis_Hari"] <= 7:
                return "Beli Sekarang"
            elif row["Prediksi_Habis_Hari"] <= 30:
                return "Monitor"
            return "Aman"

        df_dashboard["Rekomendasi"] = df_dashboard.apply(rekomendasi, axis=1)

        # Tabel yang harus dibeli segera
        st.write("### 🔥 Produk yang Harus Dibeli Sekarang")
        reorder_now = df_dashboard[df_dashboard["Rekomendasi"] == "Beli Sekarang"]

        if not reorder_now.empty:
            st.table(
                reorder_now[[
                    "Id Produk", "Nama_Produk", "Jumlah Terjual", "Stok Tersisa",
                    "Prediksi_Habis_Hari", "Rekomendasi"
                ]]
            )
        else:
            st.success("Tidak ada produk yang perlu dibeli segera 🎉")

        # ============================
        #  6️⃣ FAST-MOVING vs SLOW-MOVING
        # ============================

        st.markdown("---")
        st.subheader("6️⃣ Fast-Moving vs Slow-Moving Products")

        df_dashboard["Moving_Rate"] = (
            df_dashboard["Jumlah Terjual"] / (df_dashboard["Jumlah Terjual"] + df_dashboard["Stok Tersisa"]).replace(0, 1)
        ).round(2)

        def moving_category(val):
            if val >= 0.6:
                return "Fast Moving"
            elif val >= 0.3:
                return "Medium"
            return "Slow Moving"

        df_dashboard["Moving_Category"] = df_dashboard["Moving_Rate"].apply(moving_category)

        moving_count = df_dashboard["Moving_Category"].value_counts()

        fig_mv, axmv = plt.subplots(figsize=(6, 4))
        sns.barplot(x=moving_count.index, y=moving_count.values, palette="Set2", ax=axmv)
        axmv.set_title("Distribusi Fast vs Slow Moving")
        st.pyplot(fig_mv)

        # ============================
        # 7️⃣ Stock-to-Sales Ratio (SSR)
        # ============================

        st.markdown("---")
        st.subheader("7️⃣ Stock to Sales Ratio (SSR)")

        df_dashboard["SSR"] = (
            df_dashboard["Stok Tersisa"] / df_dashboard["Jumlah Terjual"].replace(0, 0.1)
        ).round(2)

        st.line_chart(df_dashboard["SSR"])

        st.write("### Tabel SSR")
        st.dataframe(df_dashboard[["Nama_Produk", "Jumlah Terjual", "Stok Tersisa", "SSR"]])

        # ============================
        # 8️⃣ DEADSTOCK LIST
        # ============================

        st.markdown("---")
        st.subheader("8️⃣ Daftar Produk Deadstock (Tidak Laku Sama Sekali)")

        deadstock = df_dashboard[df_dashboard["Stock_Health"] == "Deadstock"]

        if not deadstock.empty:
            st.error("Produk Deadstock ditemukan. Pertimbangkan hentikan pembelian.")
            st.table(
                deadstock[["Id Produk", "Nama_Produk", "Kategori", "Stok Tersisa", "Jumlah Terjual"]]
            )
        else:
            st.success("Tidak ada produk deadstock 🎉")

        # ============================
        # 9️⃣ OVERSTOCK LIST
        # ============================

        st.markdown("---")
        st.subheader("9️⃣ Daftar Produk Overstock (Stok Berlebih)")

        overstock = df_dashboard[df_dashboard["Stock_Health"] == "Overstock"]

        if not overstock.empty:
            st.warning("Produk Overstock ditemukan. Pertimbangkan diskon atau promosi.")
            st.table(overstock[["Id Produk", "Nama_Produk", "Jumlah Terjual", "Stok Tersisa"]])
        else:
            st.success("Tidak ada produk overstock 🎉")

            
    with tabs[2]:
        st.header("📈 Analisis Tiap Cluster — Hasil Clustering")

        if "df_clustered" not in st.session_state:
            st.warning("Belum ada hasil clustering.")
            st.stop()

        df_clustered = st.session_state["df_clustered"]
        numeric_cols = st.session_state["numeric_cols"]

        st.markdown("""
        Halaman ini menampilkan analisis lengkap setiap cluster berdasarkan hasil K-Means.
        Analisis meliputi:
        - Distribusi produk per cluster  
        - Rata-rata pendapatan & jumlah terjual  
        - Profil lengkap setiap cluster  
        - Produk favorit per cluster  
        """)

        st.markdown("---")

    # ============================================================
    # 1️⃣ Distribusi Produk per Cluster
    # ============================================================

        st.subheader("1️⃣ Distribusi Produk per Cluster")

        cluster_count = df_clustered["Cluster"].value_counts().sort_index()

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.barplot(x=cluster_count.index, y=cluster_count.values, palette="viridis", ax=ax1)
        ax1.set_xlabel("Cluster")
        ax1.set_ylabel("Jumlah Produk")
        ax1.set_title("Distribusi Produk per Cluster")
        st.pyplot(fig1)

        st.table(
            cluster_count.reset_index().rename(columns={"index": "Cluster", "Cluster": "Jumlah Produk"})
        )

        st.markdown("---")

    # ============================================================
    # 2️⃣ Rata-Rata Pendapatan & Jumlah Terjual per Cluster
    # ============================================================

        st.subheader("2️⃣ Rata-Rata Pendapatan & Jumlah Terjual per Cluster")

        cluster_mean = (
            df_clustered.groupby("Cluster")[["Total Pendapatan (Rp)", "Jumlah Terjual"]]
            .mean()
            .round(2)
        )

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax3 = ax2.twinx()

        ax2.bar(
            cluster_mean.index - 0.2, cluster_mean["Total Pendapatan (Rp)"],
            width=0.4, color="#3b82f6", label="Total Pendapatan (Rp)"
        )

        ax3.bar(
            cluster_mean.index + 0.2, cluster_mean["Jumlah Terjual"],
            width=0.4, color="#10b981", label="Jumlah Terjual"
        )

        import matplotlib.ticker as ticker
        ax2.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, pos: f"Rp {x:,.0f}".replace(",", "."))
        )

        ax2.set_ylabel("Pendapatan (Rp)", color="#3b82f6")
        ax3.set_ylabel("Jumlah Terjual", color="#10b981")
        ax2.set_xlabel("Cluster")
        ax2.set_title("Rata-Rata Pendapatan & Terjual per Cluster")

        fig2.legend(loc="upper right")
        st.pyplot(fig2)

        st.table(cluster_mean)

        st.markdown("---")

    # ============================================================
    # 3️⃣ INSIGHT PER CLUSTER (UTAMA)
    # ============================================================

        st.subheader("3️⃣ Insight Lengkap Per Cluster (Profil & Statistik)")

        for cluster in sorted(df_clustered["Cluster"].unique()):
            st.markdown(f"### 📦 Cluster {cluster}")

            subset = df_clustered[df_clustered["Cluster"] == cluster]

            pendapatan_mean = subset["Total Pendapatan (Rp)"].mean()
            jumlah_terjual_mean = subset["Jumlah Terjual"].mean()
            stok_mean = subset["Stok Tersisa"].mean()

            # tampilkan tabel cluster lengkap
            st.dataframe(subset, width="stretch")

            st.markdown(f"""
            #### 🔍 Ringkasan Cluster {cluster}
            - Rata-rata pendapatan: **{format_rupiah(pendapatan_mean)}**
            - Rata-rata jumlah terjual: **{jumlah_terjual_mean:.2f} unit**
            - Rata-rata stok tersisa: **{stok_mean:.2f} unit**

            **Interpretasi:**
            - Produk dengan pendapatan tinggi → potensi fast-moving  
            - Produk dengan stok tinggi tapi terjual rendah → potensi overstock  
            - Produk dengan stok rendah & terjual tinggi → berisiko habis (baik restock)  
            """)

            st.markdown("---")

    # ============================================================
    # 4️⃣ Produk Favorit per Cluster (Top 5)
    # ============================================================

        st.subheader("4️⃣ Produk Favorit per Cluster (Top 5 Berdasarkan Penjualan)")

        if "Jumlah Terjual" in df_clustered.columns and "Nama_Produk" in df_clustered.columns:
            for cluster in sorted(df_clustered["Cluster"].unique()):
                st.markdown(f"### ⭐ Cluster {cluster}")

                subset = df_clustered[df_clustered["Cluster"] == cluster]
                top_products = subset.sort_values(
                    by="Jumlah Terjual",
                    ascending=False
                ).head(5)

                if not top_products.empty:
                    st.table(
                        top_products[[ 
                            "Id Produk", "Nama_Produk", "Kategori",
                            "Jumlah Terjual", "Total Pendapatan (Rp)", "Stok Tersisa"
                        ]]
                    )
                else:
                    st.info(f"Tidak ada produk untuk cluster {cluster}")

        st.markdown("---")

    # ============================================================
    # 5️⃣ Ringkasan Machine Learning
    # ============================================================

        st.subheader("5️⃣ Ringkasan Machine Learning")

        best_cluster = cluster_mean["Total Pendapatan (Rp)"].idxmax()

        st.info(f"""
            💡 **Cluster terbaik adalah Cluster {best_cluster}**,  
            karena memiliki pendapatan rata-rata tertinggi.

            Model K-Means memisahkan produk berdasarkan:
            - Performa penjualan  
            - Tingkat pendapatan  
            - Tingkat stok  

            Analisis ini membantu identifikasi:
            - Cluster fast-moving  
            - Cluster slow-moving  
            - Cluster risiko overstock / deadstock  
            """)
        # st.header("📈 Analisis Tiap Cluster")
        # if "df_clustered" not in st.session_state:
        #     st.warning("Belum ada hasil clustering.")
        #     st.stop()
        # df_clustered = st.session_state["df_clustered"]
        # numeric_cols = st.session_state["numeric_cols"]
        # df_with_stats, cluster_mean, cluster_size = compute_cluster_stats(df_clustered, numeric_cols)
        # st.subheader("📊 Data Produk dengan Statistik")
        # show_df_with_rupiah(df_with_stats)
        # st.subheader("📋 Rata-Rata Numerik per Cluster")
        # cluster_mean_fmt = cluster_mean.copy()
        # for col in MONEY_COLS:
        #     if col in cluster_mean_fmt.columns:
        #         cluster_mean_fmt[col] = cluster_mean_fmt[col].apply(format_rupiah)
        # st.table(cluster_mean_fmt)
        # st.subheader("📦 Jumlah Produk per Cluster")
        # cluster_size = cluster_size.rename(columns={"Cluster": "Cluster", "Jumlah Produk": "Total Produk"})
        # st.dataframe(cluster_size)
        # st.markdown("### 🧠 Insight per Cluster")
        # for cluster in sorted(df_clustered["Cluster"].unique()):
        #     subset = df_clustered[df_clustered["Cluster"] == cluster]
        #     pendapatan_mean = subset["Total Pendapatan (Rp)"].mean()
        #     jumlah_terjual_mean = subset["Jumlah Terjual"].mean()
        #     stok_mean = subset["Stok Tersisa"].mean()
        #     st.markdown(f"#### 📦 Cluster {cluster}")
        #     show_df_with_rupiah(subset)
        #     st.markdown(f"""
        #     **Ringkasan:**
        #     - Rata-rata pendapatan produk: **{format_rupiah(pendapatan_mean)}**
        #     - Rata-rata jumlah terjual: **{jumlah_terjual_mean:.2f} unit**
        #     - Rata-rata stok tersisa: **{stok_mean:.2f} unit**
        #     """)

    # ================================
    # TAB 4 — LAPORAN
    # ================================
    with tabs[3]:
        st.markdown("""
        ### 📄 Unduh Laporan Clustering
        Download laporan lengkap hasil clustering dalam format Excel.
        """)

        if "df_clustered" in st.session_state:
            df_report = st.session_state["df_clustered"]
            excel_bytes = make_excel_report(df_report)

            st.download_button(
                label="Download Excel",
                data=excel_bytes,
                file_name="Laporan_Clustering_Produk.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.warning("Belum ada hasil clustering.")

# Run App
if __name__ == "__main__":
    main()
