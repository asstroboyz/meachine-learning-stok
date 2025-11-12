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
    """Hitung statistik rata-rata dan jumlah pelanggan per cluster"""
    cluster_mean = df_clustered.groupby("Cluster")[numeric_cols].mean().round(2)
    cluster_size = df_clustered["Cluster"].value_counts().sort_index().reset_index()
    cluster_size.columns = ["Cluster", "Jumlah Pelanggan"]

    df_with_stats = df_clustered.copy()
    for col in numeric_cols:
        df_with_stats[f"{col} (Cluster Mean)"] = df_with_stats["Cluster"].map(cluster_mean[col])

    return df_with_stats, cluster_mean, cluster_size


# ============== MAIN APP ==============
def main():
    st.set_page_config(page_title="Customer Clustering - Toko Salma Company", layout="wide")
    st.title("🧮 Salma Company — Customer Clustering")

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

            # ✅ Ganti key slider agar tidak bentrok dengan session_state
            n_clusters = st.slider(
                "Pilih jumlah cluster (k)",
                2, 10,
                value=st.session_state["n_clusters"],
                key="slider_n_clusters"
            )
            st.session_state["n_clusters"] = n_clusters

            try:
                df_clustered, kmeans, X_scaled, numeric_cols = run_clustering(df, n_clusters)
                st.session_state.update({
                    "df_clustered": df_clustered,
                    "numeric_cols": numeric_cols,
                    "X_scaled": X_scaled,
                })
                st.success("✅ Clustering selesai — Dashboard & Analisis otomatis diperbarui.")
            except Exception as e:
                st.error(f"Gagal melakukan clustering: {e}")
                st.stop()

            st.subheader("📈 Elbow Method")
            sse = run_elbow(df)
            fig, ax = plt.subplots()
            ax.plot(range(1, len(sse) + 1), sse, marker="o")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("SSE")
            ax.set_title("Grafik Elbow Method")
            st.pyplot(fig)

            st.markdown("### 🧮 Tabel Nilai SSE (Sum of Squared Errors)")
            sse_table = pd.DataFrame({"Jumlah Cluster": range(1, len(sse) + 1), "SSE": sse})
            st.table(sse_table)

            df_clustered = st.session_state["df_clustered"]
            st.subheader("📉 Nilai Davies-Bouldin Index (DBI)")
            dbi = davies_bouldin_score(X_scaled, df_clustered["Cluster"])
            st.info(f"DBI = `{dbi:.3f}` (semakin kecil semakin baik)")

            st.write("### 📋 Tabel Hasil Clustering per Cluster:")
            for cluster in sorted(df_clustered['Cluster'].unique()):
                st.write(f"#### Cluster {cluster}")
                df_sub = df_clustered[df_clustered['Cluster'] == cluster]
                show_df_with_rupiah(df_sub)

            st.markdown("### 📊 Summary Jumlah Data per Cluster")
            st.dataframe(
                df_clustered['Cluster'].value_counts().sort_index().reset_index().rename(
                    columns={'index': 'Cluster', 'Cluster': 'Jumlah Data'}
                ), width="stretch"
            )

    # =========== TAB 1: DASHBOARD ===========
  
    with tabs[0]:
        # st.header("📊 Dashboard Business Intelligence — Analisis Pelanggan")

        if "df_clustered" in st.session_state:
            df_dashboard = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]
            X_scaled = st.session_state["X_scaled"]

            # ==============================
            # 🎯 HEADER MACHINE LEARNING INSIGHT
            # ==============================
            
              # ==============================
            total_pelanggan = len(df_dashboard)
            total_cluster = df_dashboard["Cluster"].nunique()
            dbi_score = round(davies_bouldin_score(X_scaled, df_dashboard["Cluster"]), 3)
            best_cluster = df_dashboard.groupby("Cluster")["Total Spent"].mean().idxmax()
            worst_cluster = df_dashboard.groupby("Cluster")["Total Spent"].mean().idxmin()

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div style='background:linear-gradient(145deg, #1E3A8A, #1D4ED8); 
                            padding:18px; border-radius:12px; color:white; text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
                    <h4 style='margin:0;'>👥</h4>
                    <p style='margin:0; font-size:14px; opacity:0.8;'>Total Pelanggan</p>
                    <h2 style='margin:4px 0 0 0;'>{total_pelanggan:,}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style='background:linear-gradient(145deg, #0E7490, #06B6D4);
                            padding:18px; border-radius:12px; color:white; text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
                    <h4 style='margin:0;'>📦</h4>
                    <p style='margin:0; font-size:14px; opacity:0.8;'>Jumlah Cluster</p>
                    <h2 style='margin:4px 0 0 0;'>{total_cluster}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style='background:linear-gradient(145deg, #15803D, #22C55E);
                            padding:18px; border-radius:12px; color:white; text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
                    <h4 style='margin:0;'>📉</h4>
                    <p style='margin:0; font-size:14px; opacity:0.8;'>DBI Score</p>
                    <h2 style='margin:4px 0 0 0;'>{dbi_score}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div style='background:linear-gradient(145deg, #7E22CE, #9333EA);
                            padding:18px; border-radius:12px; color:white; text-align:center;
                            box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
                    <h4 style='margin:0;'>💎</h4>
                    <p style='margin:0; font-size:14px; opacity:0.8;'>Cluster Terbaik</p>
                    <h2 style='margin:4px 0 0 0;'>Cluster {best_cluster}</h2>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("""
            <div style='padding:20px; border-radius:10px; background:linear-gradient(90deg, #0F172A, #1E293B); color:#F8FAFC;'>
                <h3 style='margin-bottom:5px;'>🤖 <b>Machine Learning Insight – Customer Segmentation</b></h3>
                <p style='margin-top:6px;'>
                    Analisis ini menggunakan pendekatan <b>unsupervised learning</b> dengan algoritma 
                    <b>K-Means Clustering</b> untuk mengelompokkan pelanggan berdasarkan <b>pola pembelian, nilai transaksi,</b> 
                    dan <b>frekuensi belanja</b>. 
                    Setiap cluster merepresentasikan segmen pelanggan dengan karakteristik dan daya beli yang berbeda.
                </p>
                <p style='margin-top:4px;'>
                    Tujuan utama dari segmentasi ini adalah untuk:
                    <ul style='margin-top:6px;'>
                        <li>🎯 <b>Mengidentifikasi pelanggan potensial</b> dengan kontribusi penjualan tertinggi.</li>
                        <li>📦 <b>Menentukan prioritas stok & efisiensi alokasi dana</b> berdasarkan daya beli tiap cluster.</li>
                        <li>📈 <b>Mendukung keputusan promosi</b> dan strategi retensi pelanggan secara terarah.</li>
                    </ul>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ==============================
            # 📊 KPI Kunci dari Hasil Clustering
          

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
            for i, v in enumerate(cluster_count.values):
                ax1.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=10)
            st.pyplot(fig1)

            # ==============================
            # 2️⃣ RATA-RATA NILAI TRANSAKSI DAN PENGELUARAN
            # ==============================
            st.subheader("2️⃣ Nilai Transaksi & Pengeluaran Rata-Rata per Cluster")
            cluster_mean = df_dashboard.groupby("Cluster")[["Total Spent", "Avg Order Value"]].mean().round(2)
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            cluster_mean.plot(kind="bar", ax=ax2, color=["#3b82f6", "#10b981"])
            ax2.set_title("Rata-Rata Nilai Transaksi dan Pengeluaran per Cluster")
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Rata-rata (Rp)")
            ax2.legend(["Total Spent", "Avg Order Value"])
            st.pyplot(fig2)

            # ==============================
            # 3️⃣ PRODUK FAVORIT PER CLUSTER
            # ==============================
            if "Produk Favorit" in df_dashboard.columns:
                st.subheader("3️⃣ Produk Favorit per Cluster")
                for cluster in sorted(df_dashboard["Cluster"].unique()):
                    st.markdown(f"#### Cluster {cluster}")
                    top_product = (
                        df_dashboard[df_dashboard["Cluster"] == cluster]["Produk Favorit"]
                        .dropna()
                        .value_counts()
                        .head(5)
                    )
                    if not top_product.empty:
                        st.table(
                            top_product.reset_index().rename(
                                columns={"index": "Produk", "Produk Favorit": "Jumlah Pembelian"}
                            )
                        )
                    else:
                        st.info("Tidak ada data produk favorit di cluster ini.")

    
    # =========== TAB 3: ANALISIS ===========
    with tabs[2]:
        st.header("📈 Analisis Tiap Cluster")
        if "df_clustered" not in st.session_state:
            st.info("Belum ada hasil clustering.")
        else:
            df_clustered = st.session_state["df_clustered"]
            numeric_cols = st.session_state["numeric_cols"]
            n_clusters = st.session_state["n_clusters"]

            st.info(f"Jumlah cluster aktif: {n_clusters}")

            df_with_stats, cluster_mean, cluster_size = compute_cluster_stats(df_clustered, numeric_cols)
            st.subheader("📊 Data Pelanggan dengan Statistik Cluster")
            show_df_with_rupiah(df_with_stats)

            cluster_mean_fmt = cluster_mean.copy()
            for col in MONEY_COLS:
                if col in cluster_mean_fmt.columns:
                    cluster_mean_fmt[col] = cluster_mean_fmt[col].apply(format_rupiah)
            st.subheader("📋 Statistik Tiap Cluster (Rata-rata Fitur)")
            st.table(cluster_mean_fmt)

            st.subheader("👥 Jumlah Pelanggan per Cluster")
            
        # ========== ANALISIS OTOMATIS SETIAP CLUSTER ==========
        st.markdown("### 🧠 Analisis & Interpretasi Otomatis")

        for cluster in sorted(df_clustered["Cluster"].unique()):
            subset = df_clustered[df_clustered["Cluster"] == cluster]
            total = len(subset)
            spent_mean = subset["Total Spent"].mean()
            order_mean = subset["Avg Order Value"].mean()
            recency_mean = subset["Recency Days"].mean() if "Recency Days" in subset.columns else None
            top_product = subset["Produk Favorit"].mode()[0] if "Produk Favorit" in subset.columns and not subset["Produk Favorit"].dropna().empty else "Tidak ada"

            
            st.markdown(f"#### 📋 Data Pelanggan di Cluster {cluster}")
            show_df_with_rupiah(subset)

            # ========= Narasi analisis per cluster =========
            analysis_text = f"""
            ### 📦 Cluster {cluster}
            - Jumlah pelanggan: **{total} orang**
            - Rata-rata pengeluaran total: **{format_rupiah(spent_mean)}**
            - Rata-rata nilai transaksi: **{format_rupiah(order_mean)}**
            - Rata-rata recency (hari sejak transaksi terakhir): **{round(recency_mean, 1) if recency_mean else 'Data tidak tersedia'}**
            - Produk yang paling sering dibeli: **{top_product}**

            **Analisis & Rekomendasi:**
            Hasil clustering menunjukkan bahwa pelanggan dalam cluster ini memiliki rata-rata pengeluaran yang **{'lebih tinggi dari rata-rata keseluruhan' if spent_mean > df_clustered['Total Spent'].mean() else 'lebih rendah dari rata-rata keseluruhan'}**,
            dan kecenderungan transaksi yang **{'lebih sering' if order_mean > df_clustered['Avg Order Value'].mean() else 'lebih jarang'}** dibandingkan dengan segmen lain.
            Pelanggan dalam kelompok ini **{'termasuk segmen bernilai tinggi yang berkontribusi besar terhadap pendapatan toko' if spent_mean > df_clustered['Total Spent'].mean() else 'tergolong segmen dengan daya beli rendah yang perlu diaktifkan kembali'}**.

            Untuk pengambilan keputusan stok:
            - Produk **{top_product}** direkomendasikan untuk **{'dipertahankan dan ditambah stok minimal ' + str(max(5, int(total * 1.5))) + ' pcs' if spent_mean > df_clustered['Total Spent'].mean() else 'dievaluasi ketersediaannya agar tidak overstock.'}**
            - Jumlah pelanggan di cluster ini (**{total} orang**) dapat dijadikan acuan estimasi permintaan bulanan untuk produk favorit tersebut.
            - Strategi promosi yang disarankan: **{'program loyalitas dan rekomendasi produk sejenis' if spent_mean > df_clustered['Total Spent'].mean() else 'penawaran promo, diskon, atau bundling untuk meningkatkan aktivitas pembelian.'}**
            """
            st.markdown(analysis_text)

            
       
        st.markdown("##  Kesimpulan")

        df_all = df_clustered.copy()
        cluster_summary = df_all.groupby("Cluster").agg({
            "Total Spent": "mean",
            "Avg Order Value": "mean",
            "Recency Days": "mean" if "Recency Days" in df_all.columns else "median"
        }).reset_index()

        best_cluster = cluster_summary.loc[cluster_summary["Total Spent"].idxmax()]
        worst_cluster = cluster_summary.loc[cluster_summary["Total Spent"].idxmin()]

        if "Produk Favorit" in df_all.columns:
            product_counts = df_all["Produk Favorit"].value_counts().head(5)
            fav_all = product_counts.index[0]
            fav_count = product_counts.iloc[0]
        else:
            product_counts = None
            fav_all = "Tidak ada data"
            fav_count = 0

        # === RINGKASAN HASIL CLUSTERING ===
        summary_text = f"""
        **🔍 Ringkasan Hasil Clustering**
        - Jumlah cluster terbentuk: **{df_all["Cluster"].nunique()}**
        - Produk paling diminati: **{fav_all}** (**{fav_count} pelanggan**)
        - Cluster terbaik: **Cluster {int(best_cluster["Cluster"])}** 
        (Rata-rata pengeluaran: {format_rupiah(best_cluster["Total Spent"])})
        - Cluster terendah: **Cluster {int(worst_cluster["Cluster"])}**
        (Rata-rata pengeluaran: {format_rupiah(worst_cluster["Total Spent"])})
        """
        st.markdown(summary_text)

        if product_counts is not None:
            st.markdown("### 🛒 5 Produk Paling Diminati")
            df_top5 = pd.DataFrame({
                "Produk": product_counts.index,
                "Jumlah Pelanggan": product_counts.values
            })
            st.table(df_top5)

        # === REKOMENDASI RINGKAS & STRATEGIS ===
        recommend_text = f"""
        **💼 Rekomendasi Pengambilan Keputusan**

        1. **Pengadaan Stok**
        - Fokuskan dana pada produk **{fav_all}** (paling banyak diminati).
        - Tambahkan stok sekitar **{fav_count * 2} pcs** untuk periode berikutnya.
        - Batasi pembelian produk dengan penjualan rendah dari cluster {int(worst_cluster["Cluster"])}.

        2. **Efisiensi Dana**
        - Alokasikan **60–70% dana** ke cluster bernilai tinggi (**Cluster {int(best_cluster["Cluster"])}**) untuk menjamin perputaran modal cepat.
        - Gunakan sisa dana untuk promo atau uji pasar produk baru.

        3. **Promosi & Retensi**
        - Terapkan **program loyalitas** untuk pelanggan di cluster tinggi.
        - Berikan **promo musiman/bundling hemat** untuk pelanggan cluster rendah guna mempercepat rotasi stok.

        4. **Kesimpulan Akhir**
        - Berdasarkan hasil **KMeans** dan nilai **DBI**, segmentasi pelanggan sudah terbagi jelas.
        - Produk **{fav_all}** menjadi indikator utama permintaan pasar.
        - Implementasi strategi ini akan:
            - Meningkatkan efisiensi dana pengadaan,
            - Mengurangi risiko kelebihan stok,
            - Dan mengoptimalkan keuntungan berdasarkan nilai pelanggan tiap cluster.
        """
        st.markdown(recommend_text)


    # =========== TAB 4: LAPORAN ===========
    with tabs[3]:
            st.markdown("""
            <div style='padding:20px; background:linear-gradient(90deg, #0F172A, #1E293B); border-radius:12px; margin-bottom:20px; color:#F8FAFC;'>
                <h2 style='margin-bottom:8px;'>📥 <b>Laporan Hasil Clustering</b></h2>
                <p style='font-size:15px; margin-top:4px;'>
                    Halaman ini menyajikan laporan lengkap hasil segmentasi pelanggan menggunakan 
                    <b>Machine Learning (K-Means Clustering)</b>. 
                    Laporan dapat diunduh dalam format <b>Excel (.xlsx)</b> untuk keperluan dokumentasi, analisis lanjutan, atau pelaporan akademik.
                </p>
                <p style='font-size:14px; margin-top:8px; opacity:0.85;'>
                    File berisi data lengkap setiap pelanggan, cluster hasil analisis, dan nilai fitur penting seperti 
                    <i>Total Spent</i>, <i>Avg Order Value</i>, serta <i>Produk Favorit</i>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            if "df_clustered" in st.session_state:
                df_report = st.session_state["df_clustered"]
                excel_bytes = make_excel_report(df_report)

       
                # ===========================
                # 📥 Tombol Download Custom
                # ===========================
                st.markdown("<div style='text-align:left; margin-top:20px;'>", unsafe_allow_html=True)
                st.download_button(
                    label="⬇️ Unduh Laporan Excel",
                    data=excel_bytes,
                    file_name="Laporan_Clustering_Pelanggan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # ✅ Pesan sukses
                # st.success("✅ Laporan siap diunduh. Simpan file ini untuk analisis lanjutan atau dokumentasi penelitian.")

            else:
                st.warning("⚠️ Belum ada hasil clustering. Silakan jalankan proses terlebih dahulu di tab **Proses Clustering**.")



if __name__ == "__main__":
    main()
