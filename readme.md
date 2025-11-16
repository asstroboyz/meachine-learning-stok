# 🧶 Salma Company – Machine Learning Product Clustering  
Presentational README • Business Insight Edition • Full Code Explanation

---

# 🎯 1. Overview

Aplikasi ini adalah dashboard **Machine Learning berbasis Streamlit** yang dirancang untuk:

- 📊 Analisis Cluster Produk  
- 🤖 K-Means Clustering untuk segmentasi penjualan  
- 📈 Visualisasi & Insight otomatis  
- 📦 Rekap laporan dalam bentuk Excel  
- 🎨 Dashboard interaktif & responsif  

Aplikasi ini digunakan oleh **Salma Company** untuk meningkatkan efisiensi stok dan strategi penjualan.

---

# 📊 2. Fitur Utama

### ✔ 2.1 K-Means Clustering  
Mengelompokkan produk berdasarkan:

- Jumlah Terjual  
- Total Pendapatan  
- Stok Tersisa

### ✔ 2.2 Elbow Method  
Menentukan jumlah cluster optimal melalui perhitungan SSE.

### ✔ 2.3 Davies-Bouldin Index  
Menentukan kualitas cluster — semakin kecil semakin baik.

### ✔ 2.4 Dashboard Premium  
Tersedia card-card metrik:

- Total produk  
- Jumlah cluster  
- DBI Score  
- Cluster terbaik  

### ✔ 2.5 Insight Otomatis  
Berbasis machine learning.

### ✔ 2.6 Export Excel  
Download hasil clustering dalam format `.xlsx`.

---

# 🧠 3. Struktur File & Penjelasan Kode

```
/
├── app.py               # Main Streamlit App
├── requirements.txt     # Dependency Python
├── turtorial.txt        # Catatan internal
└── README.md            # Dokumentasi project
```

---

# 🧩 4. Penjelasan Kode Penting

## 4.1 safe_numeric_convert()
Membersihkan data menjadi numerik.

```python
def safe_numeric_convert(df):
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col].astype(str).replace(r"[^\d]", "", regex=True), errors="coerce")
    return df
```

## 4.2 run_clustering()
Menjalankan K-Means lengkap dengan scaling data:

```python
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(X_scaled)
```

## 4.3 run_elbow()
Menghitung SSE dari K=1 hingga max:

```python
sse = []
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k).fit(X_scaled)
    sse.append(kmeans.inertia_)
```

## 4.4 Export Excel
```python
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df_clustered.to_excel(writer, index=False)
```

---

# 🖥 5. Tutorial Menjalankan Aplikasi

### **Step 1 : Buat Virtual Environment**
```
py -m venv venv
```

### **Step 2 : Install Requirements**
```
pip install -r requirements.txt
```

### **Step 3 : Izinkan Script PowerShell**
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Step 4 : Aktifkan Virtual Environment**
```
.env\Scripts\Activate.ps1
```

### **Step 5 : Jalankan Aplikasi**
```
streamlit run app.py
```

---

# 🚀 6. Teknologi

- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

# 👑 7. Author  
Dibuat oleh **Lord of Code** 👑  
_Mengubah data menjadi keputusan bisnis._

