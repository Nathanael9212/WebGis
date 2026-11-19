# Penentuan Lokasi Rumah Pompa Air Pengendali Banjir Menggunakan Klasterisasi DBSCAN dan K-Means (Studi Kasus: Kota Surabaya)

Implementasi kode penelitian skripsi bidang Informatika (*Data Mining*, *Unsupervised Machine Learning*, *Spatial Clustering*, dan WebGIS) untuk menghasilkan **kandidat awal lokasi rumah pompa** berdasarkan persebaran spasial titik inventarisasi saluran drainase pada 7 rayon pematusan di Kota Surabaya.

> **Catatan Metodologis:** Hasil klasterisasi berupa **kandidat awal berbasis pola spasial** dan jarak terhadap rumah pompa aktif dihitung sebagai **jarak lurus Euclidean (*straight-line distance*)** dalam meter (`EPSG:32749`). Hasil ini bukan keputusan final pembangunan rumah pompa dan tetap memerlukan kajian teknis/hidraulis lanjutan oleh instansi terkait.

---

## 1. Struktur Proyek

```text
skripsi-rumah-pompa/
├── data/
│   ├── raw/                        # Dataset asli (read-only, tidak dimodifikasi)
│   │   ├── 21467_titik_saluran_clean.geojson
│   │   ├── 7_rayon_pematusan_clean.geojson
│   │   ├── 83_rumah_pompa_bab3_clean.geojson   # Acuan utama Bab III
│   │   └── 83_rumah_pompa_clean.geojson        # Arsip versi update
│   └── processed/                  # Data bersih & hasil klasterisasi (EPSG:32749, .gpkg & .csv)
├── notebooks/
│   ├── 01_eksplorasi_data.ipynb    # Tahap 1: Audit kualitas data & topologi 7 rayon
│   ├── 02_preprocessing.ipynb      # Tahap 2: Proyeksi UTM 49S, PiP ketat, deduplikasi
│   ├── 03_dbscan.ipynb             # Tahap 3: k-distance (k=4), Kneedle, DBSCAN per rayon
│   ├── 04_kmeans.ipynb             # Tahap 4: Elbow Method (K=2..10), K-Means++ pada Non-Noise
│   ├── 05_evaluasi.ipynb           # Tahap 5: Silhouette Coefficient & Davies-Bouldin Index
│   └── 06_analisis_jarak.ipynb     # Tahap 6: Jarak Euclidean ke 71 rumah pompa aktif & ekspor GeoJSON
├── src/
│   ├── preprocessing.py            # Fungsi modular Tahap 2
│   ├── clustering.py               # Fungsi modular Tahap 3 (DBSCAN) & Tahap 4 (K-Means++)
│   ├── evaluation.py               # Fungsi modular Tahap 5 (SC & DBI)
│   └── spatial_analysis.py         # Fungsi modular Tahap 6 (Analisis jarak & ekspor WebGIS)
├── outputs/
│   ├── figures/                    # Seluruh grafik & peta statis resolusi tinggi (.png)
│   ├── tables/                     # Seluruh tabel audit, ringkasan, evaluasi, & jarak (.csv)
│   └── geojson/                    # 6 layer GeoJSON siap pakai WebGIS (EPSG:4326)
├── app.py                          # Tahap 7: Prototipe WebGIS interaktif (Streamlit + Folium)
├── requirements.txt                # Daftar dependensi dengan versi eksak
└── README.md
```

---

## 2. Dataset & Ringkasan Pipeline

1. **Titik Saluran Drainase (`21.467` awal → `21.272` bersih siap klaster)**:
   - Normalisasi 3 typo `kode_hierarki` (`Sa`, `Sl` → `SL`).
   - Transformasi CRS seragam `EPSG:4326` → **UTM Zona 49S (`EPSG:32749`)** menghasilkan `easting` (`X1`) dan `northing` (`X2`) dalam meter.
   - *Point-in-Polygon* (PiP) ketat dan deterministik (`RYN-01` s.d. `RYN-07` pada 87 titik zona *overlap*; mengeluarkan 65 titik luar poligon ke file audit).
   - Deduplikasi 130 baris koordinat identik (103 Tipe A + 27 Tipe B) agar tidak ada jarak 0 meter pada *k-distance*/DBSCAN.
2. **DBSCAN per Rayon (`MinPts = 4`, $\varepsilon$ adaptif Kneedle `S = 1.0`)**:
   - Rentang $\varepsilon$ adaptif: `144,28 m` (Genteng) s.d. `229,85 m` (Tandes).
   - Hasil: **20.929 Core (98,39%)**, **179 Border (0,84%)** → **21.108 Non-Noise (99,23%)**, dan **164 Noise (0,77%)** (tetap disimpan).
3. **K-Means++ per Rayon (Elbow Method `K = 2..10`, `n_init = 20`, `random_state = 42`)**:
   - Dilatih pada gabungan seluruh titik **Non-Noise (Core + Border)** per rayon.
   - Menghasilkan **31 centroid kandidat awal** (`K=5` pada Benowo, Wiyung, Jambangan; `K=4` pada Tandes, Rungkut, Genteng, Gubeng).
4. **Evaluasi Clustering (SC & DBI)**:
   - Rata-rata 7 rayon (*unweighted*): **Silhouette Coefficient = `0,4627`** (`0,3974`–`0,5273`), **DBI = `0,7436`** (`0,5958`–`0,8944`).
5. **Analisis Jarak ke 71 Rumah Pompa Operasional Aktif (`EPSG:32749`)**:
   - Rentang jarak lurus Euclidean: **`155,00 m`** s.d. **`6.408,19 m`** (median `1.077,74 m`, rata-rata `1.583,06 m`; 5 centroid lintas rayon, 0 *tie* koordinat).
   - Sebanyak **12 lokasi Pengembangan Baru** disimpan dan ditampilkan terpisah sebagai konteks pendukung.

---

## 3. Cara Menjalankan Proyek (Windows PowerShell)

Lingkungan proyek menggunakan virtual environment `.venv` berbasis **Python 3.12** (dikelola via `uv`).

### A. Instalasi Ulang Dependensi (jika di komputer baru)
```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name skripsi-rumah-pompa --display-name "Python 3.12 (Skripsi Rumah Pompa)"
```

### B. Menjalankan Prototipe WebGIS (Streamlit + Folium)
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Buka alamat lokal yang ditampilkan di terminal (default `http://localhost:8501`).

### C. Menjalankan Notebook Secara Interaktif (JupyterLab / VS Code)
- Di **VS Code**: buka file `notebooks/01_eksplorasi_data.ipynb` s.d. `06_analisis_jarak.ipynb`, lalu pilih kernel **`Python 3.12 (Skripsi Rumah Pompa)`**.
- Atau via terminal:
```powershell
.venv\Scripts\python.exe -m jupyterlab
```
