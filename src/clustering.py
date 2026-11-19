"""
Modul Tahap 3 & 4 — Klasterisasi Spasial (DBSCAN dan K-Means++)
Penelitian: Penentuan Lokasi Rumah Pompa Air Pengendali Banjir Menggunakan
Klasterisasi DBSCAN dan K-Means (Studi Kasus: Kota Surabaya)

Berisi:
- Tahap 3 (DBSCAN): k-distance (k=4, konvensi Scikit-learn), deteksi epsilon adaptif
  Kneedle, klasifikasi Core/Border/Noise per rayon.
- Tahap 4 (K-Means++): Elbow Method (K = 2..10) dengan Kneedle pada gabungan seluruh
  titik Non-Noise (Core + Border) per rayon, pembentukan subcluster, dan ekstraksi
  centroid kandidat awal (UTM 49S dan proyeksi balik WGS84).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from kneed import KneeLocator


DEFAULT_MIN_PTS = 4
DEFAULT_K_RANGE = tuple(range(2, 11))  # K = 2 sampai 10
DEFAULT_KMEANS_N_INIT = 20
DEFAULT_RANDOM_STATE = 42


def compute_k_distances(
    coords_xy: np.ndarray,
    k: int = DEFAULT_MIN_PTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghitung k-distance Euclidean (dalam meter) pada koordinat proyeksi (easting, northing).

    Konvensi Scikit-learn:
    - `DBSCAN(min_samples=k)` menghitung titik acuan itu sendiri sebagai bagian dari `k` titik
      di dalam radius epsilon (`d(p, q) <= eps`).
    - Oleh karena itu, pada `NearestNeighbors(n_neighbors=k)`, kolom indeks `0` adalah jarak ke
      diri sendiri (`0.0 m`) dan kolom indeks `k - 1` (untuk k=4 adalah indeks `[3]`) adalah
      jarak ke titik ke-4 termasuk diri sendiri (yaitu tetangga terdekat ke-3 selain diri sendiri).
    - Titik `p` menjadi Core point jika dan hanya jika `k_distance[p] <= eps`.

    Returns:
    - k_dist_raw: array jarak ke-k sesuai urutan titik asli (panjang N).
    - k_dist_sorted: array jarak ke-k yang sudah diurutkan menaik (panjang N).
    """
    coords_xy = np.asarray(coords_xy, dtype=float)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError("coords_xy harus berupa array 2D berbentuk (N, 2) [easting, northing].")
    if len(coords_xy) < k:
        raise ValueError(f"Jumlah titik ({len(coords_xy)}) lebih kecil dari k ({k}).")

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(coords_xy)
    distances, _ = nn.kneighbors(coords_xy)

    # Indeks k - 1 adalah jarak ke titik ke-k (termasuk titik itu sendiri di indeks 0)
    k_dist_raw = distances[:, k - 1]
    k_dist_sorted = np.sort(k_dist_raw)
    return k_dist_raw, k_dist_sorted


def detect_epsilon_kneedle(
    k_dist_sorted: np.ndarray,
    sensitivity: float = 1.0,
    curve: str = "convex",
    direction: str = "increasing",
    online: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Mendeteksi titik knee pada kurva k-distance terurut menggunakan `kneed.KneeLocator`.

    Parameters:
    - k_dist_sorted: array jarak k-distance terurut menaik (meter).
    - sensitivity (S): parameter sensitivitas Kneedle (standar = 1.0).
    - curve: bentuk kelengkungan kurva k-distance ("convex").
    - direction: arah kurva terurut menaik ("increasing").
    - online: False untuk analisis offline pada kurva lengkap.

    Returns:
    - Dictionary berisi `knee_index`, `eps_m`, `knee_percentile`, `kneedle_found`,
      serta array `x_norm` & `y_diff` untuk kebutuhan diagnostik kurva Kneedle.
    """
    n = len(k_dist_sorted)
    x = np.arange(n)

    kl = KneeLocator(
        x=x,
        y=k_dist_sorted,
        S=sensitivity,
        curve=curve,
        direction=direction,
        online=online,
    )

    if kl.knee is None:
        return {
            "kneedle_found": False,
            "knee_index": None,
            "eps_m": None,
            "knee_percentile": None,
            "all_knees": [],
            "x_norm": kl.x_normalized,
            "y_diff": kl.y_difference,
        }

    knee_idx = int(kl.knee)
    eps_val = float(k_dist_sorted[knee_idx])
    knee_pct = float(knee_idx / (n - 1) * 100.0) if n > 1 else 100.0

    return {
        "kneedle_found": True,
        "knee_index": knee_idx,
        "eps_m": eps_val,
        "knee_percentile": knee_pct,
        "all_knees": sorted(int(k) for k in kl.all_knees),
        "x_norm": kl.x_normalized,
        "y_diff": kl.y_difference,
    }


def run_dbscan_single_rayon(
    coords_xy: np.ndarray,
    eps_m: float,
    min_samples: int = DEFAULT_MIN_PTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menjalankan DBSCAN Euclidean pada 1 rayon dan mengembalikan:
    - labels: array ID cluster (`-1` = Noise, `0..C-1` = klaster DBSCAN).
    - roles: array string kategori tiap titik (`Core`, `Border`, atau `Noise`).
    """
    if eps_m is None or eps_m <= 0:
        raise ValueError(f"Nilai epsilon harus positif, diterima: {eps_m}")

    coords_xy = np.asarray(coords_xy, dtype=float)
    model = DBSCAN(eps=float(eps_m), min_samples=int(min_samples), metric="euclidean")
    labels = model.fit_predict(coords_xy)

    is_core = np.zeros(len(coords_xy), dtype=bool)
    if len(model.core_sample_indices_) > 0:
        is_core[model.core_sample_indices_] = True

    roles = np.where(
        labels == -1,
        "Noise",
        np.where(is_core, "Core", "Border"),
    )

    # Validasi konsistensi internal DBSCAN
    if np.any((labels == -1) & is_core):
        raise AssertionError("Inkonsistensi internal: titik Noise terdeteksi sebagai Core.")

    return labels, roles


def diagnose_dbscan_result(
    kneedle_found: bool,
    noise_pct: float,
    n_clusters: int,
    knee_percentile: Optional[float],
) -> str:
    """
    Menghasilkan catatan diagnostik otomatis untuk membantu evaluasi kewajaran hasil Kneedle & DBSCAN.
    """
    flags: List[str] = []
    if not kneedle_found:
        return "PERLU EVALUASI: Kneedle tidak menemukan titik knee."
    if noise_pct > 15.0:
        flags.append(f"Noise tinggi ({noise_pct:.2f}% > 15%)")
    elif noise_pct < 0.5:
        flags.append(f"Noise sangat rendah ({noise_pct:.2f}% < 0.5%)")
    if knee_percentile is not None and knee_percentile < 80.0:
        flags.append(f"Titik knee di persentil rendah (P{knee_percentile:.1f})")
    if n_clusters == 0:
        flags.append("Tidak ada klaster terbentuk (semua Noise)")
    elif n_clusters == 1:
        flags.append("Hanya 1 klaster besar terbentuk")

    return "Wajar" if not flags else "PERLU CEK: " + "; ".join(flags)


def run_dbscan_all_rayons(
    gdf_saluran_clean: gpd.GeoDataFrame,
    min_samples: int = DEFAULT_MIN_PTS,
    sensitivity: float = 1.0,
    eps_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Menjalankan alur lengkap Tahap 3 (k-distance -> Kneedle -> DBSCAN -> Core/Border/Noise)
    secara terpisah untuk masing-masing 7 rayon pematusan.

    Parameters:
    - gdf_saluran_clean: GeoDataFrame hasil Tahap 2 (CRS EPSG:32749, memiliki `id_rayon`,
      `nama_rayon`, `easting`, `northing`).
    - min_samples: parameter MinPts (default = 4).
    - sensitivity: parameter S pada KneeLocator (default = 1.0).
    - eps_overrides: opsional dict `{nama_rayon: eps_m}` jika suatu rayon diputuskan
      menggunakan nilai epsilon alternatif setelah evaluasi. Default None (100% Kneedle).

    Returns:
    - gdf_result: GeoDataFrame lengkap (seluruh titik termasuk Noise, tidak ada yang dihapus)
      dengan kolom tambahan `k_distance_m`, `dbscan_eps_m`, `dbscan_cluster`,
      `dbscan_role`, `is_non_noise`.
    - df_summary: Tabel ringkasan hasil DBSCAN per rayon (+ baris TOTAL).
    - kdist_artifacts: Dictionary per rayon berisi `k_dist_sorted` & detail Kneedle untuk plotting.
    """
    if gdf_saluran_clean.crs is None or gdf_saluran_clean.crs.to_epsg() != 32749:
        raise ValueError(f"Input harus ber-CRS EPSG:32749, ditemukan: {gdf_saluran_clean.crs}")

    eps_overrides = eps_overrides or {}
    gdf_parts: List[gpd.GeoDataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    kdist_artifacts: Dict[str, Dict[str, object]] = {}

    rayon_order = (
        gdf_saluran_clean[["id_rayon", "nama_rayon"]]
        .drop_duplicates()
        .sort_values("id_rayon")
        .itertuples(index=False)
    )

    for rid, rname in rayon_order:
        sub = gdf_saluran_clean[gdf_saluran_clean["nama_rayon"] == rname].copy()
        coords = sub[["easting", "northing"]].to_numpy(dtype=float)

        # 1. Hitung k-distance (k = min_samples, termasuk diri sendiri)
        k_raw, k_sorted = compute_k_distances(coords, k=min_samples)

        # 2. Deteksi titik knee dengan Kneedle
        knee_info = detect_epsilon_kneedle(
            k_sorted,
            sensitivity=sensitivity,
            curve="convex",
            direction="increasing",
            online=False,
        )

        # Tentukan epsilon yang digunakan
        if rname in eps_overrides:
            eps_used = float(eps_overrides[rname])
            eps_source = "Override (Evaluasi)"
        else:
            eps_used = knee_info["eps_m"]
            eps_source = "Kneedle (S=1.0)"

        # 3. Jalankan DBSCAN pada rayon ini
        if eps_used is not None:
            labels, roles = run_dbscan_single_rayon(coords, eps_m=eps_used, min_samples=min_samples)
        else:
            labels = np.full(len(sub), -1, dtype=int)
            roles = np.full(len(sub), "Noise", dtype=object)

        sub["k_distance_m"] = np.round(k_raw, 4)
        sub["dbscan_eps_m"] = round(float(eps_used), 4) if eps_used is not None else np.nan
        sub["dbscan_cluster"] = labels.astype(int)
        sub["dbscan_role"] = roles
        sub["is_non_noise"] = sub["dbscan_role"] != "Noise"
        gdf_parts.append(sub)

        # 4. Hitung statistik ringkasan per rayon
        n_total = len(sub)
        n_core = int((roles == "Core").sum())
        n_border = int((roles == "Border").sum())
        n_noise = int((roles == "Noise").sum())
        n_non_noise = n_core + n_border
        unique_clusters = sorted(int(c) for c in np.unique(labels) if c != -1)
        n_clusters = len(unique_clusters)

        largest_cluster_n = 0
        if n_clusters > 0:
            counts = pd.Series(labels[labels != -1]).value_counts()
            largest_cluster_n = int(counts.iloc[0])

        noise_pct = n_noise / n_total * 100.0
        diag = diagnose_dbscan_result(
            kneedle_found=bool(knee_info["kneedle_found"]),
            noise_pct=noise_pct,
            n_clusters=n_clusters,
            knee_percentile=knee_info["knee_percentile"],
        )

        summary_rows.append({
            "id_rayon": rid,
            "nama_rayon": rname,
            "jumlah_titik": n_total,
            "kdist_median_m": round(float(np.median(k_sorted)), 2),
            "kdist_p90_m": round(float(np.percentile(k_sorted, 90)), 2),
            "kdist_p95_m": round(float(np.percentile(k_sorted, 95)), 2),
            "kdist_max_m": round(float(k_sorted[-1]), 2),
            "knee_index": knee_info["knee_index"],
            "knee_percentile": round(float(knee_info["knee_percentile"]), 2) if knee_info["knee_percentile"] is not None else np.nan,
            "eps_kneedle_m": round(float(knee_info["eps_m"]), 2) if knee_info["eps_m"] is not None else np.nan,
            "eps_digunakan_m": round(float(eps_used), 2) if eps_used is not None else np.nan,
            "sumber_eps": eps_source,
            "jumlah_klaster_dbscan": n_clusters,
            "klaster_terbesar_titik": largest_cluster_n,
            "jumlah_core": n_core,
            "persen_core": round(n_core / n_total * 100.0, 2),
            "jumlah_border": n_border,
            "persen_border": round(n_border / n_total * 100.0, 2),
            "jumlah_non_noise": n_non_noise,
            "persen_non_noise": round(n_non_noise / n_total * 100.0, 2),
            "jumlah_noise": n_noise,
            "persen_noise": round(noise_pct, 2),
            "status_diagnostik": diag,
        })

        kdist_artifacts[rname] = {
            "id_rayon": rid,
            "k_dist_sorted": k_sorted,
            "knee_info": knee_info,
            "eps_used_m": eps_used,
            "n_total": n_total,
            "n_clusters": n_clusters,
            "noise_pct": noise_pct,
        }

    gdf_result = gpd.GeoDataFrame(
        pd.concat(gdf_parts, ignore_index=True),
        crs=gdf_saluran_clean.crs,
    ).sort_values("no").reset_index(drop=True)

    # Validasi: tidak ada satu pun titik yang hilang
    if len(gdf_result) != len(gdf_saluran_clean):
        raise AssertionError("Jumlah titik output DBSCAN tidak sama dengan input bersih!")

    df_summary = pd.DataFrame(summary_rows)

    # Tambahkan baris agregat TOTAL (untuk kolom jumlah dan persentase keseluruhan)
    tot_n = int(df_summary["jumlah_titik"].sum())
    tot_core = int(df_summary["jumlah_core"].sum())
    tot_border = int(df_summary["jumlah_border"].sum())
    tot_non_noise = int(df_summary["jumlah_non_noise"].sum())
    tot_noise = int(df_summary["jumlah_noise"].sum())
    total_row = {
        "id_rayon": "TOTAL",
        "nama_rayon": "Seluruh Kota Surabaya (7 Rayon)",
        "jumlah_titik": tot_n,
        "kdist_median_m": np.nan,
        "kdist_p90_m": np.nan,
        "kdist_p95_m": np.nan,
        "kdist_max_m": np.nan,
        "knee_index": np.nan,
        "knee_percentile": np.nan,
        "eps_kneedle_m": np.nan,
        "eps_digunakan_m": np.nan,
        "sumber_eps": "Adaptif per Rayon",
        "jumlah_klaster_dbscan": int(df_summary["jumlah_klaster_dbscan"].sum()),
        "klaster_terbesar_titik": int(df_summary["klaster_terbesar_titik"].max()),
        "jumlah_core": tot_core,
        "persen_core": round(tot_core / tot_n * 100.0, 2),
        "jumlah_border": tot_border,
        "persen_border": round(tot_border / tot_n * 100.0, 2),
        "jumlah_non_noise": tot_non_noise,
        "persen_non_noise": round(tot_non_noise / tot_n * 100.0, 2),
        "jumlah_noise": tot_noise,
        "persen_noise": round(tot_noise / tot_n * 100.0, 2),
        "status_diagnostik": "Ringkasan Total",
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([total_row])], ignore_index=True)

    return gdf_result, df_summary, kdist_artifacts


def save_dbscan_outputs(
    processed_dir: Path,
    tables_dir: Path,
    gdf_dbscan: gpd.GeoDataFrame,
    df_summary: pd.DataFrame,
) -> Dict[str, Path]:
    """Menyimpan hasil DBSCAN (termasuk titik Noise) ke data/processed/ dan outputs/tables/."""
    processed_dir = Path(processed_dir)
    tables_dir = Path(tables_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

    p_gpkg = processed_dir / "saluran_dbscan_utm49s.gpkg"
    gdf_dbscan.to_file(p_gpkg, driver="GPKG")
    saved["saluran_dbscan_gpkg"] = p_gpkg

    p_csv = processed_dir / "saluran_dbscan_utm49s.csv"
    gdf_dbscan.drop(columns="geometry").to_csv(p_csv, index=False)
    saved["saluran_dbscan_csv"] = p_csv

    p_tab = tables_dir / "03_dbscan_ringkasan_per_rayon.csv"
    df_summary.to_csv(p_tab, index=False)
    saved["dbscan_ringkasan_csv"] = p_tab

    return saved


# =============================================================================
# TAHAP 4 — K-MEANS++ PER RAYON (PADA GABUNGAN SELURUH TITIK NON-NOISE)
# =============================================================================


def evaluate_elbow_single_rayon(
    coords_xy: np.ndarray,
    k_values: Iterable[int] = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_KMEANS_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    sensitivity: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """
    Mengevaluasi kurva Elbow Method (WCSS / inertia) pada satu rayon untuk rentang K (default 2..10)
    menggunakan K-Means++ dan mendeteksi titik siku (elbow) dengan `kneed.KneeLocator`.

    Validasi prasyarat:
    - Memastikan jumlah titik dan jumlah koordinat unik >= max(k_values).
    """
    coords_xy = np.asarray(coords_xy, dtype=float)
    k_list = sorted(int(k) for k in k_values)
    n_samples = len(coords_xy)
    n_unique = len(np.unique(coords_xy, axis=0))

    if n_samples < max(k_list) or n_unique < max(k_list):
        raise ValueError(
            f"Jumlah sampel ({n_samples}) atau koordinat unik ({n_unique}) tidak mencukupi "
            f"untuk menguji K hingga {max(k_list)}."
        )

    wcss_list: List[float] = []
    for k in k_list:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            random_state=random_state,
        )
        km.fit(coords_xy)
        wcss_list.append(float(km.inertia_))

    # Hitung penurunan WCSS (absolut dan persentase terhadap K-1 dan terhadap K=min)
    rows = []
    for idx, (k, wcss) in enumerate(zip(k_list, wcss_list)):
        if idx == 0:
            drop_abs = np.nan
            drop_pct = np.nan
        else:
            prev_w = wcss_list[idx - 1]
            drop_abs = prev_w - wcss
            drop_pct = (drop_abs / prev_w * 100.0) if prev_w > 0 else np.nan
        cum_drop_pct = (wcss_list[0] - wcss) / wcss_list[0] * 100.0 if wcss_list[0] > 0 else 0.0
        rows.append({
            "k": k,
            "wcss_m2": wcss,
            "wcss_1e9_m2": round(wcss / 1e9, 4),
            "penurunan_wcss_m2": drop_abs,
            "penurunan_relatif_pct": round(drop_pct, 2) if not np.isnan(drop_pct) else np.nan,
            "penurunan_kumulatif_dari_k2_pct": round(cum_drop_pct, 2),
        })
    df_elbow = pd.DataFrame(rows)

    # Deteksi titik siku dengan Kneedle (curve="convex", direction="decreasing")
    kl = KneeLocator(
        x=k_list,
        y=wcss_list,
        S=sensitivity,
        curve="convex",
        direction="decreasing",
        online=False,
    )

    # Diagnostik kewajaran elbow
    # Cek apakah WCSS monoton menurun dan apakah elbow terdeteksi pada titik interior
    is_monotonic = all(wcss_list[i] >= wcss_list[i + 1] for i in range(len(wcss_list) - 1))
    diag_flags: List[str] = []
    if kl.knee is None:
        diag_flags.append("Kneedle tidak menemukan titik elbow")
    else:
        if int(kl.knee) in (k_list[0], k_list[-1]):
            diag_flags.append(f"Elbow jatuh di batas rentang (K={int(kl.knee)})")
    if not is_monotonic:
        diag_flags.append("Kurva WCSS tidak monoton menurun")

    # Cek kedekatan skor selisih ternormalisasi antara K terpilih dan kandidat runner-up
    y_diff = np.asarray(kl.y_difference, dtype=float) if kl.y_difference is not None else np.array([])
    runner_up_k = None
    margin_diff = None
    if kl.knee is not None and len(y_diff) == len(k_list):
        order_desc = np.argsort(y_diff)[::-1]
        best_idx = int(order_desc[0])
        second_idx = int(order_desc[1]) if len(order_desc) > 1 else best_idx
        runner_up_k = k_list[second_idx]
        margin_diff = float(y_diff[best_idx] - y_diff[second_idx])
        if margin_diff < 0.02:
            diag_flags.append(
                f"Selisih kurva Kneedle tipis antara K={int(kl.knee)} dan K={runner_up_k} (Δ={margin_diff:.4f})"
            )

    elbow_info = {
        "kneedle_found": kl.knee is not None,
        "k_elbow": int(kl.knee) if kl.knee is not None else None,
        "runner_up_k": runner_up_k,
        "kneedle_margin": round(margin_diff, 4) if margin_diff is not None else None,
        "is_monotonic": is_monotonic,
        "y_difference": y_diff.tolist(),
        "status_diagnostik": "Wajar (Tegas)" if not diag_flags else "PERLU CEK: " + "; ".join(diag_flags),
    }

    return df_elbow, elbow_info


def run_kmeans_single_rayon(
    coords_xy: np.ndarray,
    k: int,
    n_init: int = DEFAULT_KMEANS_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Menjalankan K-Means++ pada koordinat titik Non-Noise satu rayon lalu menomori ulang
    subcluster secara deterministik (1..K) terurut berdasarkan `easting` (Barat -> Timur),
    lalu `northing` (Selatan -> Utara).

    Catatan: Permutasi ulang label subcluster sama sekali tidak mengubah keanggotaan,
    posisi centroid, WCSS, maupun metrik evaluasi clustering.
    """
    coords_xy = np.asarray(coords_xy, dtype=float)
    km = KMeans(
        n_clusters=int(k),
        init="k-means++",
        n_init=int(n_init),
        random_state=int(random_state),
    )
    raw_labels = km.fit_predict(coords_xy)
    raw_centers = km.cluster_centers_

    # Urutkan centroid dari Barat ke Timur (easting), lalu Selatan ke Utara (northing)
    sort_order = np.lexsort((raw_centers[:, 1], raw_centers[:, 0]))
    old_to_new = {int(old_idx): int(new_idx + 1) for new_idx, old_idx in enumerate(sort_order)}

    ordered_labels = np.array([old_to_new[int(lbl)] for lbl in raw_labels], dtype=int)
    ordered_centers = raw_centers[sort_order]

    return ordered_labels, ordered_centers, float(km.inertia_)


def run_kmeans_all_rayons(
    gdf_dbscan: gpd.GeoDataFrame,
    k_values: Iterable[int] = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_KMEANS_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
    sensitivity: float = 1.0,
    k_overrides: Optional[Dict[str, int]] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan Tahap 4 (Elbow Method + K-Means++) secara terpisah per rayon pada
    GABUNGAN SELURUH TITIK NON-NOISE (`Core + Border`, `is_non_noise == True`).

    Titik Noise (`is_non_noise == False`) TIDAK dilatih di K-Means, tetapi tetap
    dipertahankan pada output dengan `kmeans_cluster = -1` dan `kmeans_label = "Noise"`.

    Returns:
    - gdf_points_out: GeoDataFrame lengkap seluruh titik (21.272 titik, termasuk Noise)
      dengan kolom tambahan `kmeans_k_rayon`, `kmeans_cluster` (1..K atau -1),
      `centroid_id` (`RYN-xx-Cyy` atau `"Noise"`), `jarak_ke_centroid_m`.
    - gdf_centroids: GeoDataFrame seluruh centroid kandidat awal (CRS `EPSG:32749`)
      dilengkapi proyeksi balik `longitude`, `latitude` (`EPSG:4326`), jumlah titik anggota,
      dan statistik jarak anggota ke centroid.
    - df_elbow_all: Tabel lengkap WCSS untuk setiap (rayon, K=2..10).
    - df_summary: Tabel ringkasan K-Means per rayon (+ baris TOTAL).
    """
    if gdf_dbscan.crs is None or gdf_dbscan.crs.to_epsg() != 32749:
        raise ValueError(f"Input harus ber-CRS EPSG:32749, ditemukan: {gdf_dbscan.crs}")

    k_overrides = k_overrides or {}
    utm_to_wgs = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)

    point_parts: List[gpd.GeoDataFrame] = []
    centroid_rows: List[Dict[str, object]] = []
    elbow_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    rayon_order = (
        gdf_dbscan[["id_rayon", "nama_rayon"]]
        .drop_duplicates()
        .sort_values("id_rayon")
        .itertuples(index=False)
    )

    for rid, rname in rayon_order:
        sub = gdf_dbscan[gdf_dbscan["nama_rayon"] == rname].copy()
        mask_non_noise = sub["is_non_noise"].astype(bool).to_numpy()
        sub_nn = sub.loc[mask_non_noise].copy()
        sub_noise = sub.loc[~mask_non_noise].copy()

        coords_nn = sub_nn[["easting", "northing"]].to_numpy(dtype=float)

        # 1. Evaluasi Elbow Method (K = 2..10) pada gabungan Core + Border
        df_elbow, elbow_info = evaluate_elbow_single_rayon(
            coords_xy=coords_nn,
            k_values=k_values,
            n_init=n_init,
            random_state=random_state,
            sensitivity=sensitivity,
        )
        df_elbow.insert(0, "nama_rayon", rname)
        df_elbow.insert(0, "id_rayon", rid)
        df_elbow["is_kneedle_elbow"] = df_elbow["k"] == elbow_info["k_elbow"]
        elbow_parts.append(df_elbow)

        # Tentukan K yang digunakan
        if rname in k_overrides:
            k_used = int(k_overrides[rname])
            k_source = "Override (Evaluasi)"
        else:
            if elbow_info["k_elbow"] is None:
                raise RuntimeError(
                    f"Kneedle gagal mendeteksi titik elbow pada {rname}. "
                    "Evaluasi kurva WCSS terlebih dahulu sebelum memilih K."
                )
            k_used = int(elbow_info["k_elbow"])
            k_source = "Elbow + Kneedle (S=1.0)"

        # 2. Jalankan K-Means++ dengan K terpilih
        labels_nn, centers_utm, inertia_used = run_kmeans_single_rayon(
            coords_xy=coords_nn,
            k=k_used,
            n_init=n_init,
            random_state=random_state,
        )

        # Hitung jarak tiap titik Non-Noise ke centroid subclusternya (meter)
        assigned_centers = centers_utm[labels_nn - 1]
        dists_to_c = np.hypot(
            coords_nn[:, 0] - assigned_centers[:, 0],
            coords_nn[:, 1] - assigned_centers[:, 1],
        )

        sub_nn["kmeans_k_rayon"] = k_used
        sub_nn["kmeans_cluster"] = labels_nn
        sub_nn["centroid_id"] = [f"{rid}-C{c:02d}" for c in labels_nn]
        sub_nn["jarak_ke_centroid_m"] = np.round(dists_to_c, 2)

        # Titik Noise tetap disimpan utuh tanpa ikut K-Means
        sub_noise["kmeans_k_rayon"] = k_used
        sub_noise["kmeans_cluster"] = -1
        sub_noise["centroid_id"] = "Noise"
        sub_noise["jarak_ke_centroid_m"] = np.nan

        point_parts.append(pd.concat([sub_nn, sub_noise], ignore_index=True))

        # 3. Bangun daftar centroid kandidat awal (UTM 49S + proyeksi balik WGS84)
        c_sizes = []
        for c_idx in range(1, k_used + 1):
            c_mask = labels_nn == c_idx
            c_sub = sub_nn.loc[c_mask]
            c_dists = dists_to_c[c_mask]
            cx = float(centers_utm[c_idx - 1, 0])
            cy = float(centers_utm[c_idx - 1, 1])
            clon, clat = utm_to_wgs.transform(cx, cy)

            n_members = int(c_mask.sum())
            c_sizes.append(n_members)
            hier_counts = c_sub["kode_hierarki"].value_counts().to_dict()

            centroid_rows.append({
                "centroid_id": f"{rid}-C{c_idx:02d}",
                "id_rayon": rid,
                "nama_rayon": rname,
                "kmeans_cluster": c_idx,
                "k_rayon": k_used,
                "easting": round(cx, 2),
                "northing": round(cy, 2),
                "longitude": round(float(clon), 6),
                "latitude": round(float(clat), 6),
                "jumlah_titik": n_members,
                "persen_non_noise_rayon": round(n_members / len(sub_nn) * 100.0, 2),
                "n_sp": int(hier_counts.get("SP", 0)),
                "n_ss": int(hier_counts.get("SS", 0)),
                "n_st": int(hier_counts.get("ST", 0)),
                "n_sl": int(hier_counts.get("SL", 0)),
                "jarak_rerata_ke_centroid_m": round(float(c_dists.mean()), 2),
                "jarak_maks_ke_centroid_m": round(float(c_dists.max()), 2),
                "geometry": Point(cx, cy),
            })

        # 4. Baris ringkasan per rayon
        wcss_k2 = float(df_elbow.loc[df_elbow["k"] == min(k_values), "wcss_m2"].iloc[0])
        drop_from_k2 = (wcss_k2 - inertia_used) / wcss_k2 * 100.0 if wcss_k2 > 0 else 0.0

        summary_rows.append({
            "id_rayon": rid,
            "nama_rayon": rname,
            "jumlah_total_titik": len(sub),
            "jumlah_non_noise_input": len(sub_nn),
            "jumlah_noise_tidak_dilatih": len(sub_noise),
            "k_kneedle": elbow_info["k_elbow"],
            "runner_up_k": elbow_info["runner_up_k"],
            "kneedle_margin": elbow_info["kneedle_margin"],
            "k_digunakan": k_used,
            "sumber_k": k_source,
            "wcss_terpilih_1e9_m2": round(inertia_used / 1e9, 4),
            "penurunan_wcss_dari_k2_pct": round(drop_from_k2, 2),
            "min_anggota_subcluster": int(min(c_sizes)),
            "max_anggota_subcluster": int(max(c_sizes)),
            "rerata_anggota_subcluster": round(float(np.mean(c_sizes)), 1),
            "status_diagnostik_elbow": elbow_info["status_diagnostik"],
        })

    gdf_points_out = gpd.GeoDataFrame(
        pd.concat(point_parts, ignore_index=True),
        crs=gdf_dbscan.crs,
    ).sort_values("no").reset_index(drop=True)

    gdf_centroids = gpd.GeoDataFrame(
        centroid_rows,
        geometry="geometry",
        crs=gdf_dbscan.crs,
    )

    df_elbow_all = pd.concat(elbow_parts, ignore_index=True)
    df_summary = pd.DataFrame(summary_rows)

    # Baris TOTAL
    total_row = {
        "id_rayon": "TOTAL",
        "nama_rayon": "Seluruh Kota Surabaya (7 Rayon)",
        "jumlah_total_titik": int(df_summary["jumlah_total_titik"].sum()),
        "jumlah_non_noise_input": int(df_summary["jumlah_non_noise_input"].sum()),
        "jumlah_noise_tidak_dilatih": int(df_summary["jumlah_noise_tidak_dilatih"].sum()),
        "k_kneedle": int(df_summary["k_kneedle"].sum()),
        "runner_up_k": np.nan,
        "kneedle_margin": np.nan,
        "k_digunakan": int(df_summary["k_digunakan"].sum()),
        "sumber_k": "Total Centroid Kandidat",
        "wcss_terpilih_1e9_m2": round(float(df_summary["wcss_terpilih_1e9_m2"].sum()), 4),
        "penurunan_wcss_dari_k2_pct": np.nan,
        "min_anggota_subcluster": int(df_summary["min_anggota_subcluster"].min()),
        "max_anggota_subcluster": int(df_summary["max_anggota_subcluster"].max()),
        "rerata_anggota_subcluster": round(
            float(df_summary["jumlah_non_noise_input"].sum() / df_summary["k_digunakan"].sum()), 1
        ),
        "status_diagnostik_elbow": "Ringkasan Total",
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([total_row])], ignore_index=True)

    # Validasi ketat integritas hasil
    if len(gdf_points_out) != len(gdf_dbscan):
        raise AssertionError("Jumlah titik output K-Means tidak sama dengan input DBSCAN!")
    if int((gdf_points_out["kmeans_cluster"] == -1).sum()) != int((~gdf_dbscan["is_non_noise"]).sum()):
        raise AssertionError("Jumlah titik Noise pada output K-Means tidak cocok dengan hasil DBSCAN!")

    return gdf_points_out, gdf_centroids, df_elbow_all, df_summary


def save_kmeans_outputs(
    processed_dir: Path,
    tables_dir: Path,
    gdf_kmeans: gpd.GeoDataFrame,
    gdf_centroids: gpd.GeoDataFrame,
    df_elbow_all: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> Dict[str, Path]:
    """Menyimpan seluruh output Tahap 4 (titik K-Means, centroid, WCSS Elbow, ringkasan)."""
    processed_dir = Path(processed_dir)
    tables_dir = Path(tables_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

    p_pts_gpkg = processed_dir / "saluran_kmeans_utm49s.gpkg"
    gdf_kmeans.to_file(p_pts_gpkg, driver="GPKG")
    saved["saluran_kmeans_gpkg"] = p_pts_gpkg

    p_pts_csv = processed_dir / "saluran_kmeans_utm49s.csv"
    gdf_kmeans.drop(columns="geometry").to_csv(p_pts_csv, index=False)
    saved["saluran_kmeans_csv"] = p_pts_csv

    p_cen_gpkg = processed_dir / "centroid_kmeans_utm49s.gpkg"
    gdf_centroids.to_file(p_cen_gpkg, driver="GPKG")
    saved["centroid_kmeans_gpkg"] = p_cen_gpkg

    p_cen_csv = processed_dir / "centroid_kmeans_utm49s.csv"
    gdf_centroids.drop(columns="geometry").to_csv(p_cen_csv, index=False)
    saved["centroid_kmeans_csv"] = p_cen_csv

    p_elbow_csv = tables_dir / "04_kmeans_elbow_wcss_7_rayon.csv"
    df_elbow_all.to_csv(p_elbow_csv, index=False)
    saved["elbow_wcss_csv"] = p_elbow_csv

    p_sum_csv = tables_dir / "04_kmeans_ringkasan_per_rayon.csv"
    df_summary.to_csv(p_sum_csv, index=False)
    saved["kmeans_ringkasan_csv"] = p_sum_csv

    p_cen_tab = tables_dir / "04_kmeans_centroid_kandidat.csv"
    gdf_centroids.drop(columns="geometry").to_csv(p_cen_tab, index=False)
    saved["centroid_kandidat_table_csv"] = p_cen_tab

    return saved
