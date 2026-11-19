"""
Modul Tahap 5 — Evaluasi Struktur Klasterisasi (Silhouette Coefficient & Davies-Bouldin Index)
Penelitian: Penentuan Lokasi Rumah Pompa Air Pengendali Banjir Menggunakan
Klasterisasi DBSCAN dan K-Means (Studi Kasus: Kota Surabaya)

Catatan metodologis penting:
- Evaluasi dilakukan secara terpisah pada hasil K-Means++ (titik Non-Noise: Core + Border)
  di masing-masing 7 rayon pada koordinat proyeksi UTM Zona 49S (`easting`, `northing`).
- Titik Noise (`kmeans_cluster == -1`) tidak diikutsertakan dalam perhitungan SC dan DBI.
- SC dan DBI mengevaluasi kualitas struktur spasial klasterisasi (kohesi internal dan
  separasi antar-klaster), BUKAN kelayakan konstruksi, kebutuhan banjir, atau kapasitas
  hidraulis rumah pompa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

from src.clustering import (
    DEFAULT_K_RANGE,
    DEFAULT_KMEANS_N_INIT,
    DEFAULT_RANDOM_STATE,
    run_kmeans_single_rayon,
)


def validate_metric_requirements(
    coords_xy: np.ndarray,
    labels: np.ndarray,
) -> Tuple[bool, str]:
    """
    Memeriksa prasyarat matematis perhitungan Silhouette Coefficient dan Davies-Bouldin Index:
    - Jumlah sampel n >= 3.
    - Jumlah klaster unik k memenuhi 2 <= k <= n - 1.
    - Tidak ada nilai NaN/Inf pada koordinat.
    """
    coords_xy = np.asarray(coords_xy, dtype=float)
    labels = np.asarray(labels)

    if coords_xy.ndim != 2 or len(coords_xy) != len(labels):
        return False, "Dimensi koordinat dan panjang label tidak sesuai."
    if not np.all(np.isfinite(coords_xy)):
        return False, "Terdapat nilai NaN/Inf pada koordinat."

    n_samples = len(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_samples < 3:
        return False, f"Jumlah sampel terlalu sedikit (n={n_samples} < 3)."
    if n_clusters < 2:
        return False, f"Jumlah klaster kurang dari 2 (k={n_clusters})."
    if n_clusters >= n_samples:
        return False, f"Jumlah klaster (k={n_clusters}) >= jumlah sampel (n={n_samples})."

    return True, "Valid"


def interpret_silhouette_kaufman(score: Optional[float]) -> str:
    """
    Interpretasi kategori struktur klaster berdasarkan acuan klasik
    Kaufman & Rousseeuw (1990) untuk Silhouette Coefficient:
    - 0.71 – 1.00 : Struktur kuat
    - 0.51 – 0.70 : Struktur wajar (cukup baik)
    - 0.26 – 0.50 : Struktur lemah / perlu dicermati
    - <= 0.25     : Tidak ada struktur klaster substansial
    """
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "N/A"
    if score > 0.70:
        return "Struktur Kuat (>0.70)"
    if score > 0.50:
        return "Struktur Wajar (0.51-0.70)"
    if score > 0.25:
        return "Struktur Lemah (0.26-0.50)"
    return "Tidak Substansial (<=0.25)"


def evaluate_selected_kmeans_all_rayons(
    gdf_kmeans: gpd.GeoDataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Menghitung Silhouette Coefficient (SC), Davies-Bouldin Index (DBI), dan profil
    Silhouette per subcluster pada hasil K-Means++ terpilih untuk masing-masing 7 rayon.

    Returns:
    - df_eval_rayon: Tabel evaluasi per rayon (+ baris rata-rata 7 rayon & tertimbang).
    - df_eval_subcluster: Tabel evaluasi Silhouette per subcluster (31 subcluster).
    - sample_sil_dict: Dictionary per rayon berisi `labels`, `sil_samples`, `sc_mean`,
      `dbi` untuk plotting Silhouette Plot.
    """
    if gdf_kmeans.crs is None or gdf_kmeans.crs.to_epsg() != 32749:
        raise ValueError(f"Input harus ber-CRS EPSG:32749, ditemukan: {gdf_kmeans.crs}")

    rayon_rows: List[Dict[str, object]] = []
    subcluster_rows: List[Dict[str, object]] = []
    sample_sil_dict: Dict[str, Dict[str, object]] = {}

    rayon_order = (
        gdf_kmeans[["id_rayon", "nama_rayon"]]
        .drop_duplicates()
        .sort_values("id_rayon")
        .itertuples(index=False)
    )

    for rid, rname in rayon_order:
        sub = gdf_kmeans[gdf_kmeans["nama_rayon"] == rname]
        # Hanya titik Non-Noise (Core + Border) yang dievaluasi pada hasil K-Means
        sub_nn = sub[sub["is_non_noise"].astype(bool) & (sub["kmeans_cluster"] >= 1)].copy()
        n_noise = int((~sub["is_non_noise"].astype(bool)).sum())

        coords = sub_nn[["easting", "northing"]].to_numpy(dtype=float)
        labels = sub_nn["kmeans_cluster"].to_numpy(dtype=int)
        k_sel = int(len(np.unique(labels)))

        is_valid, reason = validate_metric_requirements(coords, labels)
        if not is_valid:
            rayon_rows.append({
                "id_rayon": rid,
                "nama_rayon": rname,
                "k_terpilih": k_sel,
                "jumlah_non_noise": len(sub_nn),
                "jumlah_noise_dikecualikan": n_noise,
                "silhouette_score": np.nan,
                "kategori_silhouette": "N/A",
                "davies_bouldin_index": np.nan,
                "sil_min": np.nan,
                "sil_median": np.nan,
                "sil_max": np.nan,
                "jumlah_sil_negatif": np.nan,
                "persen_sil_negatif": np.nan,
                "status_validasi": f"N/A: {reason}",
            })
            continue

        sc_val = float(silhouette_score(coords, labels, metric="euclidean"))
        dbi_val = float(davies_bouldin_score(coords, labels))
        sil_vals = silhouette_samples(coords, labels, metric="euclidean")

        n_neg = int((sil_vals < 0).sum())
        pct_neg = n_neg / len(sil_vals) * 100.0

        rayon_rows.append({
            "id_rayon": rid,
            "nama_rayon": rname,
            "k_terpilih": k_sel,
            "jumlah_non_noise": len(sub_nn),
            "jumlah_noise_dikecualikan": n_noise,
            "silhouette_score": round(sc_val, 4),
            "kategori_silhouette": interpret_silhouette_kaufman(sc_val),
            "davies_bouldin_index": round(dbi_val, 4),
            "sil_min": round(float(sil_vals.min()), 4),
            "sil_median": round(float(np.median(sil_vals)), 4),
            "sil_max": round(float(sil_vals.max()), 4),
            "jumlah_sil_negatif": n_neg,
            "persen_sil_negatif": round(pct_neg, 2),
            "status_validasi": "Valid",
        })

        sample_sil_dict[rname] = {
            "id_rayon": rid,
            "k_terpilih": k_sel,
            "labels": labels,
            "sil_samples": sil_vals,
            "sc_mean": sc_val,
            "dbi": dbi_val,
            "n_samples": len(sub_nn),
        }

        # Evaluasi Silhouette per subcluster
        for c_id in sorted(np.unique(labels)):
            c_mask = labels == c_id
            c_sil = sil_vals[c_mask]
            c_neg = int((c_sil < 0).sum())
            subcluster_rows.append({
                "centroid_id": f"{rid}-C{int(c_id):02d}",
                "id_rayon": rid,
                "nama_rayon": rname,
                "kmeans_cluster": int(c_id),
                "jumlah_titik": int(c_mask.sum()),
                "sil_mean": round(float(c_sil.mean()), 4),
                "sil_median": round(float(np.median(c_sil)), 4),
                "sil_min": round(float(c_sil.min()), 4),
                "sil_max": round(float(c_sil.max()), 4),
                "jumlah_sil_negatif": c_neg,
                "persen_sil_negatif": round(c_neg / int(c_mask.sum()) * 100.0, 2),
            })

    df_eval_rayon = pd.DataFrame(rayon_rows)
    df_eval_subcluster = pd.DataFrame(subcluster_rows)

    # Ringkasan rata-rata 7 rayon (unweighted & weighted by jumlah_non_noise)
    valid_df = df_eval_rayon[df_eval_rayon["status_validasi"] == "Valid"]
    if len(valid_df) > 0:
        weights = valid_df["jumlah_non_noise"].to_numpy(dtype=float)
        avg_sc = float(valid_df["silhouette_score"].mean())
        avg_dbi = float(valid_df["davies_bouldin_index"].mean())
        w_sc = float(np.average(valid_df["silhouette_score"], weights=weights))
        w_dbi = float(np.average(valid_df["davies_bouldin_index"], weights=weights))
        tot_nn = int(valid_df["jumlah_non_noise"].sum())
        tot_noise = int(valid_df["jumlah_noise_dikecualikan"].sum())
        tot_neg = int(valid_df["jumlah_sil_negatif"].sum())

        summary_extra = pd.DataFrame([
            {
                "id_rayon": "MEAN",
                "nama_rayon": "Rata-rata 7 Rayon (Unweighted)",
                "k_terpilih": int(valid_df["k_terpilih"].sum()),
                "jumlah_non_noise": tot_nn,
                "jumlah_noise_dikecualikan": tot_noise,
                "silhouette_score": round(avg_sc, 4),
                "kategori_silhouette": interpret_silhouette_kaufman(avg_sc),
                "davies_bouldin_index": round(avg_dbi, 4),
                "sil_min": round(float(valid_df["sil_min"].min()), 4),
                "sil_median": round(float(valid_df["sil_median"].mean()), 4),
                "sil_max": round(float(valid_df["sil_max"].max()), 4),
                "jumlah_sil_negatif": tot_neg,
                "persen_sil_negatif": round(tot_neg / tot_nn * 100.0, 2),
                "status_validasi": "Ringkasan Unweighted",
            },
            {
                "id_rayon": "W-MEAN",
                "nama_rayon": "Rata-rata Tertimbang Jumlah Titik",
                "k_terpilih": int(valid_df["k_terpilih"].sum()),
                "jumlah_non_noise": tot_nn,
                "jumlah_noise_dikecualikan": tot_noise,
                "silhouette_score": round(w_sc, 4),
                "kategori_silhouette": interpret_silhouette_kaufman(w_sc),
                "davies_bouldin_index": round(w_dbi, 4),
                "sil_min": np.nan,
                "sil_median": np.nan,
                "sil_max": np.nan,
                "jumlah_sil_negatif": tot_neg,
                "persen_sil_negatif": round(tot_neg / tot_nn * 100.0, 2),
                "status_validasi": "Ringkasan Weighted",
            },
        ])
        df_eval_rayon = pd.concat([df_eval_rayon, summary_extra], ignore_index=True)

    return df_eval_rayon, df_eval_subcluster, sample_sil_dict


def evaluate_k_range_comparison(
    gdf_kmeans: gpd.GeoDataFrame,
    k_values: Iterable[int] = DEFAULT_K_RANGE,
    n_init: int = DEFAULT_KMEANS_N_INIT,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghitung profil WCSS, Silhouette Coefficient (SC), dan Davies-Bouldin Index (DBI)
    di seluruh rentang K = 2..10 pada masing-masing 7 rayon sebagai analisis komparasi
    terhadap K terpilih hasil Elbow + Kneedle.

    Returns:
    - df_comp_all: Tabel panjang (7 rayon x 9 nilai K = 63 baris) lengkap dengan penanda K terpilih.
    - df_comp_summary: Tabel ringkasan perbandingan per rayon (K Elbow vs K Terbaik SC vs K Terbaik DBI).
    """
    k_list = sorted(int(k) for k in k_values)
    comp_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    rayon_order = (
        gdf_kmeans[["id_rayon", "nama_rayon"]]
        .drop_duplicates()
        .sort_values("id_rayon")
        .itertuples(index=False)
    )

    for rid, rname in rayon_order:
        sub_nn = gdf_kmeans[
            (gdf_kmeans["nama_rayon"] == rname) & (gdf_kmeans["is_non_noise"].astype(bool))
        ]
        coords = sub_nn[["easting", "northing"]].to_numpy(dtype=float)
        k_elbow = int(sub_nn["kmeans_k_rayon"].iloc[0])

        rayon_k_rows = []
        for k in k_list:
            labels_k, _, inertia_k = run_kmeans_single_rayon(
                coords_xy=coords,
                k=k,
                n_init=n_init,
                random_state=random_state,
            )
            is_valid, reason = validate_metric_requirements(coords, labels_k)
            if is_valid:
                sc_k = float(silhouette_score(coords, labels_k, metric="euclidean"))
                dbi_k = float(davies_bouldin_score(coords, labels_k))
            else:
                sc_k = np.nan
                dbi_k = np.nan

            row = {
                "id_rayon": rid,
                "nama_rayon": rname,
                "k": k,
                "is_k_elbow_terpilih": (k == k_elbow),
                "wcss_1e9_m2": round(inertia_k / 1e9, 4),
                "silhouette_score": round(sc_k, 4) if not np.isnan(sc_k) else np.nan,
                "davies_bouldin_index": round(dbi_k, 4) if not np.isnan(dbi_k) else np.nan,
                "status_validasi": reason,
            }
            comp_rows.append(row)
            rayon_k_rows.append(row)

        df_rk = pd.DataFrame(rayon_k_rows)
        row_elbow = df_rk[df_rk["k"] == k_elbow].iloc[0]
        row_best_sc = df_rk.loc[df_rk["silhouette_score"].idxmax()]
        row_best_dbi = df_rk.loc[df_rk["davies_bouldin_index"].idxmin()]

        summary_rows.append({
            "id_rayon": rid,
            "nama_rayon": rname,
            "k_elbow": k_elbow,
            "sc_pada_k_elbow": row_elbow["silhouette_score"],
            "dbi_pada_k_elbow": row_elbow["davies_bouldin_index"],
            "k_terbaik_sc": int(row_best_sc["k"]),
            "sc_maks_rentang": row_best_sc["silhouette_score"],
            "selisih_sc_vs_maks": round(float(row_elbow["silhouette_score"] - row_best_sc["silhouette_score"]), 4),
            "k_terbaik_dbi": int(row_best_dbi["k"]),
            "dbi_min_rentang": row_best_dbi["davies_bouldin_index"],
            "selisih_dbi_vs_min": round(float(row_elbow["davies_bouldin_index"] - row_best_dbi["davies_bouldin_index"]), 4),
        })

    return pd.DataFrame(comp_rows), pd.DataFrame(summary_rows)


def save_evaluation_outputs(
    tables_dir: Path,
    df_eval_rayon: pd.DataFrame,
    df_eval_subcluster: pd.DataFrame,
    df_comp_all: pd.DataFrame,
    df_comp_summary: pd.DataFrame,
) -> Dict[str, Path]:
    """Menyimpan seluruh tabel evaluasi Tahap 5 ke outputs/tables/."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "eval_per_rayon_csv": (df_eval_rayon, tables_dir / "05_evaluasi_kmeans_per_rayon.csv"),
        "eval_per_subcluster_csv": (df_eval_subcluster, tables_dir / "05_evaluasi_silhouette_per_subcluster.csv"),
        "eval_komparasi_k2_k10_csv": (df_comp_all, tables_dir / "05_evaluasi_komparasi_k2_k10.csv"),
        "eval_ringkasan_komparasi_csv": (df_comp_summary, tables_dir / "05_evaluasi_ringkasan_elbow_vs_sc_dbi.csv"),
    }
    saved: Dict[str, Path] = {}
    for key, (df, path) in targets.items():
        df.to_csv(path, index=False)
        saved[key] = path
    return saved
