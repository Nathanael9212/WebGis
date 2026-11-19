"""
Modul Tahap 6 — Analisis Jarak Spasial Centroid terhadap Rumah Pompa Aktif
Penelitian: Penentuan Lokasi Rumah Pompa Air Pengendali Banjir Menggunakan
Klasterisasi DBSCAN dan K-Means (Studi Kasus: Kota Surabaya)

Aturan metodologis ketat:
- Perhitungan jarak utama dilakukan pada CRS proyeksi seragam UTM Zona 49S (`EPSG:32749`)
  menggunakan jarak lurus Euclidean (*straight-line distance*, satuan meter) dari setiap
  centroid kandidat awal (31 centroid) terhadap SELURUH 71 rumah pompa Operasional Aktif
  Kota Surabaya (lintas batas rayon diizinkan dan dicatat secara eksplisit).
- Jika terdapat >1 rumah pompa aktif dengan jarak minimum identik (mis. duplikat koordinat
  `PMP-013` & `PMP-014` P.A Bozem Moro Krembangan I), seluruh ID dicatat secara transparan.
- Sebanyak 12 rumah pompa Pengembangan Baru TIDAK dicampur ke analisis jarak utama,
  melainkan disimpan pada analisis sekunder terpisah sebagai konteks pendukung.
- Tidak menggunakan buffer 1,5 km maupun snapping ke jaringan jalan/saluran.
- Jarak ini adalah jarak lurus Euclidean, BUKAN jarak jaringan drainase dan BUKAN bukti
  ketidakterlayanan hidraulis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


CRS_PROJECTED = "EPSG:32749"
CRS_GEOGRAPHIC = "EPSG:4326"
TIE_TOLERANCE_M = 1e-4  # 0.1 mm untuk mendeteksi jarak identik akibat duplikat koordinat


def compute_nearest_active_pump_distances(
    gdf_centroids: gpd.GeoDataFrame,
    gdf_pompa_aktif: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    """
    Menghitung jarak lurus Euclidean (meter, EPSG:32749) dari setiap centroid kandidat awal
    terhadap seluruh 71 rumah pompa Operasional Aktif Kota Surabaya.

    Returns:
    - gdf_centroid_dist: GeoDataFrame 31 centroid lengkap dengan informasi rumah pompa aktif
      terdekat (kota & serayon), flag lintas rayon, flag tie koordinat, dan peringkat jarak.
    - gdf_links: GeoDataFrame LineString penghubung (31 garis) dari centroid ke rumah pompa
      aktif terdekat pada EPSG:32749.
    - df_rayon_summary: Tabel ringkasan statistik jarak (min, median, mean, max) per rayon
      ditambah baris ringkasan keseluruhan 31 centroid.
    """
    for label, gdf in [("centroids", gdf_centroids), ("pompa_aktif", gdf_pompa_aktif)]:
        if gdf.crs is None or gdf.crs.to_epsg() != 32749:
            raise ValueError(f"CRS '{label}' harus EPSG:32749, ditemukan: {gdf.crs}")

    if not (gdf_pompa_aktif["status_fasilitas"] == "Operasional Aktif").all():
        raise ValueError("Input `gdf_pompa_aktif` harus hanya berisi 'Operasional Aktif'!")

    cen = gdf_centroids.sort_values(["id_rayon", "kmeans_cluster"]).reset_index(drop=True).copy()
    pmp = gdf_pompa_aktif.reset_index(drop=True).copy()

    c_xy = cen[["easting", "northing"]].to_numpy(dtype=float)
    p_xy = pmp[["easting", "northing"]].to_numpy(dtype=float)

    # Matriks jarak Euclidean (N_centroid x N_pompa_aktif) dalam meter
    dx = c_xy[:, 0:1] - p_xy[None, :, 0]
    dy = c_xy[:, 1:2] - p_xy[None, :, 1]
    dist_matrix = np.hypot(dx, dy)

    enriched_rows: List[Dict[str, object]] = []
    link_rows: List[Dict[str, object]] = []

    for i, crow in cen.iterrows():
        dists_i = dist_matrix[i]
        min_idx = int(np.argmin(dists_i))
        min_dist = float(dists_i[min_idx])

        # Deteksi tie jarak minimum (mis. PMP-013 & PMP-014 koordinat identik)
        tie_indices = np.where(dists_i <= min_dist + TIE_TOLERANCE_M)[0]
        is_tie = len(tie_indices) > 1
        tie_ids = [str(pmp.loc[idx, "id_pompa"]) for idx in tie_indices]
        tie_note = (
            f"Tie {len(tie_indices)} entri koordinat identik: {', '.join(tie_ids)}"
            if is_tie
            else ""
        )

        p_nearest = pmp.loc[min_idx]
        same_rayon = (p_nearest["rayon_pematusan"] == crow["nama_rayon"])

        # Juga hitung pompa aktif terdekat di dalam rayon yang sama (sebagai info pelengkap)
        mask_same_r = (pmp["rayon_pematusan"] == crow["nama_rayon"]).to_numpy()
        if np.any(mask_same_r):
            same_r_indices = np.where(mask_same_r)[0]
            best_sr_local = int(np.argmin(dists_i[same_r_indices]))
            best_sr_idx = int(same_r_indices[best_sr_local])
            sr_dist = round(float(dists_i[best_sr_idx]), 2)
            sr_id = str(pmp.loc[best_sr_idx, "id_pompa"])
            sr_name = str(pmp.loc[best_sr_idx, "nama_stasiun_pompa"])
        else:
            sr_dist, sr_id, sr_name = np.nan, "N/A", "Tidak ada pompa aktif di rayon ini"

        row_dict = crow.to_dict()
        row_dict.update({
            "id_pompa_aktif_terdekat": " / ".join(tie_ids) if is_tie else str(p_nearest["id_pompa"]),
            "nama_pompa_aktif_terdekat": str(p_nearest["nama_stasiun_pompa"]),
            "rayon_pompa_aktif_terdekat": str(p_nearest["rayon_pematusan"]),
            "pompa_aktif_easting": round(float(p_nearest["easting"]), 2),
            "pompa_aktif_northing": round(float(p_nearest["northing"]), 2),
            "pompa_aktif_longitude": round(float(p_nearest["longitude"]), 6),
            "pompa_aktif_latitude": round(float(p_nearest["latitude"]), 6),
            "jarak_pompa_aktif_terdekat_m": round(min_dist, 2),
            "jarak_pompa_aktif_terdekat_km": round(min_dist / 1000.0, 3),
            "status_lintas_rayon": "Sama Rayon" if same_rayon else "Lintas Rayon",
            "is_tie_pompa_aktif": bool(is_tie),
            "catatan_tie_pompa_aktif": tie_note,
            "id_pompa_aktif_serayon": sr_id,
            "nama_pompa_aktif_serayon": sr_name,
            "jarak_pompa_aktif_serayon_m": sr_dist,
        })
        enriched_rows.append(row_dict)

        link_rows.append({
            "centroid_id": crow["centroid_id"],
            "id_rayon": crow["id_rayon"],
            "nama_rayon": crow["nama_rayon"],
            "id_pompa_aktif_terdekat": row_dict["id_pompa_aktif_terdekat"],
            "nama_pompa_aktif_terdekat": row_dict["nama_pompa_aktif_terdekat"],
            "rayon_pompa_aktif_terdekat": row_dict["rayon_pompa_aktif_terdekat"],
            "status_lintas_rayon": row_dict["status_lintas_rayon"],
            "jarak_m": round(min_dist, 2),
            "geometry": LineString([
                (float(crow["easting"]), float(crow["northing"])),
                (float(p_nearest["easting"]), float(p_nearest["northing"])),
            ]),
        })

    gdf_centroid_dist = gpd.GeoDataFrame(enriched_rows, geometry="geometry", crs=CRS_PROJECTED)
    # Tambahkan peringkat jarak terjauh (1 = jarak lurus terbesar ke rumah pompa aktif terdekat)
    gdf_centroid_dist["peringkat_jarak_terjauh"] = (
        gdf_centroid_dist["jarak_pompa_aktif_terdekat_m"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    gdf_links = gpd.GeoDataFrame(link_rows, geometry="geometry", crs=CRS_PROJECTED)
    gdf_links = gdf_links.merge(
        gdf_centroid_dist[["centroid_id", "peringkat_jarak_terjauh"]],
        on="centroid_id",
        how="left",
    )

    # Ringkasan jarak per rayon
    summary_rows: List[Dict[str, object]] = []
    for (rid, rname), grp in gdf_centroid_dist.groupby(["id_rayon", "nama_rayon"], sort=True):
        d = grp["jarak_pompa_aktif_terdekat_m"]
        idx_min = grp["jarak_pompa_aktif_terdekat_m"].idxmin()
        idx_max = grp["jarak_pompa_aktif_terdekat_m"].idxmax()
        n_pompa_r = int((pmp["rayon_pematusan"] == rname).sum())
        summary_rows.append({
            "id_rayon": rid,
            "nama_rayon": rname,
            "jumlah_centroid": len(grp),
            "jumlah_pompa_aktif_di_rayon": n_pompa_r,
            "jarak_min_m": round(float(d.min()), 2),
            "centroid_terdekat": str(grp.loc[idx_min, "centroid_id"]),
            "jarak_median_m": round(float(d.median()), 2),
            "jarak_rerata_m": round(float(d.mean()), 2),
            "jarak_max_m": round(float(d.max()), 2),
            "centroid_terjauh": str(grp.loc[idx_max, "centroid_id"]),
            "jumlah_lintas_rayon": int((grp["status_lintas_rayon"] == "Lintas Rayon").sum()),
            "jumlah_tie_koordinat": int(grp["is_tie_pompa_aktif"].sum()),
        })

    df_summary = pd.DataFrame(summary_rows)

    # Baris keseluruhan 31 centroid
    d_all = gdf_centroid_dist["jarak_pompa_aktif_terdekat_m"]
    idx_min_all = d_all.idxmin()
    idx_max_all = d_all.idxmax()
    total_row = {
        "id_rayon": "ALL",
        "nama_rayon": "Seluruh 7 Rayon (31 Centroid)",
        "jumlah_centroid": len(gdf_centroid_dist),
        "jumlah_pompa_aktif_di_rayon": len(pmp),
        "jarak_min_m": round(float(d_all.min()), 2),
        "centroid_terdekat": str(gdf_centroid_dist.loc[idx_min_all, "centroid_id"]),
        "jarak_median_m": round(float(d_all.median()), 2),
        "jarak_rerata_m": round(float(d_all.mean()), 2),
        "jarak_max_m": round(float(d_all.max()), 2),
        "centroid_terjauh": str(gdf_centroid_dist.loc[idx_max_all, "centroid_id"]),
        "jumlah_lintas_rayon": int((gdf_centroid_dist["status_lintas_rayon"] == "Lintas Rayon").sum()),
        "jumlah_tie_koordinat": int(gdf_centroid_dist["is_tie_pompa_aktif"].sum()),
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([total_row])], ignore_index=True)

    return gdf_centroid_dist, gdf_links, df_summary


def compute_development_context_distances(
    gdf_centroid_dist: gpd.GeoDataFrame,
    gdf_pompa_pengembangan: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Analisis sekunder TERPISAH (konteks pendukung, BUKAN analisis utama):
    Menghitung jarak lurus Euclidean (meter, EPSG:32749) dari 31 centroid terhadap
    12 lokasi rumah pompa berstatus 'Pengembangan Baru', untuk melihat centroid mana
    yang secara kebetulan berada lebih dekat ke rencana pengembangan dibanding pompa aktif.
    """
    if gdf_pompa_pengembangan.crs is None or gdf_pompa_pengembangan.crs.to_epsg() != 32749:
        raise ValueError("CRS gdf_pompa_pengembangan harus EPSG:32749.")

    cen = gdf_centroid_dist.reset_index(drop=True)
    dev = gdf_pompa_pengembangan.reset_index(drop=True)

    c_xy = cen[["easting", "northing"]].to_numpy(dtype=float)
    d_xy = dev[["easting", "northing"]].to_numpy(dtype=float)

    dist_mat = np.hypot(c_xy[:, 0:1] - d_xy[None, :, 0], c_xy[:, 1:2] - d_xy[None, :, 1])

    rows = []
    for i, crow in cen.iterrows():
        j = int(np.argmin(dist_mat[i]))
        d_dev = float(dist_mat[i, j])
        d_akt = float(crow["jarak_pompa_aktif_terdekat_m"])
        p_dev = dev.loc[j]
        rows.append({
            "centroid_id": crow["centroid_id"],
            "nama_rayon": crow["nama_rayon"],
            "jarak_pompa_aktif_terdekat_m": d_akt,
            "id_pompa_aktif_terdekat": crow["id_pompa_aktif_terdekat"],
            "id_pengembangan_terdekat": str(p_dev["id_pompa"]),
            "nama_pengembangan_terdekat": str(p_dev["nama_stasiun_pompa"]),
            "rayon_pengembangan_terdekat": str(p_dev["rayon_pematusan"]),
            "jarak_pengembangan_terdekat_m": round(d_dev, 2),
            "pengembangan_lebih_dekat_dari_aktif": bool(d_dev < d_akt),
            "selisih_jarak_aktif_minus_kembang_m": round(d_akt - d_dev, 2),
        })

    return pd.DataFrame(rows)


def export_analysis_and_webgis_outputs(
    processed_dir: Path,
    tables_dir: Path,
    geojson_dir: Path,
    gdf_centroid_dist: gpd.GeoDataFrame,
    gdf_links: gpd.GeoDataFrame,
    df_rayon_summary: pd.DataFrame,
    df_dev_context: pd.DataFrame,
    gdf_rayon_utm: gpd.GeoDataFrame,
    gdf_pompa_aktif_utm: gpd.GeoDataFrame,
    gdf_pompa_pengembangan_utm: gpd.GeoDataFrame,
    gdf_kmeans_utm: gpd.GeoDataFrame,
    df_dbscan_summary: pd.DataFrame,
    df_eval_rayon: pd.DataFrame,
) -> Dict[str, Path]:
    """
    Menyimpan seluruh output Tahap 6:
    1. GeoPackage terproses (`EPSG:32749`) di `data/processed/`.
    2. Tabel hasil jarak utama, ringkasan per rayon, dan konteks pengembangan di `outputs/tables/`.
    3. Layer GeoJSON siap pakai WebGIS (`EPSG:4326`) di `outputs/geojson/`.
    """
    processed_dir = Path(processed_dir)
    tables_dir = Path(tables_dir)
    geojson_dir = Path(geojson_dir)
    for d in (processed_dir, tables_dir, geojson_dir):
        d.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

    # 1. Processed GPKG (EPSG:32749)
    p_cen_gpkg = processed_dir / "centroid_jarak_pompa_utm49s.gpkg"
    gdf_centroid_dist.to_file(p_cen_gpkg, driver="GPKG")
    saved["centroid_jarak_gpkg"] = p_cen_gpkg

    p_lnk_gpkg = processed_dir / "garis_jarak_centroid_pompa_utm49s.gpkg"
    gdf_links.to_file(p_lnk_gpkg, driver="GPKG")
    saved["garis_jarak_gpkg"] = p_lnk_gpkg

    # 2. Tables CSV
    p_t1 = tables_dir / "06_jarak_centroid_ke_pompa_aktif.csv"
    gdf_centroid_dist.drop(columns="geometry").to_csv(p_t1, index=False)
    saved["tabel_jarak_centroid_csv"] = p_t1

    p_t2 = tables_dir / "06_ringkasan_jarak_per_rayon.csv"
    df_rayon_summary.to_csv(p_t2, index=False)
    saved["tabel_ringkasan_jarak_csv"] = p_t2

    p_t3 = tables_dir / "06_konteks_sekunder_pompa_pengembangan.csv"
    df_dev_context.to_csv(p_t3, index=False)
    saved["tabel_konteks_pengembangan_csv"] = p_t3

    # 3. Layer GeoJSON (EPSG:4326) siap pakai WebGIS Folium / Streamlit
    # Perkaya layer 7 rayon dengan ringkasan DBSCAN, K-Means, Evaluasi SC/DBI, dan jarak pompa
    db_r = df_dbscan_summary[df_dbscan_summary["id_rayon"] != "TOTAL"][[
        "id_rayon", "jumlah_titik", "eps_digunakan_m", "jumlah_klaster_dbscan",
        "jumlah_core", "jumlah_border", "jumlah_noise", "persen_noise"
    ]]
    ev_r = df_eval_rayon[~df_eval_rayon["id_rayon"].isin(["MEAN", "W-MEAN"])][[
        "id_rayon", "k_terpilih", "silhouette_score", "kategori_silhouette", "davies_bouldin_index"
    ]]
    jr_r = df_rayon_summary[df_rayon_summary["id_rayon"] != "ALL"][[
        "id_rayon", "jumlah_pompa_aktif_di_rayon", "jarak_min_m", "jarak_rerata_m", "jarak_max_m"
    ]]

    gdf_rayon_enriched = (
        gdf_rayon_utm.merge(db_r, on="id_rayon", how="left")
        .merge(ev_r, on="id_rayon", how="left")
        .merge(jr_r, on="id_rayon", how="left")
        .to_crs(CRS_GEOGRAPHIC)
    )

    geojson_layers = {
        "geojson_rayon": (gdf_rayon_enriched, geojson_dir / "7_rayon_pematusan_ringkasan.geojson"),
        "geojson_centroid": (gdf_centroid_dist.to_crs(CRS_GEOGRAPHIC), geojson_dir / "31_centroid_kandidat_jarak.geojson"),
        "geojson_links": (gdf_links.to_crs(CRS_GEOGRAPHIC), geojson_dir / "31_garis_jarak_pompa_aktif.geojson"),
        "geojson_pompa_aktif": (gdf_pompa_aktif_utm.to_crs(CRS_GEOGRAPHIC), geojson_dir / "71_rumah_pompa_aktif.geojson"),
        "geojson_pompa_pengembangan": (
            gdf_pompa_pengembangan_utm.to_crs(CRS_GEOGRAPHIC),
            geojson_dir / "12_rumah_pompa_pengembangan.geojson",
        ),
        "geojson_noise": (
            gdf_kmeans_utm[~gdf_kmeans_utm["is_non_noise"].astype(bool)].to_crs(CRS_GEOGRAPHIC),
            geojson_dir / "164_titik_noise_dbscan.geojson",
        ),
    }

    for key, (gdf, path) in geojson_layers.items():
        gdf.to_file(path, driver="GeoJSON")
        saved[key] = path

    return saved
