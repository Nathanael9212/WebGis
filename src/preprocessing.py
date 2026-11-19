"""
Modul Tahap 2 — Preprocessing Data Spasial
Penelitian: Penentuan Lokasi Rumah Pompa Air Pengendali Banjir Menggunakan
Klasterisasi DBSCAN dan K-Means (Studi Kasus: Kota Surabaya)

Aturan metodologis yang diimplementasikan (disepakati setelah Tahap 1):
1. CRS proyeksi seragam: WGS84 (EPSG:4326) -> UTM Zona 49S (EPSG:32749) dalam satuan meter
   dengan kolom eksplisit `easting` (X1) dan `northing` (X2).
2. Normalisasi typo `kode_hierarki` (`Sa`, `Sl` -> `SL`) agar selaras dengan `hierarki_saluran`.
3. Point-in-Polygon (PiP) ketat & deterministik:
   - Opsi 1A: Titik pada zona overlap antar-poligon ditetapkan secara deterministik
     mengikuti urutan `id_rayon` (RYN-01 s.d. RYN-07) agar mutually exclusive dan reproducible.
   - Opsi 2A: Hanya titik di dalam poligon rayon yang masuk ke analisis clustering;
     titik di luar seluruh poligon disimpan ke file audit terpisah.
4. Deduplikasi koordinat persis (Opsi 3B):
   - Menyisakan tepat 1 titik representatif per koordinat unik (prioritas hierarki SP>SS>ST>SL,
     lalu `no` terkecil) agar tidak ada jarak 0 meter pada k-distance/DBSCAN maupun bobot ganda K-Means.
   - Seluruh baris terhapus dicatat lengkap (Tipe A Identik Penuh vs Tipe B ID/Nama Berbeda).
5. Rumah pompa (acuan `83_rumah_pompa_bab3_clean.geojson`):
   - Dipisahkan eksplisit menjadi 71 Operasional Aktif (acuan analisis jarak utama,
     dengan catatan transparan 1 pasang koordinat identik PMP-013/PMP-014) dan
     12 Pengembangan Baru sebagai layer pendukung terpisah.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import NearestNeighbors


CRS_GEOGRAPHIC = "EPSG:4326"
CRS_PROJECTED = "EPSG:32749"

RAW_FILES = {
    "saluran": "21467_titik_saluran_clean.geojson",
    "rayon": "7_rayon_pematusan_clean.geojson",
    "pompa": "83_rumah_pompa_bab3_clean.geojson",
}

HIERARKI_PRIORITY = {"SP": 1, "SS": 2, "ST": 3, "SL": 4}
TYPO_HIERARKI_MAP = {"Sa": "SL", "Sl": "SL"}


def load_raw_datasets(raw_dir: Path) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Membaca ketiga dataset mentah dari data/raw/ secara read-only."""
    raw_dir = Path(raw_dir)
    paths = {k: raw_dir / v for k, v in RAW_FILES.items()}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset mentah '{name}' tidak ditemukan di: {path}")

    gdf_saluran = gpd.read_file(paths["saluran"])
    gdf_rayon = gpd.read_file(paths["rayon"])
    gdf_pompa = gpd.read_file(paths["pompa"])

    # Validasi CRS sumber wajib WGS84 / EPSG:4326
    for label, gdf in [("saluran", gdf_saluran), ("rayon", gdf_rayon), ("pompa", gdf_pompa)]:
        if gdf.crs is None:
            raise ValueError(f"Dataset '{label}' tidak memiliki informasi CRS.")
        if gdf.crs.to_epsg() != 4326:
            raise ValueError(
                f"CRS dataset '{label}' adalah {gdf.crs}, diharapkan EPSG:4326 (WGS84)."
            )

    return gdf_saluran, gdf_rayon, gdf_pompa


def validate_geometries_and_coords(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
    lon_bounds: Tuple[float, float] = (112.4, 113.0),
    lat_bounds: Tuple[float, float] = (-7.5, -7.0),
) -> Dict[str, object]:
    """Memeriksa integritas geometri dan kewajaran rentang koordinat geografis Surabaya."""
    n_total = len(gdf)
    n_null = int(gdf.geometry.isna().sum())
    n_empty = int(gdf.geometry.is_empty.sum())
    n_invalid = int((~gdf.geometry.is_valid).sum())

    minx, miny, maxx, maxy = gdf.total_bounds
    in_bounds = (
        lon_bounds[0] <= minx <= maxx <= lon_bounds[1]
        and lat_bounds[0] <= miny <= maxy <= lat_bounds[1]
    )

    n_nan_coords = 0
    if all(gdf.geom_type == "Point"):
        xs = gdf.geometry.x
        ys = gdf.geometry.y
        n_nan_coords = int((~np.isfinite(xs) | ~np.isfinite(ys)).sum())

    report = {
        "dataset": dataset_name,
        "jumlah_record": n_total,
        "crs": str(gdf.crs),
        "geometri_null": n_null,
        "geometri_kosong": n_empty,
        "geometri_tidak_valid": n_invalid,
        "koordinat_nan_inf": n_nan_coords,
        "dalam_rentang_surabaya": bool(in_bounds),
        "min_lon": round(float(minx), 6),
        "max_lon": round(float(maxx), 6),
        "min_lat": round(float(miny), 6),
        "max_lat": round(float(maxy), 6),
    }

    if n_null > 0 or n_empty > 0 or n_invalid > 0 or n_nan_coords > 0 or not in_bounds:
        raise ValueError(f"Validasi geometri gagal pada dataset '{dataset_name}': {report}")

    return report


def normalize_hierarchy_code(
    gdf_saluran: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Menormalisasi typo kolom `kode_hierarki` (Sa, Sl -> SL) sambil menyimpan log perubahan."""
    gdf = gdf_saluran.copy()
    gdf["kode_hierarki_raw"] = gdf["kode_hierarki"]

    mask_typo = gdf["kode_hierarki"].isin(TYPO_HIERARKI_MAP.keys())
    df_log = gdf.loc[
        mask_typo,
        ["no", "id_titik", "nama_saluran", "kode_hierarki_raw", "hierarki_saluran"],
    ].copy()
    df_log["kode_hierarki_baru"] = df_log["kode_hierarki_raw"].map(TYPO_HIERARKI_MAP)

    gdf["kode_hierarki"] = gdf["kode_hierarki"].replace(TYPO_HIERARKI_MAP)

    # Validasi seluruh kode kini hanya SP, SS, ST, SL
    kode_valid = set(HIERARKI_PRIORITY.keys())
    kode_aktual = set(gdf["kode_hierarki"].unique())
    if not kode_aktual.issubset(kode_valid):
        raise ValueError(f"Ditemukan kode_hierarki tidak dikenal setelah normalisasi: {kode_aktual - kode_valid}")

    return gdf, df_log.reset_index(drop=True)


def project_to_utm49s(
    gdf: gpd.GeoDataFrame,
    add_xy_cols: bool = True,
) -> gpd.GeoDataFrame:
    """
    Memproyeksikan GeoDataFrame dari EPSG:4326 ke EPSG:32749 (UTM Zona 49S, satuan meter).
    Untuk geometri Point, menambahkan kolom eksplisit `easting` (X1) dan `northing` (X2).
    """
    gdf_utm = gdf.to_crs(CRS_PROJECTED).copy()
    if add_xy_cols and all(gdf_utm.geom_type == "Point"):
        gdf_utm["easting"] = gdf_utm.geometry.x.astype(float)
        gdf_utm["northing"] = gdf_utm.geometry.y.astype(float)
    return gdf_utm


def assign_rayon_pip(
    gdf_saluran_utm: gpd.GeoDataFrame,
    gdf_rayon_utm: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict[str, int]]:
    """
    Membagi titik saluran ke dalam 7 rayon menggunakan Point-in-Polygon (EPSG:32749).
    - Opsi 1A: Jika 1 titik berada di zona overlap (>1 poligon), penetapan dilakukan
      secara deterministik berdasarkan urutan `id_rayon` (RYN-01 s.d. RYN-07).
    - Opsi 2A: Titik di luar seluruh poligon dipisahkan ke tabel audit terpisah beserta
      jarak Euclidean ke batas rayon terdekat (meter).
    """
    rayon_ref = gdf_rayon_utm[["id_rayon", "nama_rayon", "geometry"]].sort_values("id_rayon").copy()

    # Uji predikat within dan intersects untuk mendeteksi titik tepat di batas (touches)
    join_within = gpd.sjoin(
        gdf_saluran_utm,
        rayon_ref,
        how="left",
        predicate="within",
    )
    join_intersects = gpd.sjoin(
        gdf_saluran_utm[["no", "geometry"]],
        rayon_ref[["id_rayon", "geometry"]],
        how="left",
        predicate="intersects",
    )

    set_within = set(join_within.loc[join_within["id_rayon"].notna(), "no"])
    set_intersects = set(join_intersects.loc[join_intersects["id_rayon"].notna(), "no"])
    set_touches = set_intersects - set_within

    # Deteksi titik yang masuk >1 poligon (zona overlap)
    matched = join_within[join_within["id_rayon"].notna()].copy()
    overlap_map = (
        matched.groupby("no")["nama_rayon"]
        .apply(lambda s: " & ".join(sorted(s.unique())))
        .to_dict()
    )
    overlap_counts = matched.groupby("no")["id_rayon"].nunique()
    overlap_nos = set(overlap_counts[overlap_counts > 1].index)

    # Opsi 1A: Urutkan berdasarkan `no` dan `id_rayon` (RYN-01 -> RYN-07), ambil baris pertama
    matched_sorted = matched.sort_values(["no", "id_rayon"]).drop_duplicates(subset=["no"], keep="first")
    matched_sorted = matched_sorted.drop(columns=["index_right"]).copy()
    matched_sorted["is_overlap_zone"] = matched_sorted["no"].isin(overlap_nos)
    matched_sorted["overlap_rayons"] = np.where(
        matched_sorted["is_overlap_zone"],
        matched_sorted["no"].map(overlap_map),
        "",
    )
    matched_sorted["status_pip_verified"] = "Di Dalam Poligon"

    # Opsi 2A: Titik di luar seluruh poligon (audit terpisah)
    gdf_outside = gdf_saluran_utm[~gdf_saluran_utm["no"].isin(set_within)].copy()
    rayon_geoms = list(zip(rayon_ref["id_rayon"], rayon_ref["nama_rayon"], rayon_ref.geometry))

    nearest_id, nearest_name, nearest_dist = [], [], []
    second_name, second_dist = [], []
    for geom in gdf_outside.geometry:
        dists = sorted([(rid, rname, float(geom.distance(rgeom))) for rid, rname, rgeom in rayon_geoms], key=lambda x: x[2])
        nearest_id.append(dists[0][0])
        nearest_name.append(dists[0][1])
        nearest_dist.append(round(dists[0][2], 2))
        second_name.append(dists[1][1])
        second_dist.append(round(dists[1][2], 2))

    gdf_outside["status_pip_verified"] = "Di Luar Poligon"
    gdf_outside["id_rayon_terdekat"] = nearest_id
    gdf_outside["rayon_terdekat_verified"] = nearest_name
    gdf_outside["jarak_batas_terdekat_m"] = nearest_dist
    gdf_outside["rayon_kedua_terdekat"] = second_name
    gdf_outside["jarak_batas_kedua_m"] = second_dist

    # Cek kecocokan terhadap kolom `rayon_pip` bawaan
    match_raw = int((matched_sorted["nama_rayon"] == matched_sorted["rayon_pip"]).sum())

    pip_summary = {
        "total_awal": int(len(gdf_saluran_utm)),
        "dalam_poligon_within": int(len(matched_sorted)),
        "tepat_di_batas_touches": int(len(set_touches)),
        "di_luar_poligon": int(len(gdf_outside)),
        "titik_di_zona_overlap": int(len(overlap_nos)),
        "cocok_dengan_rayon_pip_sumber": match_raw,
    }

    return matched_sorted.reset_index(drop=True), gdf_outside.reset_index(drop=True), pip_summary


def deduplicate_exact_coordinates(
    gdf_in_polygon: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Menghapus duplikat koordinat persis (Opsi 3B) agar tersisa tepat 1 titik representatif
    per koordinat unik `(longitude, latitude)`.

    Aturan pemilihan titik representatif (deterministik):
    1. Prioritas hierarki saluran lebih tinggi (`SP` > `SS` > `ST` > `SL`).
    2. Nomor urut (`no`) terkecil.

    Mengembalikan:
    - gdf_clean: GeoDataFrame bersih tanpa duplikat koordinat (dengan kolom `jumlah_entri_koordinat`).
    - df_removed_log: DataFrame audit seluruh baris yang dihapus lengkap dengan klasifikasi
      Tipe A (Identik Penuh: ID & Nama sama) vs Tipe B (Koordinat sama, ID/Nama berbeda).
    """
    gdf = gdf_in_polygon.copy()
    gdf["_hierarki_rank"] = gdf["kode_hierarki"].map(HIERARKI_PRIORITY)

    # Urutkan secara deterministik sebelum deduplikasi
    gdf = gdf.sort_values(["longitude", "latitude", "_hierarki_rank", "no"]).reset_index(drop=True)

    # Klasifikasi setiap kelompok koordinat (untuk audit Tipe A vs Tipe B)
    coord_cols = ["longitude", "latitude"]
    group_sizes = gdf.groupby(coord_cols)["no"].transform("count")
    gdf["jumlah_entri_koordinat"] = group_sizes.astype(int)

    # Identifikasi baris representatif (first) dan baris terhapus (duplicated)
    mask_removed = gdf.duplicated(subset=coord_cols, keep="first")

    # Bangun log audit detail untuk baris yang dihapus
    rep_df = gdf.loc[~mask_removed, ["longitude", "latitude", "no", "id_titik", "nama_saluran"]].rename(
        columns={
            "no": "no_representatif",
            "id_titik": "id_titik_representatif",
            "nama_saluran": "nama_saluran_representatif",
        }
    )

    removed_df = gdf.loc[mask_removed].drop(columns=["geometry", "_hierarki_rank"]).merge(
        rep_df,
        on=coord_cols,
        how="left",
    )

    is_exact_same = (
        (removed_df["id_titik"] == removed_df["id_titik_representatif"])
        & (removed_df["nama_saluran"] == removed_df["nama_saluran_representatif"])
    )
    removed_df["tipe_duplikat"] = np.where(
        is_exact_same,
        "Tipe A: Identik Penuh (ID & Nama Sama)",
        "Tipe B: Koordinat Sama, ID/Nama Berbeda",
    )

    gdf_clean = gdf.loc[~mask_removed].drop(columns=["_hierarki_rank"]).sort_values("no").reset_index(drop=True)

    # Pastikan tidak ada duplikat tersisa pada (easting, northing) juga
    if gdf_clean.duplicated(subset=["easting", "northing"]).any():
        raise ValueError("Masih ditemukan duplikat pada (easting, northing) setelah deduplikasi.")

    return gdf_clean, removed_df.sort_values(["nama_rayon", "tipe_duplikat", "no"]).reset_index(drop=True)


def prepare_pump_datasets(
    gdf_pompa_utm: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Menyiapkan data rumah pompa pada EPSG:32749:
    - Menambahkan flag `is_duplicate_coord` untuk transparansi (PMP-013 & PMP-014).
    - Memisahkan secara eksplisit subset `Operasional Aktif` (71) dan `Pengembangan Baru` (12).
    """
    gdf_all = gdf_pompa_utm.copy()
    gdf_all["is_duplicate_coord"] = gdf_all.duplicated(subset=["longitude", "latitude"], keep=False)

    gdf_aktif = gdf_all[gdf_all["status_fasilitas"] == "Operasional Aktif"].reset_index(drop=True).copy()
    gdf_pengembangan = gdf_all[gdf_all["status_fasilitas"] == "Pengembangan Baru"].reset_index(drop=True).copy()

    if len(gdf_aktif) + len(gdf_pengembangan) != len(gdf_all):
        raise ValueError("Ditemukan status_fasilitas di luar 'Operasional Aktif' dan 'Pengembangan Baru'.")

    return gdf_all, gdf_aktif, gdf_pengembangan


def build_rekap_per_rayon(
    gdf_rayon_utm: gpd.GeoDataFrame,
    gdf_in_polygon: gpd.GeoDataFrame,
    gdf_out_polygon: gpd.GeoDataFrame,
    gdf_clean: gpd.GeoDataFrame,
    df_removed_log: pd.DataFrame,
    gdf_pompa_aktif: gpd.GeoDataFrame,
    gdf_pompa_pengembangan: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Menyusun tabel rekapitulasi jumlah data sebelum vs sesudah preprocessing per rayon."""
    rows = []
    for _, r in gdf_rayon_utm.sort_values("id_rayon").iterrows():
        rid = r["id_rayon"]
        rname = r["nama_rayon"]

        sub_in = gdf_in_polygon[gdf_in_polygon["nama_rayon"] == rname]
        sub_out = gdf_out_polygon[gdf_out_polygon["rayon_terdekat_verified"] == rname]
        sub_clean = gdf_clean[gdf_clean["nama_rayon"] == rname]
        sub_rem = df_removed_log[df_removed_log["nama_rayon"] == rname]

        n_in = len(sub_in)
        n_overlap = int(sub_in["is_overlap_zone"].sum())
        n_out = len(sub_out)
        n_dup_a = int((sub_rem["tipe_duplikat"].str.startswith("Tipe A")).sum())
        n_dup_b = int((sub_rem["tipe_duplikat"].str.startswith("Tipe B")).sum())
        n_clean = len(sub_clean)

        n_pa = int((gdf_pompa_aktif["rayon_pematusan"] == rname).sum())
        n_pk = int((gdf_pompa_pengembangan["rayon_pematusan"] == rname).sum())

        rows.append({
            "id_rayon": rid,
            "nama_rayon": rname,
            "luas_utm49s_km2": round(float(r.geometry.area / 1e6), 2),
            "sebelum_pip_terdekat": n_in + n_out,
            "dikeluarkan_luar_poligon": n_out,
            "setelah_pip_ketat": n_in,
            "catatan_zona_overlap": n_overlap,
            "dihapus_duplikat_tipe_a": n_dup_a,
            "dihapus_duplikat_tipe_b": n_dup_b,
            "total_dihapus_duplikat": n_dup_a + n_dup_b,
            "sesudah_preprocessing_siap_cluster": n_clean,
            "pompa_aktif": n_pa,
            "pompa_pengembangan": n_pk,
        })

    df_rekap = pd.DataFrame(rows)
    total_row = {
        "id_rayon": "TOTAL",
        "nama_rayon": "Seluruh Kota Surabaya (7 Rayon)",
        "luas_utm49s_km2": round(float(df_rekap["luas_utm49s_km2"].sum()), 2),
    }
    for col in df_rekap.columns[3:]:
        total_row[col] = int(df_rekap[col].sum())

    return pd.concat([df_rekap, pd.DataFrame([total_row])], ignore_index=True)


def verify_processed_outputs(
    gdf_clean: gpd.GeoDataFrame,
    gdf_rayon_utm: gpd.GeoDataFrame,
    gdf_pompa_aktif: gpd.GeoDataFrame,
    gdf_pompa_pengembangan: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Validasi otomatis pasca-preprocessing:
    - CRS seluruh layer harus EPSG:32749.
    - Tidak ada missing value pada kolom kunci clustering.
    - Tidak ada duplikat koordinat (easting, northing) di saluran bersih.
    - Jarak tetangga terdekat (NN-1) di setiap rayon harus > 0 meter.
    """
    for label, gdf in [
        ("saluran_clean", gdf_clean),
        ("rayon_utm49s", gdf_rayon_utm),
        ("pompa_aktif", gdf_pompa_aktif),
        ("pompa_pengembangan", gdf_pompa_pengembangan),
    ]:
        if gdf.crs is None or gdf.crs.to_epsg() != 32749:
            raise AssertionError(f"CRS '{label}' bukan EPSG:32749: {gdf.crs}")

    required_cols = ["no", "id_titik", "kode_hierarki", "id_rayon", "nama_rayon", "easting", "northing"]
    for col in required_cols:
        if gdf_clean[col].isna().any():
            raise AssertionError(f"Masih ditemukan nilai kosong pada kolom '{col}' di saluran bersih.")

    if gdf_clean["no"].duplicated().any():
        raise AssertionError("Ditemukan duplikat `no` (pelanggaran mutually exclusive antar-rayon).")

    if gdf_clean.duplicated(subset=["easting", "northing"]).any():
        raise AssertionError("Masih ditemukan duplikat koordinat (easting, northing) di saluran bersih.")

    # Verifikasi jarak minimum antar-titik (NN-1) per rayon > 0 meter
    nn_rows = []
    for rname, grp in gdf_clean.groupby("nama_rayon"):
        coords = grp[["easting", "northing"]].to_numpy(dtype=float)
        if len(coords) < 10:
            raise AssertionError(f"Jumlah titik di {rname} terlalu sedikit ({len(coords)} < 10).")
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(coords)
        dists, _ = nn.kneighbors(coords)
        min_nn1 = float(dists[:, 1].min())
        median_nn1 = float(np.median(dists[:, 1]))
        if min_nn1 <= 0.0:
            raise AssertionError(f"Ditemukan jarak tetangga terdekat 0 meter di {rname}.")
        nn_rows.append({
            "nama_rayon": rname,
            "jumlah_titik_unik": len(coords),
            "jarak_nn1_min_m": round(min_nn1, 4),
            "jarak_nn1_median_m": round(median_nn1, 2),
        })

    return pd.DataFrame(nn_rows).sort_values("nama_rayon").reset_index(drop=True)


def save_processed_outputs(
    processed_dir: Path,
    tables_dir: Path,
    gdf_clean: gpd.GeoDataFrame,
    gdf_outside: gpd.GeoDataFrame,
    gdf_rayon_utm: gpd.GeoDataFrame,
    gdf_pompa_all: gpd.GeoDataFrame,
    gdf_pompa_aktif: gpd.GeoDataFrame,
    gdf_pompa_pengembangan: gpd.GeoDataFrame,
    df_typo_log: pd.DataFrame,
    df_removed_log: pd.DataFrame,
    df_rekap: pd.DataFrame,
) -> Dict[str, Path]:
    """Menyimpan seluruh dataset terproses ke data/processed/ dan tabel audit ke outputs/tables/."""
    processed_dir = Path(processed_dir)
    tables_dir = Path(tables_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    saved: Dict[str, Path] = {}

    # 1. File spasial GeoPackage (.gpkg) — menjaga CRS EPSG:32749 secara native & presisi penuh
    gpkg_targets = {
        "saluran_clean_gpkg": (gdf_clean, processed_dir / "saluran_clean_utm49s.gpkg"),
        "saluran_luar_poligon_gpkg": (gdf_outside, processed_dir / "saluran_luar_poligon_audit_utm49s.gpkg"),
        "rayon_utm49s_gpkg": (gdf_rayon_utm, processed_dir / "rayon_pematusan_utm49s.gpkg"),
        "pompa_all_gpkg": (gdf_pompa_all, processed_dir / "rumah_pompa_all_utm49s.gpkg"),
        "pompa_aktif_gpkg": (gdf_pompa_aktif, processed_dir / "rumah_pompa_aktif_utm49s.gpkg"),
        "pompa_pengembangan_gpkg": (gdf_pompa_pengembangan, processed_dir / "rumah_pompa_pengembangan_utm49s.gpkg"),
    }
    for key, (gdf, path) in gpkg_targets.items():
        gdf.to_file(path, driver="GPKG")
        saved[key] = path

    # 2. File CSV tabular di data/processed/ (untuk pemuatan cepat & inspeksi Excel)
    csv_processed = {
        "saluran_clean_csv": (gdf_clean.drop(columns="geometry"), processed_dir / "saluran_clean_utm49s.csv"),
        "pompa_aktif_csv": (gdf_pompa_aktif.drop(columns="geometry"), processed_dir / "rumah_pompa_aktif_utm49s.csv"),
        "pompa_pengembangan_csv": (
            gdf_pompa_pengembangan.drop(columns="geometry"),
            processed_dir / "rumah_pompa_pengembangan_utm49s.csv",
        ),
    }
    for key, (df, path) in csv_processed.items():
        df.to_csv(path, index=False)
        saved[key] = path

    # 3. Tabel audit & rekapitulasi di outputs/tables/ (untuk dokumentasi Bab III & IV)
    csv_tables = {
        "rekap_preprocessing_csv": (df_rekap, tables_dir / "02_rekap_preprocessing_per_rayon.csv"),
        "audit_typo_hierarki_csv": (df_typo_log, tables_dir / "02_audit_typo_kode_hierarki.csv"),
        "audit_luar_poligon_csv": (gdf_outside.drop(columns="geometry"), tables_dir / "02_audit_titik_luar_poligon.csv"),
        "audit_duplikat_dihapus_csv": (df_removed_log, tables_dir / "02_audit_duplikat_koordinat_dihapus.csv"),
    }
    for key, (df, path) in csv_tables.items():
        df.to_csv(path, index=False)
        saved[key] = path

    return saved
