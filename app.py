"""
Prototipe WebGIS Penelitian Skripsi — Streamlit + Folium
Judul: Penentuan Lokasi Rumah Pompa Air Pengendali Banjir Menggunakan
       Klasterisasi DBSCAN dan K-Means (Studi Kasus: Kota Surabaya)

Jalankan aplikasi dari root proyek:
    .venv\\Scripts\\python.exe -m streamlit run app.py

Catatan metodologis:
- Aplikasi ini hanya memuat hasil olahan terverifikasi dari `data/processed/`
  dan `outputs/` menggunakan `@st.cache_data` (tidak menjalankan ulang
  klasterisasi saat interaksi UI agar ringan dan stabil untuk demo sidang).
- Tidak terdapat fitur buffer 1,5 km maupun snapping ke jalan/saluran.
- Jarak yang ditampilkan adalah jarak lurus Euclidean (straight-line distance)
  pada proyeksi UTM Zona 49S (EPSG:32749) terhadap 71 rumah pompa aktif.
- Sebanyak 12 rumah pompa Pengembangan Baru ditampilkan sebagai layer terpisah.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
GEOJSON_DIR = PROJECT_ROOT / "outputs" / "geojson"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# Palet warna subcluster K-Means (konsisten dengan Notebook 04 & 05)
SUBCLUSTER_PALETTE = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD", "#17BECF",
    "#E377C2", "#8C564B", "#BCBD22", "#393B79", "#AD494A",
]

DBSCAN_ROLE_COLORS = {
    "Core": "#1F77B4",
    "Border": "#FF9800",
    "Noise": "#D62728",
}

SURABAYA_CENTER = [-7.2756, 112.7120]


@st.cache_data(show_spinner="Memuat data spasial dan tabel hasil penelitian...")
def load_all_artifacts() -> Dict[str, object]:
    """Memuat seluruh hasil pra-hitung dari data/processed/ dan outputs/ secara ter-cache."""
    gdf_rayon = gpd.read_file(GEOJSON_DIR / "7_rayon_pematusan_ringkasan.geojson").sort_values("id_rayon").reset_index(drop=True)
    gdf_centroid = gpd.read_file(GEOJSON_DIR / "31_centroid_kandidat_jarak.geojson").sort_values("peringkat_jarak_terjauh").reset_index(drop=True)
    gdf_links = gpd.read_file(GEOJSON_DIR / "31_garis_jarak_pompa_aktif.geojson")
    gdf_pompa_aktif = gpd.read_file(GEOJSON_DIR / "71_rumah_pompa_aktif.geojson")
    gdf_pompa_kembang = gpd.read_file(GEOJSON_DIR / "12_rumah_pompa_pengembangan.geojson")
    gdf_noise = gpd.read_file(GEOJSON_DIR / "164_titik_noise_dbscan.geojson")

    # Data 21.272 titik saluran bersih hasil DBSCAN + K-Means (CSV ringan untuk Canvas rendering)
    saluran_cols = [
        "no", "id_titik", "nama_saluran", "kode_hierarki", "id_rayon", "nama_rayon",
        "longitude", "latitude", "easting", "northing",
        "dbscan_role", "dbscan_cluster", "is_non_noise", "kmeans_cluster", "centroid_id",
    ]
    df_saluran = pd.read_csv(PROCESSED_DIR / "saluran_kmeans_utm49s.csv", usecols=saluran_cols)

    # Tabel ringkasan & evaluasi
    df_dbscan_sum = pd.read_csv(TABLES_DIR / "03_dbscan_ringkasan_per_rayon.csv")
    df_kmeans_sum = pd.read_csv(TABLES_DIR / "04_kmeans_ringkasan_per_rayon.csv")
    df_eval_rayon = pd.read_csv(TABLES_DIR / "05_evaluasi_kmeans_per_rayon.csv")
    df_eval_sub = pd.read_csv(TABLES_DIR / "05_evaluasi_silhouette_per_subcluster.csv")
    df_jarak_sum = pd.read_csv(TABLES_DIR / "06_ringkasan_jarak_per_rayon.csv")
    df_dev_ctx = pd.read_csv(TABLES_DIR / "06_konteks_sekunder_pompa_pengembangan.csv")

    return {
        "rayon": gdf_rayon,
        "centroid": gdf_centroid,
        "links": gdf_links,
        "pompa_aktif": gdf_pompa_aktif,
        "pompa_kembang": gdf_pompa_kembang,
        "noise": gdf_noise,
        "saluran": df_saluran,
        "dbscan_sum": df_dbscan_sum,
        "kmeans_sum": df_kmeans_sum,
        "eval_rayon": df_eval_rayon,
        "eval_sub": df_eval_sub,
        "jarak_sum": df_jarak_sum,
        "dev_ctx": df_dev_ctx,
    }


def _star_div_icon(label: str, fill_color: str = "#FFD700") -> folium.DivIcon:
    """Ikon SVG bintang kuning berbingkai hitam agar centroid kandidat sangat jelas di peta."""
    svg = f"""
    <div style="position: relative; width: 32px; height: 32px; transform: translate(-16px, -16px);">
      <svg viewBox="0 0 32 32" width="30" height="30" style="filter: drop-shadow(0 1px 2px rgba(0,0,0,0.45));">
        <polygon points="16,2 20.1,11.2 30,12.2 22.5,18.8 24.7,28.8 16,23.6 7.3,28.8 9.5,18.8 2,12.2 11.9,11.2"
                 fill="{fill_color}" stroke="#111111" stroke-width="1.8"/>
      </svg>
      <div style="position:absolute; top:25px; left:50%; transform:translateX(-50%);
                  background:rgba(255,255,255,0.92); border:1px solid #333; border-radius:3px;
                  padding:0 3px; font-size:9px; font-weight:700; color:#111; white-space:nowrap;">
        {label}
      </div>
    </div>
    """
    return folium.DivIcon(html=svg, icon_size=(0, 0), icon_anchor=(0, 0))


def _pump_div_icon(is_active: bool) -> folium.DivIcon:
    """Ikon SVG pembeda tegas: segitiga merah (71 Aktif) vs kotak ungu (12 Pengembangan)."""
    if is_active:
        svg = """
        <div style="transform: translate(-9px, -9px);">
          <svg viewBox="0 0 20 20" width="18" height="18">
            <polygon points="10,2 18,17 2,17" fill="#D62728" stroke="#FFFFFF" stroke-width="1.6"/>
          </svg>
        </div>
        """
    else:
        svg = """
        <div style="transform: translate(-8px, -8px);">
          <svg viewBox="0 0 18 18" width="16" height="16">
            <rect x="2" y="2" width="14" height="14" rx="2" fill="#8E44AD" stroke="#FFFFFF" stroke-width="1.6"/>
          </svg>
        </div>
        """
    return folium.DivIcon(html=svg, icon_size=(0, 0), icon_anchor=(0, 0))


def _build_legend_html(
    color_mode: str,
    show_saluran: bool,
    n_cen: int,
    n_pmp_akt: int,
    n_pmp_dev: int,
    n_noise: int,
) -> str:
    """Legenda peta mengambang di pojok kiri bawah kanvas Folium."""
    saluran_note = ""
    if show_saluran:
        if color_mode == "Status DBSCAN (Core / Border / Noise)":
            saluran_note = (
                "<div style='margin-top:5px; padding-top:4px; border-top:1px solid #ddd;'>"
                "<b>Titik Saluran (DBSCAN):</b><br>"
                "<span style='color:#1F77B4;'>&#9679;</span> Core &nbsp;"
                "<span style='color:#FF9800;'>&#9679;</span> Border &nbsp;"
                "<span style='color:#D62728;'>&#9679;</span> Noise"
                "</div>"
            )
        else:
            saluran_note = (
                "<div style='margin-top:5px; padding-top:4px; border-top:1px solid #ddd;'>"
                "<b>Titik Saluran (K-Means):</b><br>"
                "Warna membedakan Subcluster C1–CK per rayon"
                "</div>"
            )

    return f"""
    <div style="
        position: fixed;
        bottom: 18px; left: 18px; z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #888; border-radius: 6px;
        padding: 9px 12px; font-size: 11.5px; line-height: 1.45;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2); max-width: 270px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
      <div style="font-weight:700; margin-bottom:4px; color:#111;">Legenda Peta WebGIS</div>
      <div><span style="color:#E6B800; font-size:14px;">&#9733;</span> Centroid Kandidat Awal K-Means ({n_cen})</div>
      <div><span style="color:#D62728; font-size:13px;">&#9650;</span> Rumah Pompa Operasional Aktif ({n_pmp_akt})</div>
      <div><span style="color:#8E44AD; font-size:12px;">&#9632;</span> Pengembangan Baru — Layer Terpisah ({n_pmp_dev})</div>
      <div><span style="color:#D62728;">&#9675;</span> Titik Noise DBSCAN ({n_noise})</div>
      <div><span style="color:#1F77B4; font-weight:700;">&#8213;</span> Garis Jarak Lurus (Sama Rayon)</div>
      <div><span style="color:#D95F02; font-weight:700;">- - -</span> Garis Jarak Lurus (Lintas Rayon)</div>
      {saluran_note}
    </div>
    """


def build_webgis_map(
    data: Dict[str, object],
    selected_rayon: str,
    basemap: str,
    show_rayon: bool,
    show_saluran: bool,
    saluran_color_mode: str,
    show_noise: bool,
    show_centroid: bool,
    show_links: bool,
    show_pompa_aktif: bool,
    show_pompa_kembang: bool,
) -> folium.Map:
    """Membangun objek peta interaktif Folium dengan prefer_canvas=True agar ringan."""
    gdf_rayon: gpd.GeoDataFrame = data["rayon"]
    gdf_centroid: gpd.GeoDataFrame = data["centroid"]
    gdf_links: gpd.GeoDataFrame = data["links"]
    gdf_pompa_aktif: gpd.GeoDataFrame = data["pompa_aktif"]
    gdf_pompa_kembang: gpd.GeoDataFrame = data["pompa_kembang"]
    gdf_noise: gpd.GeoDataFrame = data["noise"]
    df_saluran: pd.DataFrame = data["saluran"]

    is_all = (selected_rayon == "Semua 7 Rayon")

    # Filter data sesuai pilihan wilayah
    cen_view = gdf_centroid if is_all else gdf_centroid[gdf_centroid["nama_rayon"] == selected_rayon]
    lnk_view = gdf_links if is_all else gdf_links[gdf_links["nama_rayon"] == selected_rayon]
    noise_view = gdf_noise if is_all else gdf_noise[gdf_noise["nama_rayon"] == selected_rayon]
    sal_view = df_saluran if is_all else df_saluran[df_saluran["nama_rayon"] == selected_rayon]

    # Untuk rumah pompa aktif: pada mode 1 rayon, tetap tampilkan pompa lintas rayon yang menjadi target terdekat!
    if is_all:
        pmp_akt_view = gdf_pompa_aktif
        pmp_dev_view = gdf_pompa_kembang
    else:
        linked_pump_ids = set()
        for raw_ids in cen_view["id_pompa_aktif_terdekat"].dropna():
            for pid in str(raw_ids).split(" / "):
                linked_pump_ids.add(pid.strip())
        pmp_akt_view = gdf_pompa_aktif[
            (gdf_pompa_aktif["rayon_pematusan"] == selected_rayon)
            | (gdf_pompa_aktif["id_pompa"].isin(linked_pump_ids))
        ]
        pmp_dev_view = gdf_pompa_kembang[gdf_pompa_kembang["rayon_pematusan"] == selected_rayon]

    if basemap == "Esri Light Gray (Bersih)":
        m = folium.Map(
            location=SURABAYA_CENTER,
            zoom_start=12,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ",
            prefer_canvas=True,
            control_scale=True,
        )
    else:
        m = folium.Map(
            location=SURABAYA_CENTER,
            zoom_start=12,
            tiles="OpenStreetMap",
            prefer_canvas=True,
            control_scale=True,
        )

    # 1. Layer Batas Rayon Pematusan (Jika memilih 1 rayon: sembunyikan 6 rayon lainnya total)
    if show_rayon:
        rayon_display = gdf_rayon if is_all else gdf_rayon[gdf_rayon["nama_rayon"] == selected_rayon]
        fg_rayon = folium.FeatureGroup(name="Batas 7 Rayon Pematusan" if is_all else f"Batas {selected_rayon}")
        for _, row in rayon_display.iterrows():
            rname = row["nama_rayon"]
            fill_op = 0.16 if not is_all else 0.10
            line_w = 2.4 if not is_all else 1.4
            edge_col = "#0F172A" if not is_all else "#475569"

            tooltip_html = (
                f"<b>{row['id_rayon']} — {rname}</b><br>"
                f"Luas: {row['luas_km2']:.2f} km² | Titik Bersih: {int(row['jumlah_titik']):,}<br>"
                f"DBSCAN ε: {row['eps_digunakan_m']:.2f} m | Noise: {int(row['jumlah_noise'])} ({row['persen_noise']:.2f}%)<br>"
                f"K-Means K: {int(row['k_terpilih'])} | SC: {row['silhouette_score']:.4f} | DBI: {row['davies_bouldin_index']:.4f}<br>"
                f"Pompa Aktif di Rayon: {int(row['jumlah_pompa_aktif_di_rayon'])} | Rerata Jarak: {row['jarak_rerata_m']:,.1f} m"
            )
            folium.GeoJson(
                data=row.geometry.__geo_interface__,
                style_function=lambda _, fc=row["warna_qgis"], fo=fill_op, ec=edge_col, lw=line_w: {
                    "fillColor": fc,
                    "fillOpacity": fo,
                    "color": ec,
                    "weight": lw,
                },
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
            ).add_to(fg_rayon)
        fg_rayon.add_to(m)

    # 2. Layer Titik Saluran Drainase (Canvas CircleMarker)
    if show_saluran and len(sal_view) > 0:
        fg_sal = folium.FeatureGroup(name=f"Titik Saluran ({len(sal_view):,} titik)")
        use_dbscan_mode = (saluran_color_mode == "Status DBSCAN (Core / Border / Noise)")

        for row in sal_view.itertuples(index=False):
            if use_dbscan_mode:
                col = DBSCAN_ROLE_COLORS.get(row.dbscan_role, "#666666")
                rad = 2.6 if row.dbscan_role == "Core" else 3.8
            else:
                if int(row.kmeans_cluster) == -1:
                    col = "#D62728"
                    rad = 3.2
                else:
                    col = SUBCLUSTER_PALETTE[(int(row.kmeans_cluster) - 1) % len(SUBCLUSTER_PALETTE)]
                    rad = 2.6

            folium.CircleMarker(
                location=[float(row.latitude), float(row.longitude)],
                radius=rad,
                color=col,
                fill=True,
                fill_color=col,
                fill_opacity=0.68,
                weight=0.4,
            ).add_to(fg_sal)
        fg_sal.add_to(m)

    # 3. Layer Titik Noise DBSCAN (164 titik)
    if show_noise and len(noise_view) > 0:
        fg_noise = folium.FeatureGroup(name=f"Titik Noise DBSCAN ({len(noise_view)} titik)")
        for _, row in noise_view.iterrows():
            pop_n = (
                f"<b>Titik Noise DBSCAN</b><br>"
                f"ID: {row['id_titik']} ({row['kode_hierarki']})<br>"
                f"Saluran: {row['nama_saluran']}<br>"
                f"Rayon: {row['nama_rayon']}<br>"
                f"k-distance (k=4): {row['k_distance_m']:.1f} m (ε rayon: {row['dbscan_eps_m']:.1f} m)"
            )
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=4.5,
                color="#7F0000",
                fill=True,
                fill_color="#FF3B30",
                fill_opacity=0.9,
                weight=1.0,
                popup=folium.Popup(pop_n, max_width=280),
            ).add_to(fg_noise)
        fg_noise.add_to(m)

    # 4. Layer Garis Jarak Lurus Euclidean ke Rumah Pompa Aktif Terdekat
    if show_links and len(lnk_view) > 0:
        fg_links = folium.FeatureGroup(name="Garis Jarak Lurus ke Pompa Aktif Terdekat")
        for _, row in lnk_view.iterrows():
            coords_latlon = [(pt[1], pt[0]) for pt in row.geometry.coords]
            is_cross = (row["status_lintas_rayon"] == "Lintas Rayon")
            tip = (
                f"<b>{row['centroid_id']} → {row['nama_pompa_aktif_terdekat']}</b><br>"
                f"Jarak Lurus Euclidean: <b>{row['jarak_m']:,.2f} m</b> ({row['status_lintas_rayon']})"
            )
            folium.PolyLine(
                locations=coords_latlon,
                color="#D95F02" if is_cross else "#1F77B4",
                weight=2.4,
                opacity=0.88,
                dash_array="7, 6" if is_cross else None,
                tooltip=folium.Tooltip(tip, sticky=True),
            ).add_to(fg_links)
        fg_links.add_to(m)

    # 5. Layer 71 Rumah Pompa Operasional Aktif (Acuan Utama)
    if show_pompa_aktif and len(pmp_akt_view) > 0:
        fg_akt = folium.FeatureGroup(name="Rumah Pompa Operasional Aktif (71)")
        for _, row in pmp_akt_view.iterrows():
            dup_note = (
                "<br><span style='color:#B00020;'><i>Catatan data sumber: koordinat identik dengan pasangan PMP-013/PMP-014.</i></span>"
                if bool(row.get("is_duplicate_coord", False))
                else ""
            )
            pop_p = (
                f"<b>{row['nama_stasiun_pompa']}</b> ({row['id_pompa']})<br>"
                f"Status: <b style='color:#C62828;'>Operasional Aktif (Acuan Utama)</b><br>"
                f"Rayon: {row['rayon_pematusan']}<br>"
                f"WGS84: {row['latitude']:.6f}, {row['longitude']:.6f}<br>"
                f"UTM 49S: E {row['easting']:,.2f} m, N {row['northing']:,.2f} m"
                f"{dup_note}"
            )
            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                icon=_pump_div_icon(is_active=True),
                tooltip=f"Pompa Aktif: {row['nama_stasiun_pompa']} ({row['id_pompa']}) | Lat: {row['latitude']:.5f}, Lng: {row['longitude']:.5f}",
                popup=folium.Popup(pop_p, max_width=310),
            ).add_to(fg_akt)
        fg_akt.add_to(m)

    # 6. Layer 12 Rumah Pompa Pengembangan Baru (Layer Pendukung Terpisah)
    if show_pompa_kembang and len(pmp_dev_view) > 0:
        fg_dev = folium.FeatureGroup(name="Rumah Pompa Pengembangan Baru — Layer Terpisah (12)")
        for _, row in pmp_dev_view.iterrows():
            pop_d = (
                f"<b>{row['nama_stasiun_pompa']}</b> ({row['id_pompa']})<br>"
                f"Status: <b style='color:#6A1B9A;'>Pengembangan Baru (Layer Pendukung Terpisah)</b><br>"
                f"Rayon: {row['rayon_pematusan']}<br>"
                f"WGS84: {row['latitude']:.6f}, {row['longitude']:.6f}<br>"
                f"UTM 49S: E {row['easting']:,.2f} m, N {row['northing']:,.2f} m<br>"
                f"<i>Tidak dicampur ke perhitungan jarak utama 71 rumah pompa aktif.</i>"
            )
            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                icon=_pump_div_icon(is_active=False),
                tooltip=f"Pengembangan Baru: {row['nama_stasiun_pompa']} ({row['id_pompa']}) | Lat: {row['latitude']:.5f}, Lng: {row['longitude']:.5f}",
                popup=folium.Popup(pop_d, max_width=320),
            ).add_to(fg_dev)
        fg_dev.add_to(m)

    # 7. Layer Centroid Kandidat Awal K-Means++ (Popup Detail Lengkap)
    if show_centroid and len(cen_view) > 0:
        fg_cen = folium.FeatureGroup(name=f"Centroid Kandidat Awal K-Means ({len(cen_view)})")
        for _, row in cen_view.iterrows():
            short_lbl = str(row["centroid_id"]).replace("RYN-0", "R")
            mikro_note = (
                f"<br><span style='color:#B45309;'><i>Catatan kritis: subcluster mikro ({int(row['jumlah_titik'])} titik terpisah jauh).</i></span>"
                if int(row["jumlah_titik"]) <= 10
                else ""
            )
            popup_html = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 12px; line-height: 1.45;">
              <div style="font-size:13.5px; font-weight:700; color:#1A365D; border-bottom:1px solid #ccc; padding-bottom:4px; margin-bottom:6px;">
                Centroid Kandidat Awal: {row['centroid_id']}
              </div>
              <table style="border-collapse:collapse; width:100%;">
                <tr><td style="padding:2px 6px 2px 0;"><b>Rayon</b></td><td>: {row['nama_rayon']} (Subcluster C{int(row['kmeans_cluster'])})</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>Anggota</b></td><td>: <b>{int(row['jumlah_titik']):,} titik</b> ({row['persen_non_noise_rayon']:.2f}% non-noise rayon)</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>Hierarki</b></td><td>: SP={int(row['n_sp'])}, SS={int(row['n_ss'])}, ST={int(row['n_st'])}, SL={int(row['n_sl'])}</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>WGS84 (Lat, Lng)</b></td><td>: {row['latitude']:.6f}, {row['longitude']:.6f}</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>UTM 49S (E, N)</b></td><td>: {row['easting']:,.2f} m, {row['northing']:,.2f} m</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>Pompa Aktif Terdekat</b></td><td>: <b>{row['nama_pompa_aktif_terdekat']}</b> ({row['id_pompa_aktif_terdekat']} — {row['rayon_pompa_aktif_terdekat']})</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>Jarak Lurus Euclidean</b></td><td>: <b style="color:#C62828;">{row['jarak_pompa_aktif_terdekat_m']:,.2f} m</b> ({row['jarak_pompa_aktif_terdekat_km']:.3f} km)</td></tr>
                <tr><td style="padding:2px 6px 2px 0;"><b>Relasi Wilayah</b></td><td>: {row['status_lintas_rayon']} (Peringkat terjauh #{int(row['peringkat_jarak_terjauh'])}/31)</td></tr>
              </table>
              {mikro_note}
            </div>
            """
            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                icon=_star_div_icon(label=short_lbl, fill_color="#FFD700"),
                tooltip=(
                    f"Kandidat Centroid {row['centroid_id']} ({row['nama_rayon']}) | "
                    f"Lat: {row['latitude']:.5f}, Lng: {row['longitude']:.5f} | "
                    f"Anggota: {int(row['jumlah_titik']):,} ttk | "
                    f"Pompa Terdekat: {row['nama_pompa_aktif_terdekat']} ({row['jarak_pompa_aktif_terdekat_m']:,.0f} m)"
                ),
                popup=folium.Popup(popup_html, max_width=360),
            ).add_to(fg_cen)
        fg_cen.add_to(m)

    # Auto-zoom rapat ke rayon terpilih atau Surabaya makro
    target_bounds_gdf = gdf_rayon if is_all else gdf_rayon[gdf_rayon["nama_rayon"] == selected_rayon]
    minx, miny, maxx, maxy = target_bounds_gdf.total_bounds
    pad = (16, 16) if is_all else (8, 8)
    m.fit_bounds([[miny, minx], [maxy, maxx]], padding=pad)

    # Tambahkan legenda HTML mengambang
    m.get_root().html.add_child(
        folium.Element(
            _build_legend_html(
                color_mode=saluran_color_mode,
                show_saluran=show_saluran,
                n_cen=len(cen_view),
                n_pmp_akt=len(pmp_akt_view),
                n_pmp_dev=len(pmp_dev_view),
                n_noise=len(noise_view),
            )
        )
    )
    return m


@st.cache_data(show_spinner="Merender peta interaktif...")
def render_cached_map_html(
    selected_rayon: str,
    basemap: str,
    show_rayon: bool,
    show_saluran: bool,
    saluran_color_mode: str,
    show_noise: bool,
    show_centroid: bool,
    show_links: bool,
    show_pompa_aktif: bool,
    show_pompa_kembang: bool,
) -> str:
    """Menghasilkan dan meng-cache HTML peta Folium berdasarkan konfigurasi kontrol sidebar."""
    data = load_all_artifacts()
    fmap = build_webgis_map(
        data=data,
        selected_rayon=selected_rayon,
        basemap=basemap,
        show_rayon=show_rayon,
        show_saluran=show_saluran,
        saluran_color_mode=saluran_color_mode,
        show_noise=show_noise,
        show_centroid=show_centroid,
        show_links=show_links,
        show_pompa_aktif=show_pompa_aktif,
        show_pompa_kembang=show_pompa_kembang,
    )
    return fmap._repr_html_()


def main() -> None:
    st.set_page_config(
        page_title="WebGIS Kandidat Lokasi Rumah Pompa Surabaya",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data = load_all_artifacts()
    gdf_rayon: gpd.GeoDataFrame = data["rayon"]
    gdf_centroid: gpd.GeoDataFrame = data["centroid"]
    df_dbscan_sum: pd.DataFrame = data["dbscan_sum"]
    df_kmeans_sum: pd.DataFrame = data["kmeans_sum"]
    df_eval_rayon: pd.DataFrame = data["eval_rayon"]
    df_eval_sub: pd.DataFrame = data["eval_sub"]
    df_jarak_sum: pd.DataFrame = data["jarak_sum"]
    df_dev_ctx: pd.DataFrame = data["dev_ctx"]

    # --- HEADER ---
    st.title("Prototipe WebGIS — Kandidat Awal Lokasi Rumah Pompa Kota Surabaya")
    st.caption(
        "Klasterisasi Spasial Dua Tahap **DBSCAN (MinPts=4, ε Adaptif Kneedle)** dan **K-Means++ (Elbow Method)** "
        "pada 21.272 Titik Inventarisasi Saluran Drainase di 7 Rayon Pematusan (Proyeksi UTM Zona 49S / EPSG:32749)."
    )

    # --- SIDEBAR PANEL KONTROL ---
    with st.sidebar:
        st.header("Panel Kontrol Peta")

        rayon_options = ["Semua 7 Rayon"] + gdf_rayon["nama_rayon"].tolist()
        selected_rayon = st.selectbox("Pilih Wilayah Rayon Pematusan", rayon_options, index=0)
        is_all = (selected_rayon == "Semua 7 Rayon")

        basemap = st.selectbox(
            "Peta Dasar (Basemap)",
            ["OpenStreetMap", "Esri Light Gray (Bersih)"],
            index=0,
        )

        st.markdown("---")
        st.subheader("Pengaturan Layer Peta")

        show_rayon = st.checkbox("Batas 7 Rayon Pematusan", value=True)

        # Smart default: mode 'Semua 7 Rayon' default FALSE (tampilan makro bersih), mode 1 rayon default TRUE
        if "last_selected_rayon" not in st.session_state:
            st.session_state["last_selected_rayon"] = selected_rayon
            st.session_state["show_saluran"] = not is_all
        elif st.session_state["last_selected_rayon"] != selected_rayon:
            st.session_state["last_selected_rayon"] = selected_rayon
            st.session_state["show_saluran"] = not is_all

        show_saluran = st.checkbox(
            "Titik Saluran Drainase (21.272 titik)",
            key="show_saluran",
            help="Pada mode 'Semua 7 Rayon', layer ini non-aktif default agar peta makro bersih. "
                 "Otomatis aktif saat memilih 1 rayon spesifik, atau dapat dicentang manual.",
        )
        saluran_color_mode = st.radio(
            "Pewarnaan Titik Saluran",
            ["Subcluster K-Means", "Status DBSCAN (Core / Border / Noise)"],
            index=0,
            disabled=not show_saluran,
        )

        show_noise = st.checkbox("Titik Noise DBSCAN (164 titik)", value=True)
        show_centroid = st.checkbox("Centroid Kandidat Awal K-Means (31)", value=True)
        show_links = st.checkbox(
            "Garis Jarak Lurus ke Pompa Aktif Terdekat",
            value=True,
            disabled=not show_centroid,
        )

        st.markdown("---")
        st.subheader("Layer Rumah Pompa")
        show_pompa_aktif = st.checkbox("71 Rumah Pompa Operasional Aktif (Acuan Utama)", value=True)
        show_pompa_kembang = st.checkbox(
            "12 Pengembangan Baru (Layer Pendukung Terpisah)",
            value=True,
            help="Ditampilkan terpisah sebagai informasi konteks pendukung, tidak dicampur ke analisis jarak utama.",
        )

        st.markdown("---")
        st.info(
            "**Catatan Metodologi:**\n"
            "- Jarak dihitung sebagai **jarak lurus Euclidean** (*straight-line*) dalam meter pada `EPSG:32749`.\n"
            "- Tanpa fitur *buffer* 1,5 km maupun *snapping* ke jaringan jalan/saluran."
        )

    # --- KANVAS PETA INTERAKTIF FOLIUM (LANGSUNG DI BAWAH HEADER) ---
    map_html = render_cached_map_html(
        selected_rayon=selected_rayon,
        basemap=basemap,
        show_rayon=show_rayon,
        show_saluran=show_saluran,
        saluran_color_mode=saluran_color_mode,
        show_noise=show_noise,
        show_centroid=show_centroid,
        show_links=show_links,
        show_pompa_aktif=show_pompa_aktif,
        show_pompa_kembang=show_pompa_kembang,
    )
    st.iframe(map_html, height=680, width="stretch")

    # --- TATA LETAK BAWAH PERSIS WIREFRAME (2 KOLOM: EVALUASI & TABEL CENTROID) ---
    st.markdown("---")

    if is_all:
        ev_row = df_eval_rayon[df_eval_rayon["id_rayon"] == "MEAN"].iloc[0]
        sc_label = "Silhouette Coefficient (SC)"
        dbi_label = "Davies-Bouldin Index (DBI)"
        sc_delta = f"Rata-rata 7 Rayon ({ev_row['kategori_silhouette']})"
        dbi_delta = "Rata-rata 7 Rayon (Unweighted)"
    else:
        ev_row = df_eval_rayon[df_eval_rayon["nama_rayon"] == selected_rayon].iloc[0]
        r_short = selected_rayon.replace("Rayon ", "")
        sc_label = f"Silhouette Coefficient ({r_short})"
        dbi_label = f"Davies-Bouldin Index ({r_short})"
        sc_delta = f"Subcluster K={int(ev_row['k_terpilih'])} ({ev_row['kategori_silhouette']})"
        dbi_delta = f"Subcluster K={int(ev_row['k_terpilih'])}"

    col_eval, col_table = st.columns([0.28, 0.72], gap="large")

    with col_eval:
        st.subheader("📊 Evaluasi Klaster")
        st.metric(
            label=sc_label,
            value=f"{ev_row['silhouette_score']:.4f}",
            delta=sc_delta,
            delta_color="off",
            delta_arrow="off",
            help="Silhouette Coefficient: rentang -1 s.d. +1 (skor lebih tinggi menunjukkan klaster lebih kompak dan terpisah)",
        )
        st.metric(
            label=dbi_label,
            value=f"{ev_row['davies_bouldin_index']:.4f}",
            delta=dbi_delta,
            delta_color="off",
            delta_arrow="off",
            help="Davies-Bouldin Index: non-negatif (skor lebih rendah menunjukkan struktur klaster lebih optimal)",
        )
        if not is_all:
            db_row = df_dbscan_sum[df_dbscan_sum["nama_rayon"] == selected_rayon].iloc[0]
            st.caption(
                f"**DBSCAN ε:** `{db_row['eps_digunakan_m']:.2f} m`\n\n"
                f"**Noise:** `{int(db_row['jumlah_noise'])} titik` ({db_row['persen_noise']:.2f}%)\n\n"
                f"**Non-Noise:** `{int(db_row['jumlah_non_noise']):,} titik`"
            )
        else:
            tot_db = df_dbscan_sum[df_dbscan_sum["id_rayon"] == "TOTAL"].iloc[0]
            st.caption(
                f"**Total Non-Noise:** `{int(tot_db['jumlah_non_noise']):,} titik`\n\n"
                f"**Total Noise:** `{int(tot_db['jumlah_noise'])} titik` ({tot_db['persen_noise']:.2f}%)\n\n"
                f"**Total Centroid:** `31 Titik Kandidat`"
            )

    with col_table:
        cen_filtered = gdf_centroid if is_all else gdf_centroid[gdf_centroid["nama_rayon"] == selected_rayon]
        cen_filtered = cen_filtered.sort_values("peringkat_jarak_terjauh").reset_index(drop=True)

        st.subheader(f"📋 Tabel Kandidat Lokasi Rumah Pompa (Centroid K-Means) — {selected_rayon}")

        cols_display = [
            "peringkat_jarak_terjauh", "centroid_id", "nama_rayon", "jumlah_titik",
            "latitude", "longitude", "nama_pompa_aktif_terdekat", "jarak_pompa_aktif_terdekat_m",
        ]
        df_disp = cen_filtered[cols_display].rename(columns={
            "peringkat_jarak_terjauh": "Peringkat",
            "centroid_id": "ID Centroid",
            "nama_rayon": "Rayon",
            "jumlah_titik": "Titik Anggota",
            "latitude": "Lat (WGS84)",
            "longitude": "Lng (WGS84)",
            "nama_pompa_aktif_terdekat": "Pompa Aktif Terdekat",
            "jarak_pompa_aktif_terdekat_m": "Jarak Euclidean (m)",
        })
        st.dataframe(df_disp, width="stretch", hide_index=True)

        # Tombol Unduh CSV & GeoJSON (mengikuti filter rayon aktif)
        slug = "semua_7_rayon" if is_all else selected_rayon.lower().replace(" ", "_")
        csv_bytes = cen_filtered.drop(columns="geometry").to_csv(index=False).encode("utf-8")
        geojson_bytes = cen_filtered.to_json(indent=2).encode("utf-8")

        col_dl1, col_dl2, _ = st.columns([1.5, 1.6, 3.5])
        col_dl1.download_button(
            label="📥 Unduh CSV",
            data=csv_bytes,
            file_name=f"centroid_kandidat_{slug}.csv",
            mime="text/csv",
            width="stretch",
        )
        col_dl2.download_button(
            label="📥 Unduh GeoJSON",
            data=geojson_bytes,
            file_name=f"centroid_kandidat_{slug}.geojson",
            mime="application/geo+json",
            width="stretch",
        )

    # --- DOKUMENTASI & ANALISIS PENDUKUNG LANJUTAN (EXPANDER RAPI DI BAWAH) ---
    st.markdown("---")
    with st.expander("ℹ️ Lihat Dokumentasi & Analisis Pendukung Lanjutan", expanded=False):
        subtab1, subtab2, subtab3 = st.tabs([
            "Rekapitulasi Parameter DBSCAN & K-Means (7 Rayon)",
            "Konteks 12 Pompa Pengembangan Baru (Terpisah)",
            "Galeri Grafik Hasil Penelitian",
        ])

        with subtab1:
            st.markdown("##### Rekapitulasi Parameter DBSCAN, K-Means++, dan Evaluasi (7 Rayon)")
            merged_sum = (
                df_dbscan_sum[df_dbscan_sum["id_rayon"] != "TOTAL"][[
                    "id_rayon", "nama_rayon", "jumlah_titik", "eps_digunakan_m",
                    "jumlah_klaster_dbscan", "jumlah_core", "jumlah_border", "jumlah_noise", "persen_noise",
                ]]
                .merge(
                    df_kmeans_sum[df_kmeans_sum["id_rayon"] != "TOTAL"][[
                        "id_rayon", "k_digunakan", "wcss_terpilih_1e9_m2",
                    ]],
                    on="id_rayon",
                )
                .merge(
                    df_eval_rayon[~df_eval_rayon["id_rayon"].isin(["MEAN", "W-MEAN"])][[
                        "id_rayon", "silhouette_score", "kategori_silhouette", "davies_bouldin_index",
                    ]],
                    on="id_rayon",
                )
                .merge(
                    df_jarak_sum[df_jarak_sum["id_rayon"] != "ALL"][[
                        "id_rayon", "jumlah_pompa_aktif_di_rayon", "jarak_min_m", "jarak_rerata_m", "jarak_max_m",
                    ]],
                    on="id_rayon",
                )
            )
            merged_sum_disp = merged_sum.rename(columns={
                "id_rayon": "ID Rayon",
                "nama_rayon": "Rayon",
                "jumlah_titik": "Titik Bersih",
                "eps_digunakan_m": "ε DBSCAN (m)",
                "jumlah_klaster_dbscan": "Klaster DBSCAN",
                "jumlah_core": "Core",
                "jumlah_border": "Border",
                "jumlah_noise": "Noise",
                "persen_noise": "% Noise",
                "k_digunakan": "K Terpilih",
                "wcss_terpilih_1e9_m2": "WCSS (10⁹ m²)",
                "silhouette_score": "Silhouette Score",
                "kategori_silhouette": "Kategori SC",
                "davies_bouldin_index": "DBI",
                "jumlah_pompa_aktif_di_rayon": "Pompa Aktif di Rayon",
                "jarak_min_m": "Jarak Min (m)",
                "jarak_rerata_m": "Rerata Jarak (m)",
                "jarak_max_m": "Jarak Maks (m)",
            })
            st.dataframe(merged_sum_disp, width="stretch", hide_index=True)

            st.markdown(f"##### Detail Silhouette per Subcluster — {selected_rayon}")
            sub_eval_view = df_eval_sub if is_all else df_eval_sub[df_eval_sub["nama_rayon"] == selected_rayon]
            sub_eval_disp = sub_eval_view.rename(columns={
                "centroid_id": "ID Centroid",
                "id_rayon": "ID Rayon",
                "nama_rayon": "Rayon",
                "kmeans_cluster": "Subcluster",
                "jumlah_titik": "Titik Anggota",
                "sil_mean": "Rerata Silhouette",
                "sil_median": "Median Silhouette",
                "sil_min": "Min Silhouette",
                "sil_max": "Maks Silhouette",
                "jumlah_sil_negatif": "Jumlah Sil < 0",
                "persen_sil_negatif": "% Sil < 0",
            })
            st.dataframe(sub_eval_disp, width="stretch", hide_index=True)

        with subtab2:
            st.markdown("##### Analisis Sekunder Terpisah: Konteks 12 Lokasi Rumah Pompa Pengembangan Baru")
            st.caption(
                "Layer 12 lokasi `Pengembangan Baru` **tidak dicampur** ke perhitungan jarak utama 71 rumah pompa aktif. "
                "Tabel berikut sekadar memberi konteks spasial mengenai jarak centroid ke lokasi pengembangan terdekat."
            )
            dev_view = df_dev_ctx if is_all else df_dev_ctx[df_dev_ctx["nama_rayon"] == selected_rayon]
            dev_sorted = dev_view.sort_values("selisih_jarak_aktif_minus_kembang_m", ascending=False).reset_index(drop=True)
            dev_disp = dev_sorted.rename(columns={
                "centroid_id": "ID Centroid",
                "nama_rayon": "Rayon Centroid",
                "jarak_pompa_aktif_terdekat_m": "Jarak Pompa Aktif (m)",
                "id_pompa_aktif_terdekat": "ID Pompa Aktif Terdekat",
                "id_pengembangan_terdekat": "ID Pompa Pengembangan",
                "nama_pengembangan_terdekat": "Pompa Pengembangan Terdekat",
                "rayon_pengembangan_terdekat": "Rayon Pengembangan",
                "jarak_pengembangan_terdekat_m": "Jarak ke Pengembangan (m)",
                "pengembangan_lebih_dekat_dari_aktif": "Lebih Dekat ke Pengembangan?",
                "selisih_jarak_aktif_minus_kembang_m": "Selisih Jarak (Aktif - Kembang) (m)",
            })
            st.dataframe(dev_disp, width="stretch", hide_index=True)

        with subtab3:
            st.markdown("##### Galeri Grafik Hasil Penelitian (`outputs/figures/`)")
            gallery = [
                ("Tahap 3 — Grafik k-Distance (k=4) & Epsilon Kneedle per Rayon", FIGURES_DIR / "03_kdistance_kneedle_7_rayon.png"),
                ("Tahap 4 — Grafik Elbow Method (K=2..10) & K Terpilih per Rayon", FIGURES_DIR / "04_elbow_method_7_rayon.png"),
                ("Tahap 5 — Silhouette Plot per Subcluster (7 Rayon)", FIGURES_DIR / "05_silhouette_plot_7_rayon.png"),
                ("Tahap 5 — Komparasi SC & DBI pada Rentang K=2..10", FIGURES_DIR / "05_komparasi_k2_k10_sc_dbi.png"),
                ("Tahap 6 — Peta Garis Penghubung (Spider Map) ke 71 Pompa Aktif", FIGURES_DIR / "06_peta_spider_jarak_pompa.png"),
                ("Tahap 6 — Peringkat & Distribusi Jarak 31 Centroid", FIGURES_DIR / "06_distribusi_jarak_centroid_pompa.png"),
            ]
            for title, path in gallery:
                if path.exists():
                    with st.expander(title, expanded=False):
                        st.image(str(path), width="stretch")

    # --- DISCLAIMER AKADEMIS ---
    st.markdown("---")
    st.warning(
        "**Batasan Interpretasi Penelitian:** Titik centroid yang ditampilkan merupakan **kandidat awal berbasis pola spasial** "
        "titik inventarisasi saluran drainase (DBSCAN + K-Means++), dan jarak ke rumah pompa aktif merupakan **jarak lurus Euclidean "
        "(*straight-line distance*)** dalam satuan meter pada proyeksi UTM Zona 49S (`EPSG:32749`). Hasil ini **bukan** keputusan final "
        "pembangunan rumah pompa, bukan jarak sepanjang jaringan saluran, serta bukan kajian kapasitas hidraulis, elevasi lahan, atau "
        "kelayakan konstruksi — sehingga tetap memerlukan verifikasi teknis lebih lanjut oleh instansi terkait."
    )


if __name__ == "__main__":
    main()
