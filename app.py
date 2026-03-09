"""
app.py
------
Landslide Detection System - Satellite imagery analysis with map overlay.

Usage: streamlit run app.py
"""

import os
import tempfile
import numpy as np
import streamlit as st
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import torch
import torch.nn as nn
from PIL import Image
import io
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import base64
from scipy import ndimage

# ─── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Landslide Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.1);
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 8px;
        padding: 10px 14px;
    }
    .section-divider {
        border: none;
        border-top: 2px solid rgba(128,128,128,0.3);
        margin: 1.5rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Model ───────────────────────────────────────────────────────────────────


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_filters=32):
        super().__init__()
        self.enc1 = self._block(n_channels, base_filters)
        self.enc2 = self._block(base_filters, base_filters * 2)
        self.enc3 = self._block(base_filters * 2, base_filters * 4)
        self.enc4 = self._block(base_filters * 4, base_filters * 8)
        self.bottleneck = self._block(base_filters * 8, base_filters * 16)
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = self._block(base_filters * 16, base_filters * 8)
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._block(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._block(base_filters * 2, base_filters)
        self.out = nn.Conv2d(base_filters, n_classes, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _block(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)


class PatchNormalizer:
    def __init__(self, means, stds):
        self.means = means
        self.stds = np.where(stds < 1e-6, 1.0, stds)

    def normalize(self, features):
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        m = self.means[:, None, None]
        s = self.stds[:, None, None]
        return np.clip((features - m) / s, -10.0, 10.0).astype(np.float32)


# ─── Model Loading ───────────────────────────────────────────────────────────


@st.cache_resource
def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None, None, None
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = UNet(cfg.get("N_CHANNELS", 19), 1, cfg.get("BASE_FILTERS", 32))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    norm = PatchNormalizer(ckpt["normalizer_means"], ckpt["normalizer_stds"])
    return model, norm, ckpt.get("val_metrics", {})


# ─── Inference ───────────────────────────────────────────────────────────────


def predict_image(model, normalizer, image_data, patch_size=128, progress_cb=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _, h, w = image_data.shape
    prob = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    stride = patch_size // 2
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    total = len(ys) * len(xs)
    if total == 0:
        return prob
    done = 0
    with torch.no_grad():
        for y in ys:
            for x in xs:
                p = image_data[:, y:y + patch_size, x:x + patch_size]
                if np.isnan(p).sum() / p.size > 0.5:
                    done += 1
                    continue
                p = normalizer.normalize(p)
                t = torch.from_numpy(p).unsqueeze(0).to(device)
                out = torch.sigmoid(model(t)).squeeze().cpu().numpy()
                prob[y:y + patch_size, x:x + patch_size] += out
                cnt[y:y + patch_size, x:x + patch_size] += 1
                done += 1
                if progress_cb and done % 10 == 0:
                    progress_cb(min(done / total, 1.0))
    if progress_cb:
        progress_cb(1.0)
    return prob / np.maximum(cnt, 1)


# ─── Visualization Helpers ───────────────────────────────────────────────────


def _to_b64_png(arr):
    mode = "RGBA" if arr.shape[2] == 4 else "RGB"
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def extract_rgb(image_data):
    nb = image_data.shape[0]
    if nb >= 4:
        rgb = np.stack([image_data[3], image_data[2], image_data[1]], axis=0)
    elif nb >= 3:
        rgb = image_data[:3]
    else:
        rgb = np.stack([image_data[0]] * 3, axis=0)
    rgb = np.nan_to_num(rgb, nan=0.0)
    out = np.zeros((rgb.shape[1], rgb.shape[2], 3), dtype=np.uint8)
    for c in range(3):
        b = rgb[c]
        lo, hi = np.percentile(b, 2), np.percentile(b, 98)
        if hi - lo < 1e-6:
            hi = lo + 1.0
        out[:, :, c] = np.clip((b - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return out


def make_segmentation_rgba(prob_map, threshold, alpha):
    """Three-tier colored overlay: yellow / orange / red with per-pixel alpha."""
    h, w = prob_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = prob_map >= threshold
    if not mask.any():
        return rgba

    probs = prob_map[mask]
    idx = np.where(mask)

    low = probs < 0.6
    mid = (probs >= 0.6) & (probs < 0.8)
    high = probs >= 0.8

    rgba[idx[0][low], idx[1][low]] = [255, 230, 0, alpha]
    rgba[idx[0][mid], idx[1][mid]] = [255, 120, 0, min(alpha + 40, 255)]
    rgba[idx[0][high], idx[1][high]] = [255, 0, 0, min(alpha + 80, 255)]

    return rgba


def make_rgb_visualization(prob_map, threshold):
    h, w = prob_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask = prob_map >= threshold
    if not mask.any():
        return rgb
    probs = prob_map[mask]
    idx = np.where(mask)
    low = probs < 0.6
    mid = (probs >= 0.6) & (probs < 0.8)
    high = probs >= 0.8
    rgb[idx[0][low], idx[1][low]] = [255, 230, 0]
    rgb[idx[0][mid], idx[1][mid]] = [255, 120, 0]
    rgb[idx[0][high], idx[1][high]] = [255, 0, 0]
    return rgb


# ─── Cluster Detection ───────────────────────────────────────────────────────


def find_landslide_clusters(prob_map, bounds_wgs84, threshold, min_cluster_pixels=50):
    """
    Find connected landslide clusters and return a list of dicts with:
    - lat, lon (center of cluster in WGS84)
    - area_pixels, mean_prob, max_prob
    - risk_level (Low / Medium / High / Critical)
    """
    south, west = bounds_wgs84[1], bounds_wgs84[0]
    north, east = bounds_wgs84[3], bounds_wgs84[2]
    h, w = prob_map.shape

    binary = (prob_map >= threshold).astype(np.int32)
    labeled, num_features = ndimage.label(binary)

    clusters = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_cluster_pixels:
            continue

        # Pixel center of mass
        cy = ys.mean()
        cx = xs.mean()

        # Convert pixel coords to WGS84
        lat = north - (cy / h) * (north - south)
        lon = west + (cx / w) * (east - west)

        probs_in_cluster = prob_map[labeled == i]
        mean_p = float(probs_in_cluster.mean())
        max_p = float(probs_in_cluster.max())
        area = int(len(ys))

        if max_p >= 0.85:
            risk = "Critical"
        elif max_p >= 0.7:
            risk = "High"
        elif max_p >= 0.55:
            risk = "Medium"
        else:
            risk = "Low"

        clusters.append({
            "lat": float(lat),
            "lon": float(lon),
            "area_pixels": area,
            "mean_prob": mean_p,
            "max_prob": max_p,
            "risk": risk,
        })

    # Sort by max probability descending
    clusters.sort(key=lambda c: c["max_prob"], reverse=True)

    return clusters


# ─── Map Builder ─────────────────────────────────────────────────────────────


RISK_COLORS = {
    "Critical": "#ff0000",
    "High": "#ff6600",
    "Medium": "#ffaa00",
    "Low": "#ffdd00",
}

RISK_ICONS = {
    "Critical": "exclamation-triangle",
    "High": "exclamation-circle",
    "Medium": "info-circle",
    "Low": "info-sign",
}

RISK_ICON_COLORS = {
    "Critical": "red",
    "High": "orange",
    "Medium": "orange",
    "Low": "beige",
}


def build_map(bounds_wgs84, seg_rgba, clusters):
    """
    Build a map with:
    1. Satellite basemap (real imagery)
    2. Transparent segmentation heatmap overlay
    3. Pin markers at every landslide cluster with popups
    4. Pulsing circles around critical/high risk zones
    """
    south, west = bounds_wgs84[1], bounds_wgs84[0]
    north, east = bounds_wgs84[3], bounds_wgs84[2]
    clat = (south + north) / 2
    clon = (west + east) / 2

    span = max(abs(north - south), abs(east - west))
    if span > 5:
        zoom = 7
    elif span > 1:
        zoom = 9
    elif span > 0.1:
        zoom = 12
    elif span > 0.01:
        zoom = 14
    else:
        zoom = 16

    m = folium.Map(location=[clat, clon], zoom_start=zoom, tiles=None)

    # Satellite basemap
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map",
        overlay=False,
        control=True,
    ).add_to(m)

    image_bounds = [[south, west], [north, east]]

    # --- Segmentation heatmap overlay (transparent) ---
    seg_b64 = _to_b64_png(seg_rgba)
    folium.raster_layers.ImageOverlay(
        image=seg_b64,
        bounds=image_bounds,
        opacity=1.0,
        name="Heatmap Overlay",
        show=True,
        interactive=False,  # clicks pass through to markers below
    ).add_to(m)

    # --- White glow border around detected zones ---
    binary = (seg_rgba[:, :, 3] > 0).astype(np.uint8)
    dilated = ndimage.binary_dilation(binary, iterations=3).astype(np.uint8)
    border = dilated - binary
    border_rgba = np.zeros_like(seg_rgba)
    border_rgba[border == 1] = [255, 255, 255, 180]
    border_b64 = _to_b64_png(border_rgba)
    folium.raster_layers.ImageOverlay(
        image=border_b64,
        bounds=image_bounds,
        opacity=1.0,
        name="Zone Outlines",
        show=True,
        interactive=False,  # clicks pass through to markers below
    ).add_to(m)

    # --- Pin markers at each landslide cluster ---
    marker_group = folium.FeatureGroup(name="Landslide Markers", show=True)

    for idx, cl in enumerate(clusters):
        color = RISK_COLORS.get(cl["risk"], "#ffdd00")
        icon_name = RISK_ICONS.get(cl["risk"], "info-sign")
        icon_color = RISK_ICON_COLORS.get(cl["risk"], "beige")

        # Popup with details
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 180px;">
            <h4 style="margin:0 0 6px 0; color:{color}; border-bottom:2px solid {color}; padding-bottom:4px;">
                Zone {idx + 1} — {cl['risk']} Risk
            </h4>
            <table style="font-size:13px; width:100%;">
                <tr><td><b>Latitude</b></td><td>{cl['lat']:.5f}</td></tr>
                <tr><td><b>Longitude</b></td><td>{cl['lon']:.5f}</td></tr>
                <tr><td><b>Max Probability</b></td><td>{cl['max_prob']:.1%}</td></tr>
                <tr><td><b>Mean Probability</b></td><td>{cl['mean_prob']:.1%}</td></tr>
                <tr><td><b>Area</b></td><td>{cl['area_pixels']:,} px</td></tr>
            </table>
        </div>
        """

        # Pin marker
        folium.Marker(
            location=[cl["lat"], cl["lon"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Zone {idx + 1}: {cl['risk']} ({cl['max_prob']:.0%})",
            icon=folium.Icon(
                color=icon_color,
                icon=icon_name,
                prefix="fa" if icon_name != "info-sign" else "glyphicon",
            ),
        ).add_to(marker_group)

        # Pulsing circle for high/critical zones
        if cl["risk"] in ("Critical", "High"):
            folium.CircleMarker(
                location=[cl["lat"], cl["lon"]],
                radius=18 if cl["risk"] == "Critical" else 14,
                color=color,
                weight=3,
                fill=True,
                fill_color=color,
                fill_opacity=0.15,
                tooltip=f"Zone {idx + 1}: {cl['risk']}",
            ).add_to(marker_group)

        # Large translucent ring to highlight the area
        folium.CircleMarker(
            location=[cl["lat"], cl["lon"]],
            radius=25 if cl["risk"] == "Critical" else 20 if cl["risk"] == "High" else 14,
            color=color,
            weight=2,
            fill=False,
            dash_array="5",
        ).add_to(marker_group)

    marker_group.add_to(m)

    # Image extent border
    folium.Rectangle(
        bounds=image_bounds,
        color="#ffffff",
        weight=1.5,
        dash_array="6",
        fill=False,
        name="Image Extent",
    ).add_to(m)

    folium.LatLngPopup().add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Raise markers to top of DOM so overlays never block clicks
    m.get_root().html.add_child(folium.Element(
        "<script>document.addEventListener('DOMContentLoaded',function(){"
        "document.querySelectorAll('.leaflet-marker-pane,.leaflet-popup-pane')"
        ".forEach(function(el){el.style.zIndex=1000;});});</script>"
    ))

    return m


# ─── Sidebar ─────────────────────────────────────────────────────────────────


def render_sidebar():
    with st.sidebar:
        st.header("Configuration")

        checkpoint_path = st.text_input(
            "Model checkpoint path",
            value="checkpoints/best_model_smart.pth",
        )

        st.divider()
        st.subheader("Detection")

        threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05,
                              help="Probability cutoff for detection.")

        downsample = st.selectbox("Resolution", [1, 2, 4, 8], index=1,
                                  format_func=lambda x: "Full" if x == 1 else f"1/{x}",
                                  help="Higher = faster, lower detail.")

        overlay_alpha = st.slider("Overlay transparency", 0, 255, 140, 5,
                                  help="0 = invisible, 255 = fully opaque.")

        min_cluster = st.slider("Min cluster size (px)", 10, 500, 50, 10,
                                help="Ignore clusters smaller than this.")

    return {
        "checkpoint_path": checkpoint_path,
        "threshold": threshold,
        "downsample": downsample,
        "overlay_alpha": overlay_alpha,
        "min_cluster": min_cluster,
    }


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    st.title("Landslide Detection System")
    st.caption("Upload satellite GeoTIFF. Detection runs automatically. "
               "Each landslide zone gets a pin marker on the real map.")

    cfg = render_sidebar()

    model, normalizer, metrics = load_model(cfg["checkpoint_path"])

    if model is None:
        st.error(f"Model not found: {cfg['checkpoint_path']}")
        st.info("Train a model first and set the correct path.")
        return

    if metrics:
        with st.sidebar:
            st.divider()
            st.subheader("Model Metrics")
            c1, c2 = st.columns(2)
            c1.metric("IoU", f"{metrics.get('iou', 0):.4f}")
            c2.metric("F1", f"{metrics.get('f1', 0):.4f}")
            c1.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            c2.metric("Recall", f"{metrics.get('recall', 0):.4f}")

    uploaded = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"],
                                help="Multi-band GeoTIFF with georeferencing.")

    if uploaded is None:
        st.info("Upload a GeoTIFF file to start.")
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        tmp.write(uploaded.read())
        tmp.close()
        temp_path = tmp.name

        with rasterio.open(temp_path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            ow, oh, nb = src.width, src.height, src.count
            ds = cfg["downsample"]
            if ds > 1:
                nw, nh = ow // ds, oh // ds
                data = src.read(out_shape=(nb, nh, nw),
                                resampling=Resampling.bilinear).astype(np.float32)
            else:
                nw, nh = ow, oh
                data = src.read().astype(np.float32)

        # Summary
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Size", f"{ow} x {oh}")
        c2.metric("Bands", str(nb))
        c3.metric("Processing", f"{nw} x {nh}")
        clat = (bounds[1] + bounds[3]) / 2
        clon = (bounds[0] + bounds[2]) / 2
        c4.metric("Center", f"{clat:.4f}, {clon:.4f}")

        # Inference
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        bar = st.progress(0, text="Running detection...")
        prob_map = predict_image(
            model, normalizer, data,
            progress_cb=lambda f: bar.progress(min(f, 1.0), text=f"Detecting... {f * 100:.0f}%"),
        )
        bar.empty()
        st.success("Detection complete.")

        # Stats
        total = prob_map.size
        det = int(np.sum(prob_map >= cfg["threshold"]))
        high = int(np.sum(prob_map >= 0.75))

        m1, m2, m3 = st.columns(3)
        m1.metric("Detected area", f"{100 * det / total:.2f}%")
        m2.metric("High-risk area", f"{100 * high / total:.2f}%")
        m3.metric("Peak probability", f"{prob_map.max():.3f}")

        # Find landslide clusters
        clusters = find_landslide_clusters(
            prob_map, bounds, cfg["threshold"], cfg["min_cluster"],
        )

        # Cluster summary table
        if clusters:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader(f"Detected Landslide Zones: {len(clusters)}")

            for i, cl in enumerate(clusters):
                risk = cl["risk"]
                color = RISK_COLORS[risk]
                st.markdown(
                    f'<span style="color:{color}; font-weight:bold;">Zone {i + 1}</span> '
                    f'— **{risk}** risk | '
                    f'Max: {cl["max_prob"]:.1%} | '
                    f'Mean: {cl["mean_prob"]:.1%} | '
                    f'Area: {cl["area_pixels"]:,} px | '
                    f'Location: ({cl["lat"]:.5f}, {cl["lon"]:.5f})',
                    unsafe_allow_html=True,
                )

        # Build overlays
        seg_rgba = make_segmentation_rgba(prob_map, cfg["threshold"], cfg["overlay_alpha"])

        # Map
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Map — Landslide Locations")
        st.caption("Pin markers show each detected landslide zone. "
                   "Click a pin for details. Toggle layers with the control panel.")

        detection_map = build_map(bounds, seg_rgba, clusters)
        st_folium(detection_map, use_container_width=True, height=650)

        # Visual results
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Visual Results")
        rgb_preview = extract_rgb(data)
        rgb_det = make_rgb_visualization(prob_map, cfg["threshold"])

        ca, cb, cc = st.columns(3)
        with ca:
            st.caption("Satellite RGB")
            st.image(rgb_preview, use_container_width=True)
        with cb:
            st.caption("Probability map")
            st.image(prob_map, clamp=True, use_container_width=True)
        with cc:
            st.caption("Detection mask")
            st.image(rgb_det, use_container_width=True)

        # Downloads
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.subheader("Downloads")
        d1, d2 = st.columns(2)

        png_buf = io.BytesIO()
        Image.fromarray((prob_map * 255).astype(np.uint8)).save(png_buf, format="PNG")
        with d1:
            st.download_button("Download probability map (PNG)",
                               png_buf.getvalue(), "landslide_probability.png",
                               "image/png", use_container_width=True)

        with rasterio.open(temp_path) as src:
            meta = src.meta.copy()
            meta.update(count=1, dtype="float32")
            otmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            otmp.close()
            with rasterio.open(otmp.name, "w", **meta) as dst:
                dst.write(prob_map, 1)
            with open(otmp.name, "rb") as f:
                geo_bytes = f.read()
            os.remove(otmp.name)

        with d2:
            st.download_button("Download detection GeoTIFF",
                               geo_bytes, "landslide_detection.tif",
                               "application/octet-stream", use_container_width=True)

    except rasterio.errors.RasterioIOError as e:
        st.error(f"Invalid GeoTIFF: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except OSError:
                pass


if __name__ == "__main__":
    main()