"""Theme, results loading and charts for the HYDRA-RAG dashboard.

The eval scripts write JSON files into results/ (phase1_latency.json,
phase1_ragas.json, phase2_latency.json, phase2_ragas.json). Their exact shape
may differ slightly, so lookups search nested keys by normalised name
(case, spaces and punctuation ignored) instead of assuming one layout.
"""
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

RESULTS_DIR = Path("results")
CATEGORIES_FILE = Path("data/categories.json")   # written by the Phase 2 k-means step
TARGETS = {"precision": 0.75, "recall": 0.70, "p95_ms": 300.0}
PHASE_COLORS = {"Phase 1": "#FF4D5E", "Phase 2": "#6C63F7"}
STAGES = ("embed", "search", "fuse")
JUDGE_FREE = {
    "Hit@5": ("hit@5", "hit_at_5", "hit_rate"),
    "Recall@5": ("recall@5", "recall_at_5"),
    "MRR": ("mrr", "mean_reciprocal_rank"),
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { padding-top: 2.5rem; max-width: 1280px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; border-color: #E5E7EB; }
[data-testid="stSidebar"] { border-right: 1px solid #E5E7EB; }
[data-testid="stSidebarNav"] a[aria-current="page"],
a[data-testid="stSidebarNavLink"][aria-current="page"] { background: #EEF0FF; color: #4F46E5; font-weight: 500; }
.page-title { font-size: 1.75rem; font-weight: 600; color: #1F2328; margin: 0; }
.page-sub { color: #6B7280; margin: 0.25rem 0 1.25rem 0; }
.card-title { font-size: 1.05rem; font-weight: 600; color: #1F2328; }
.card-sub { color: #6B7280; font-size: 0.9rem; margin-bottom: 0.5rem; }
.score { text-align: right; font-weight: 600; color: #4F46E5; font-variant-numeric: tabular-nums; }
"""


def inject_css():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(
        f"<div class='page-title'>{title}</div><div class='page-sub'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


@contextmanager
def card(title, subtitle=""):
    """White bordered card with a title and a grey subtitle, like the reference design."""
    with st.container(border=True):
        st.markdown(
            f"<div class='card-title'>{title}</div><div class='card-sub'>{subtitle}</div>",
            unsafe_allow_html=True,
        )
        yield


# ---------- results loading ----------

def _norm(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)  # NaN counts as missing


def _walk(data):
    """Yield (normalised_key, value) for every key in nested JSON, shallowest first."""
    queue = [data]
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            for key, val in cur.items():
                yield _norm(key), val
            queue.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            queue.extend(v for v in cur if isinstance(v, (dict, list)))


def find_value(data, *names):
    wanted = {_norm(n) for n in names}
    for key, val in _walk(data):
        if key in wanted and _is_num(val):
            return float(val)
    return None


def find_series(data, *names):
    wanted = {_norm(n) for n in names}
    for key, val in _walk(data):
        if key in wanted and isinstance(val, list) and val and all(_is_num(x) for x in val):
            return [float(x) for x in val]
    return None


def _load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load_categories():
    """Unique category names. Reads data/categories.json or falls back to Qdrant collection payloads."""
    # 1. Attempt to load from the k-means generated JSON file
    data = _load_json(CATEGORIES_FILE)
    if isinstance(data, list) and data:
        names = {(r.get("category") if isinstance(r, dict) else r) for r in data}
        return sorted(str(n) for n in names if n)

    # 2. Fallback: Query distinct categories directly from Qdrant payloads
    try:
        import yaml
        from qdrant_client import QdrantClient
        cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["qdrant"]
        client = QdrantClient(host=cfg["host"], port=cfg["port"], timeout=5)

        # Scroll through the collection to extract distinct category payloads
        points, _ = client.scroll(
            collection_name=cfg["collection_name"],
            limit=10000,
            with_payload=["category"],
            with_vectors=False
        )
        cats = {p.payload.get("category") for p in points if p.payload and p.payload.get("category")}
        return sorted(str(c) for c in cats if c)

    except Exception as e:
        print(f"Category extraction failed: {e}")
        return []


@st.cache_data(ttl=30, show_spinner=False)
def get_index_size():
    """Number of passages currently in Qdrant, or None if Qdrant or config.yaml is unavailable."""
    try:
        import yaml
        from qdrant_client import QdrantClient
        cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["qdrant"]
        client = QdrantClient(host=cfg["host"], port=cfg["port"], timeout=5)
        return int(client.count(cfg["collection_name"]).count)
    except Exception:
        return None


def load_phase(n):
    """Everything the dashboard shows for one phase; missing values are None."""
    lat = _load_json(RESULTS_DIR / f"phase{n}_latency.json")
    rag = _load_json(RESULTS_DIR / f"phase{n}_ragas.json")
    series = find_series(lat, "latencies_ms", "latencies", "per_query_ms", "total_ms")

    def pct(q, *names):
        value = find_value(lat, *names)
        if value is None and series:
            value = float(np.percentile(series, q))
        return value

    mean = find_value(lat, "mean_ms", "mean", "avg_ms", "avg_total_ms")
    if mean is None and series:
        mean = float(np.mean(series))
    return {
        "has_data": lat is not None or rag is not None,
        "n_passages": find_value(lat, "n_passages") or find_value(rag, "n_passages"),
        "series": series,
        "p50": pct(50, "p50_ms", "p50", "p50_total_ms"),
        "p95": pct(95, "p95_ms", "p95", "p95_total_ms"),
        "p99": pct(99, "p99_ms", "p99", "p99_total_ms"),
        "mean": mean,
        "stages": {s: find_value(lat, f"{s}_ms", f"mean_{s}_ms", f"avg_{s}_ms", f"{s}_ms_mean", s) for s in STAGES},
        "precision": find_value(rag, "context_precision", "ragas_context_precision"),
        "recall": find_value(rag, "context_recall", "ragas_context_recall"),
        "judge_free": {label: find_value(rag, *names) for label, names in JUDGE_FREE.items()},
    }


# ---------- charts ----------

def _color():
    return alt.Color(
        "phase:N",
        scale=alt.Scale(domain=list(PHASE_COLORS), range=list(PHASE_COLORS.values())),
        legend=alt.Legend(orient="bottom", title=None),
    )


def _style(chart):
    return (
        chart.configure_axis(grid=True, gridColor="#EEF0F4", domain=False, tickColor="#EEF0F4",
                             labelColor="#6B7280", titleColor="#6B7280")
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor="#6B7280")
    )


def latency_chart(series_by_phase, target_ms=None):
    """Smooth line per phase, one point per query, with an optional dashed target line."""
    rows = [{"query": i + 1, "ms": v, "phase": name}
            for name, series in series_by_phase.items() for i, v in enumerate(series)]
    line = alt.Chart(pd.DataFrame(rows)).mark_line(interpolate="monotone", strokeWidth=2.5).encode(
        x=alt.X("query:Q", title="Query number"), y=alt.Y("ms:Q", title="Latency (ms)"), color=_color())
    layers = [line]
    if target_ms:
        layers.append(alt.Chart(pd.DataFrame({"y": [target_ms]})).mark_rule(
            strokeDash=[4, 4], color="#9CA3AF").encode(y="y:Q"))
    st.altair_chart(_style(alt.layer(*layers).properties(height=240)), use_container_width=True)


def grouped_bar(rows, fmt=".2f", y_domain=None):
    """Bars grouped by metric, one bar per phase, value printed above each bar."""
    y = alt.Y("value:Q", title=None, scale=alt.Scale(domain=y_domain)) if y_domain else alt.Y("value:Q", title=None)
    x = alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0))
    base = alt.Chart(pd.DataFrame(rows))
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=x, xOffset="phase:N", y=y, color=_color())
    labels = base.mark_text(dy=-8, fontSize=12, color="#374151").encode(
        x=x, xOffset="phase:N", y="value:Q", text=alt.Text("value:Q", format=fmt))
    st.altair_chart(_style(alt.layer(bars, labels).properties(height=240)), use_container_width=True)