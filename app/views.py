"""Page functions for the HYDRA-RAG dashboard with eager pre-warming and telemetry caching."""
import numpy as np
import pandas as pd
import streamlit as st

try:
    from app.ui_helpers import (
        JUDGE_FREE,
        TARGETS,
        card,
        get_index_size,
        grouped_bar,
        latency_chart,
        load_categories,
        load_phase,
        page_header,
    )
except ModuleNotFoundError:
    from ui_helpers import (
        JUDGE_FREE,
        TARGETS,
        card,
        get_index_size,
        grouped_bar,
        latency_chart,
        load_categories,
        load_phase,
        page_header,
    )

FULL_CORPUS = 100_000
PHASE_OF_MODE = {"dense": "Phase 1", "hybrid": "Phase 2"}


@st.cache_resource(show_spinner=False)
def warmup_retrieval_pipeline():
    """Eagerly load transformer checkpoints, sparse tokenizers, and Qdrant socket pools."""
    try:
        from retrieval.dense import search
        search(query="system warm up signal", mode="dense", top_k=1)
        search(query="system warm up signal", mode="hybrid", top_k=1)
        return True
    except Exception as exc:
        return False


@st.cache_data(show_spinner=False)
def _cached_search(query: str, mode: str, category: str):
    """Execute vector retrieval and memoize exact output payloads and latency telemetry."""
    from retrieval.dense import search
    return search(
        query=query,
        mode=mode,
        top_k=5,
        filters=None if category == "All" else {"category": category},
    )


def _tile(column, label, value, delta=None, delta_color="normal"):
    with column, st.container(border=True):
        st.metric(label, value, delta, delta_color=delta_color)


def _run_search(query: str, mode: str, category: str):
    hits = _cached_search(query=query, mode=mode, category=category)
    if hits:
        history = st.session_state.setdefault("history", {"Phase 1": [], "Phase 2": []})
        phase_label = PHASE_OF_MODE[mode]
        total_latency = hits[0].get("timings", {}).get("total_ms", 0.0)

        # Prevent unbounded latency metric duplication during mode toggling
        tracked_queries = st.session_state.setdefault("tracked_queries", set())
        query_sig = (query.strip().lower(), mode, category)
        if query_sig not in tracked_queries:
            history[phase_label].append(total_latency)
            tracked_queries.add(query_sig)

    return hits


def _render_hits(hits, compact=False, other_ids=None):
    timings = hits[0].get("timings", {})
    line = (
        f"Encoding {timings.get('embed_ms', 0):.1f} ms, Qdrant {timings.get('search_ms', 0):.1f} ms, "
        f"total {timings.get('total_ms', 0):.1f} ms"
    )
    if not compact:
        a, b, c = st.columns(3)
        _tile(a, "Query encoding", f"{timings.get('embed_ms', 0):.1f} ms")
        _tile(b, "Qdrant search", f"{timings.get('search_ms', 0):.1f} ms")
        _tile(c, "Total latency", f"{timings.get('total_ms', 0):.1f} ms")
    else:
        st.caption(line)
    for rank, hit in enumerate(hits, 1):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            tag = " &nbsp; *new in hybrid*" if other_ids is not None and hit["id"] not in other_ids else ""
            left.markdown(f"**#{rank}** &nbsp; {hit.get('source', 'unknown source')}{tag}")
            score = hit.get("score")
            right.markdown(
                f"<div class='score'>{score:.4f}</div>" if score is not None else "fused rank",
                unsafe_allow_html=True,
            )
            st.write(hit["text"])
            st.caption(f"Passage {hit['id']} in category {hit.get('category', 'n/a')}")


def page_search():
    page_header("Search", "Ask a question and see the top 5 passages with scores and timings.")
    c_query, c_mode, c_cat = st.columns([4, 2, 1.5])
    query = c_query.text_input("Question", placeholder="what is a corporation?")
    choice = c_mode.radio("Retrieval mode", ["Dense", "Hybrid", "Compare"], horizontal=True)
    category = c_cat.selectbox("Category", ["All"] + load_categories())

    if not query:
        st.info("Type a question to see the top 5 passages, their scores and the time each step took.")
        return
    try:
        with st.spinner("Searching..."):
            if choice == "Compare":
                dense_hits = _run_search(query, "dense", category)
                hybrid_hits = _run_search(query, "hybrid", category)
            else:
                hits = _run_search(query, choice.lower(), category)
    except Exception as exc:
        st.error(f"Search failed: {exc}. Check that Qdrant is running with `docker compose up -d`.")
        return

    if choice == "Compare":
        if not (dense_hits or hybrid_hits):
            st.warning("No passages matched. Try removing the category filter.")
            return
        left, right = st.columns(2)
        with left:
            st.subheader("Dense (Phase 1)")
            if dense_hits:
                _render_hits(dense_hits, compact=True)
            else:
                st.warning("No dense results.")
        with right:
            st.subheader("Hybrid (Phase 2)")
            if hybrid_hits:
                _render_hits(hybrid_hits, compact=True, other_ids={h["id"] for h in dense_hits})
            else:
                st.warning("No hybrid results.")
        st.caption("Scores differ in kind: dense shows cosine similarity, hybrid shows a fused rank score.")
    elif not hits:
        st.warning("No passages matched. Try removing the category filter.")
    else:
        _render_hits(hits)
        if choice == "Hybrid":
            st.caption("Hybrid scores are fused rank scores, so they are not comparable with dense cosine scores.")


def _comparison_table(phases, selected):
    spec = [
        ("Context precision", "precision", "{:.3f}", TARGETS["precision"], True),
        ("Context recall", "recall", "{:.3f}", TARGETS["recall"], True),
        ("p50 latency (ms)", "p50", "{:.1f}", None, False),
        ("p95 latency (ms)", "p95", "{:.1f}", TARGETS["p95_ms"], False),
        ("p99 latency (ms)", "p99", "{:.1f}", None, False),
    ]
    rows = []
    for label, key, fmt, target, higher_is_better in spec:
        row = {"Metric": label}
        for name in selected:
            value = phases[name][key]
            row[name] = fmt.format(value) if value is not None else "n/a"
        first, last = phases[selected[0]][key], phases[selected[-1]][key]
        if len(selected) > 1:
            row["Change"] = fmt.replace("{:", "{:+").format(last - first) if None not in (first, last) else "n/a"
        row["Target"] = ("above " if higher_is_better else "under ") + fmt.format(target) if target else ""
        row["Status"] = ""
        if target and last is not None:
            row["Status"] = "Met" if (last > target if higher_is_better else last < target) else "Not yet"
        rows.append(row)
    return pd.DataFrame(rows)


def page_dashboard():
    head, action = st.columns([5, 1])
    with head:
        page_header("Dashboard", "Phase 1 is dense retrieval. Phase 2 is hybrid: dense plus BM25, fused with RRF.")
    action.button("Refresh results", type="primary")

    phases = {f"Phase {n}": load_phase(n) for n in (1, 2)}
    available = [name for name, data in phases.items() if data["has_data"]]
    if not available:
        st.info("No results yet. Run eval/run_ragas.py and eval/latency.py; they write JSON files to results/.")
        return
    size = get_index_size() or next((phases[n]["n_passages"] for n in available if phases[n]["n_passages"]), None)
    if size and size < FULL_CORPUS:
        st.warning(
            f"These results come from an index of {int(size):,} passages. The challenge needs "
            f"{FULL_CORPUS:,}+, so run the evaluation again once the full set is indexed."
        )
    selected = st.multiselect("Compare phases", available, default=available)
    if not selected:
        st.info("Pick at least one phase to show.")
        return
    latest, first = selected[-1], selected[0]
    cur, ref = phases[latest], phases[first]

    cols = st.columns(4)
    for col, (label, key) in zip(cols, [("p50", "p50"), ("p95", "p95"), ("p99", "p99"), ("Mean", "mean")]):
        value = cur[key]
        if value is None:
            _tile(col, f"{label} latency", "n/a")
        elif len(selected) > 1 and ref[key] is not None:
            _tile(col, f"{label} latency ({latest})", f"{value:.1f} ms",
                  f"{value - ref[key]:+.1f} ms vs {first}", delta_color="inverse")
        elif key == "p95":
            _tile(col, f"{label} latency ({latest})", f"{value:.1f} ms",
                  f"{TARGETS['p95_ms'] - value:+.0f} ms vs {TARGETS['p95_ms']:.0f} ms target")
        else:
            _tile(col, f"{label} latency ({latest})", f"{value:.1f} ms")

    left, right = st.columns(2)
    with left, card("Query latency", "One line per phase, dashed line is the 300 ms target"):
        series = {n: phases[n]["series"] for n in selected if phases[n]["series"]}
        if series:
            latency_chart(series, TARGETS["p95_ms"])
        else:
            st.info("Per-query latencies appear after eval/latency.py runs.")
    with right, card("Latency by stage", "Average milliseconds per step"):
        rows = [
            {"metric": s.capitalize(), "value": v, "phase": n}
            for n in selected for s, v in phases[n]["stages"].items() if v is not None
        ]
        if rows:
            grouped_bar(rows, fmt=".1f")
        else:
            st.info("Stage timings appear once eval/latency.py logs embed, search and fuse times.")

    left, right = st.columns(2)
    with left, card("RAGAS scores", "Context precision and recall, scale 0 to 1"):
        rows = [
            {"metric": label, "value": phases[n][key], "phase": n}
            for n in selected for label, key in [("Context precision", "precision"), ("Context recall", "recall")]
            if phases[n][key] is not None
        ]
        if rows:
            grouped_bar(rows, y_domain=[0, 1])
            st.caption(f"Targets: precision above {TARGETS['precision']}, recall above {TARGETS['recall']}")
        else:
            st.info("RAGAS scores appear after eval/run_ragas.py runs.")
    with right, card("Retrieval accuracy", "Checked against the labelled relevant passages, no LLM judge"):
        rows = [
            {"metric": label, "value": v, "phase": n}
            for n in selected for label, v in phases[n]["judge_free"].items() if v is not None
        ]
        if rows:
            grouped_bar(rows, y_domain=[0, 1])
        else:
            st.info(f"Add {', '.join(JUDGE_FREE)} to the RAGAS results file to show this chart.")

    with card("Scorecard", "Latest selected phase against the hackathon targets"):
        st.dataframe(_comparison_table(phases, selected), hide_index=True, use_container_width=True)

    history = st.session_state.get("history", {})
    runs = {name: values for name, values in history.items() if values}
    if runs:
        with card("This session", "Searches run in this browser tab"):
            st.caption(
                ", ".join(
                    f"{name}: p50 {np.percentile(v, 50):.1f} ms, p95 {np.percentile(v, 95):.1f} ms ({len(v)} searches)"
                    for name, v in runs.items()
                )
            )
            latency_chart(runs)